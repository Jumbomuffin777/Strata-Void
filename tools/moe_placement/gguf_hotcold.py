"""Rewrite a MoE GGUF into the hot/cold expert layout a patched llama.cpp reads (ffn_*_exps_hot / _cold).

For every MoE layer the routed-expert tensors `blk.L.ffn_{gate,up,down}_exps.weight` ([in, out, n_expert]) are
replaced by
    blk.L.ffn_{gate,up,down}_exps_hot.weight    the hot experts in rank order, then ONE all-zero expert
    blk.L.ffn_{gate,up,down}_exps_cold.weight   the cold experts in rank order, then ONE all-zero expert
    blk.L.ffn_exps_map_hot.weight / _cold       I32 [1, n_expert]: canonical id -> index in that part, or the index
                                                of its zero expert when the other part holds it
Everything else (metadata, every other tensor) is copied byte for byte. The router is untouched, so canonical ids,
biases and hash tables stay valid; the zero expert gives an exact 0, so hot + cold equals the original per slot.

An all-zero block decodes to 0 in every ggml type used for experts here (each block's scale is its leading fp16 or
e8m0/fp16 field; zero scale -> zero weights, and MXFP4 code 0 is +0.0).

    python tools/moe_placement/gguf_hotcold.py MODEL-00001-of-0000N.gguf --cold cold.json --out MODEL-hotcold.gguf

`cold.json`: {"<layer>": [canonical expert ids held cold, ...], ...}; layers absent keep every expert hot.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moe_placement.descriptor import read_header, tensor_bytes, _EXP  # noqa: E402

GGML_I32 = 26


def _raw_kvs(path: Path) -> tuple[list[bytes], int]:
    """The metadata key/value records of one GGUF, as raw bytes (copied verbatim), minus split.* keys."""
    out, align = [], 32
    with open(path, "rb") as f:
        magic, ver, n_t, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != 0x46554747 or ver != 3:
            raise ValueError(f"{path}: not a GGUF v3 file")
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip_value(t):
            if t == 8:
                (n,) = struct.unpack("<Q", f.read(8)); f.seek(n, 1)
            elif t == 9:
                et, cnt = struct.unpack("<IQ", f.read(12))
                if et == 8:
                    for _ in range(cnt):
                        (n,) = struct.unpack("<Q", f.read(8)); f.seek(n, 1)
                else:
                    f.seek(sizes[et] * cnt, 1)
            else:
                f.seek(sizes[t], 1)
        for _ in range(n_kv):
            start = f.tell()
            (n,) = struct.unpack("<Q", f.read(8)); key = f.read(n).decode()
            (t,) = struct.unpack("<I", f.read(4))
            vstart = f.tell(); skip_value(t); end = f.tell()
            if key == "general.alignment":
                f.seek(vstart); align = struct.unpack("<I", f.read(4))[0]; f.seek(end)
            if key.startswith("split."):
                continue
            f.seek(start); out.append(f.read(end - start))
    return out, align


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def plan_tensors(h, cold: dict, type_ids: dict) -> list:
    """[(name, shape, type_id, nbytes, source)] where source is ('copy', file, offset) or ('parts', [...])."""
    out = []
    for name, shape, typ, fname, off in h.tensors:
        m = _EXP.match(name)
        if not m or str(int(m.group(1))) not in cold:
            out.append((name, shape, type_ids[typ], _nbytes(typ, shape), ("copy", fname, off)))
            continue
        layer, role = int(m.group(1)), m.group(2)
        n_expert = shape[-1]
        c = [int(e) for e in cold[str(layer)]]
        cs = set(c)
        if len(cs) != len(c) or not all(0 <= e < n_expert for e in c):
            raise ValueError(f"layer {layer}: cold list must be distinct ids in 0..{n_expert - 1}")
        hot = [e for e in range(n_expert) if e not in cs]
        slice_b = tensor_bytes(typ, shape[:-1])
        for part, ids in (("hot", hot), ("cold", c)):
            segs = [("copy", fname, off + e * slice_b, slice_b) for e in ids] + [("zero", slice_b)]
            out.append((f"blk.{layer}.ffn_{role}_exps_{part}.weight", list(shape[:-1]) + [len(ids) + 1],
                        type_ids[typ], slice_b * (len(ids) + 1), ("parts", segs)))
        if role == "gate":
            for part, ids in (("hot", hot), ("cold", c)):
                idx = {e: i for i, e in enumerate(ids)}
                table = [idx.get(e, len(ids)) for e in range(n_expert)]
                out.append((f"blk.{layer}.ffn_exps_map_{part}.weight", [1, n_expert], GGML_I32, 4 * n_expert,
                            ("bytes", struct.pack(f"<{n_expert}i", *table))))
    return out


def _nbytes(typ, shape) -> int:
    if typ in ("I32", "F32_I32"):
        n = 1
        for d in shape:
            n *= d
        return 4 * n
    return tensor_bytes(typ, shape)


def write(src_first: Path, cold: dict, out_path: Path, progress=print) -> None:
    from gguf_reader import GGML_TYPES
    h = read_header([src_first])
    type_ids = {v: k for k, v in GGML_TYPES.items()}
    kvs, align = _raw_kvs(h.paths[0])
    tensors = plan_tensors(h, cold, type_ids)
    head = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(kvs)) + b"".join(kvs)
    offs, pos = [], 0
    for _name, _shape, _t, nbytes, _src in tensors:
        offs.append(pos)
        pos += (nbytes + align - 1) // align * align
    for (name, shape, t, _n, _src), o in zip(tensors, offs):
        head += _s(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", t, o)
    head += b"\0" * ((-len(head)) % align)
    fds = {p.name: os.open(p, os.O_RDONLY) for p in h.paths}
    tmp = out_path.with_name(out_path.name + ".partial")
    with open(tmp, "wb") as f:
        f.write(head); f.flush()
        out = f.fileno()
        done, total = 0, pos
        for (name, _shape, _t, nbytes, src), o in zip(tensors, offs):
            want = len(head) + o
            if os.lseek(out, 0, os.SEEK_CUR) != want:
                os.write(out, b"\0" * (want - os.lseek(out, 0, os.SEEK_CUR)))
            segs = [("copy", src[1], src[2], nbytes)] if src[0] == "copy" else src[1] if src[0] == "parts" else [("raw", src[1])]
            for seg in segs:
                if seg[0] == "copy":
                    _, fname, soff, n = seg
                    left = n
                    while left:
                        k = os.copy_file_range(fds[fname], out, min(left, 1 << 30), soff + (n - left))
                        if k <= 0:
                            raise OSError(f"short copy for {name}")
                        left -= k
                elif seg[0] == "zero":
                    os.write(out, b"\0" * seg[1])
                else:
                    os.write(out, seg[1])
            done += nbytes
            if progress and name.endswith("down_exps_cold.weight"):
                progress(f"{name}: {done / 1e9:.1f} of {total / 1e9:.1f} GB")
        end = len(head) + pos
        if os.lseek(out, 0, os.SEEK_CUR) < end:
            os.write(out, b"\0" * (end - os.lseek(out, 0, os.SEEK_CUR)))
    for fd in fds.values():
        os.close(fd)
    tmp.replace(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--cold", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    write(Path(a.gguf), json.loads(Path(a.cold).read_text()), Path(a.out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
