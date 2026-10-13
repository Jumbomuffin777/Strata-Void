"""Placement for llama.cpp's layer split with hot/cold expert tensors (see gguf_hotcold.py and the llama.cpp patch).

llama.cpp assigns whole layers to devices (`--split-mode layer --tensor-split ...`); with the hot/cold rewrite a
layer's hot experts follow its layer, and the cold parts go to host RAM (`-ot ffn_.*_exps_cold=CPU`). This picks:

  * the per-device layer ranges, emitted as `--tensor-split` (llama.cpp splits layers by cumulative fraction);
  * per device, the cold set: the least-routed experts of ITS layers (measured activations per byte) that do not fit
    its budget, so no device is left with unused room; the split minimising cold activations wins.

Budgets are bytes for weights + KV per device; compute buffers and driver reserves are the caller's margin.

    python tools/moe_placement/llamacpp_hotcold.py MODEL.gguf --trace routing.bin --ctx 131072 --kv-bytes 1.0 \
        --device SYCL0=28.5GiB --device SYCL1=28.5GiB ... --ram 18GiB --out-cold cold.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moe_placement import descriptor, stats  # noqa: E402
from moe_placement.planner import parse_bytes  # noqa: E402


def plan(d, s, devices: list, ram: int, ctx: int, kv_scale: float = 1.0, no_split=(), slot_zeros=()):
    """-> (cold {layer: [ids]}, layer counts per device, RAM bytes for the cold parts, cold activation share).

    Every contiguous split of the layers over the devices (in order) is considered: a device holds its layers' fixed
    bytes (dense, KV, one zero expert per MoE layer), then its layers' experts by measured activations per byte until
    its budget is full; the rest of its experts are cold. Dynamic programming picks the split with the fewest cold
    activations, then the fewest cold bytes, that fits the RAM budget. Devices named in `no_split` keep their layers
    whole (every expert in their memory, no hot/cold tensors): for backends that should not run the split path."""
    budgets = [b for _, b in devices]
    L = d.n_layers
    fixed = [d.dense_bytes[l] + d.kv_bytes_per_token[l] * ctx * kv_scale + d.state_bytes[l] +
             (d.expert_bytes(l) if l in d.expert_tensors else 0) for l in range(L)]
    cnt = {l: (s.counts.get(l) if s.tokens.get(l) else [0] * d.n_expert) for l in d.moe_layers}

    def device_cost(dev, i, j):
        """(cold activations, cold bytes, cold set) for layers [i, j) on device dev, or None if it cannot fit."""
        if devices[dev][0] in no_split:
            whole = sum(fixed[l] + (d.expert_bytes(l) * (d.n_expert - 1) if l in d.expert_tensors else 0)
                        for l in range(i, j))
            return (0, 0, {}) if whole <= budgets[dev] else None
        room = budgets[dev] - sum(fixed[i:j])
        if devices[dev][0] in slot_zeros:          # CUDA: one zero expert per routing slot instead of one in total
            room -= sum(d.expert_bytes(l) * (max(d.n_expert_used, 1) - 1) for l in range(i, j) if l in d.expert_tensors)
        if room < 0:
            return None
        items = sorted(((-cnt[l][e] / d.expert_bytes(l), l, e) for l in range(i, j) if l in cnt
                        for e in range(d.n_expert)))
        cold, ca, cb = {}, 0, 0
        for _, l, e in items:
            eb = d.expert_bytes(l)
            if eb <= room:
                room -= eb
            else:
                cold.setdefault(l, []).append(e); ca += cnt[l][e]; cb += eb
        return ca, cb, cold

    import functools

    @functools.lru_cache(maxsize=None)
    def best(dev, i):
        """Best (cold activations, cold bytes, splits) for layers [i, L) on devices [dev, ...)."""
        if i == L:
            return (0, 0, ())
        if dev == len(budgets):
            return None
        out = None
        for j in range(i, L + 1):
            c = device_cost(dev, i, j) if j > i else (0, 0, {})
            if c is None:
                break                                   # more layers only need more fixed bytes
            rest = best(dev + 1, j)
            if rest is None:
                continue
            cand = (c[0] + rest[0], c[1] + rest[1], (j - i,) + rest[2])
            if out is None or cand[:2] < out[:2]:
                out = cand
        return out

    r = best(0, 0)
    if r is None:
        raise ValueError("the layers' fixed bytes do not fit these devices at this context")
    counts = list(r[2]) + [0] * (len(budgets) - len(r[2]))
    cold, i = {}, 0
    for dev, n in enumerate(counts):
        if n:
            cold.update(device_cost(dev, i, i + n)[2]); i += n
    ram_b = sum(d.expert_bytes(l) * (len(es) + 1) for l, es in cold.items())
    if ram_b > ram:
        raise ValueError(f"cold experts need {ram_b / 2**30:.1f} GiB of RAM, budget {ram / 2**30:.1f} GiB")
    tot = sum(sum(c) for c in cnt.values()) or 1
    return {str(l): sorted(es) for l, es in sorted(cold.items())}, counts, ram_b, r[0] / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--trace", action="append", required=True)
    ap.add_argument("--device", action="append", required=True, help="NAME=BUDGET in layer order, e.g. SYCL0=28.5GiB")
    ap.add_argument("--ram", required=True, help="host RAM budget for the cold parts")
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--no-split", action="append", default=[], help="device NAME whose layers stay whole (repeatable)")
    ap.add_argument("--slot-zeros", action="append", default=[],
                    help="device NAME running CUDA: its split layers get one zero expert per routing slot (repeatable)")
    ap.add_argument("--kv-scale", type=float, default=1.0, help="KV bytes relative to the adapter's f16 estimate")
    ap.add_argument("--out-cold", required=True)
    a = ap.parse_args()
    d, _ = descriptor.describe([a.gguf])
    s = stats.RoutingStats.empty(d.n_layers, d.n_expert)
    for t in a.trace:
        s.merge(stats.read_strata_trace(t, d.n_layers, d.n_expert))
    devs = [(x.split("=")[0], parse_bytes(x.split("=")[1])) for x in a.device]
    cold, counts, ram_b, share = plan(d, s, devs, parse_bytes(a.ram), a.ctx, a.kv_scale, set(a.no_split), set(a.slot_zeros))
    Path(a.out_cold).write_text(json.dumps(cold))
    n_cold = sum(len(v) for v in cold.values())
    print(f"cold: {n_cold} experts in {len(cold)} layers, {ram_b / 2**30:.2f} GiB RAM, "
          f"{share:.1%} of measured activations")
    print(" ".join(split_args([n for n, _ in devs], counts)))
    print("layers per device:", dict(zip((n for n, _ in devs), counts)))
    first, sz = 0, []
    for (n, _), c in zip(devs, counts):
        if n in a.slot_zeros and c:
            sz += [l for l in range(first, first + c) if str(l) in cold]
        first += c
    if sz:
        print(f"rewrite with: --slot-zeros {min(sz)}-{max(sz)}")


def split_args(names: list, counts: list) -> list:
    """llama.cpp puts layer il on the first device whose cumulative split exceeds il / (n_layer + 1), the output
    layer being index n_layer: so the split is the layer counts with the output counted on the last device used."""
    used = [(n, c) for n, c in zip(names, counts) if c > 0]
    split = [c for _, c in used]
    split[-1] += 1
    return ["--device", ",".join(n for n, _ in used), "--split-mode", "layer",
            "--tensor-split", ",".join(map(str, split)), "-ot", "ffn_.*_exps_cold\\.=CPU"]


if __name__ == "__main__":
    main()
