"""GGUF header -> ModelDescriptor: what the placement planner needs to know about one checkpoint, and nothing else.

Only the header is read (metadata and the tensor table); no weight byte is touched unless `content_sample` is asked
for, and then a few KiB per expert tensor. Split checkpoints (`-0000N-of-0000M.gguf`) are read shard by shard.

An adapter turns an architecture's metadata into byte costs: per (layer, role) expert slice sizes, the dense bytes
each layer pins, its KV bytes per token and its fixed recurrent state. Those are ESTIMATES derived from the
metadata, stated as such in the map, until a runtime that executes the architecture measures them. An adapter does
not make an architecture runnable: `runtime_support` says what actually executes it, and is recomputed on every
validation rather than trusted from a map file.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402

FINGERPRINT_METHOD = "gguf-layout-v1"
_EXP = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$")
_BLK = re.compile(r"blk\.(\d+)\.")
ROLES = ("gate", "up", "down", "gate_up")


class UnsupportedArchitecture(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class ExpertTensor:
    """One routed-expert tensor of one layer. ggml stores it [in, out, n_expert]: the expert index is the outermost
    dimension, so expert `e`'s slice is the contiguous byte range `offset + e * expert_bytes`."""
    layer: int
    role: str
    name: str
    type: str
    shape: tuple
    expert_bytes: int
    file: str
    offset: int          # absolute byte offset of the tensor's data in `file`

    def slice_offset(self, expert: int) -> int:
        return self.offset + expert * self.expert_bytes


@dataclasses.dataclass
class ModelDescriptor:
    architecture: str
    name: str
    n_layers: int
    n_expert: int
    n_expert_used: int
    n_shared: int
    embd: int
    moe_layers: list
    expert_tensors: dict                 # layer -> [ExpertTensor] (gate, up, down order)
    dense_bytes: dict                    # layer -> bytes pinned with that layer (attention, router, shared expert, ...)
    global_bytes: dict                   # tensor name -> bytes, for tensors outside any block
    kv_bytes_per_token: dict             # layer -> bytes per context token (estimate)
    state_bytes: dict                    # layer -> fixed per-sequence bytes (recurrent state, sliding window)
    hash_routed_layers: list = dataclasses.field(default_factory=list)
    mtp_layers: list = dataclasses.field(default_factory=list)
    quant_types: dict = dataclasses.field(default_factory=dict)   # "layer.role" -> ggml type
    file_type: int | None = None
    fingerprint: str = ""
    files: list = dataclasses.field(default_factory=list)          # [{"name", "size"}], no paths
    runtime_support: dict = dataclasses.field(default_factory=dict)
    notes: list = dataclasses.field(default_factory=list)

    def expert_bytes(self, layer: int) -> int:
        """Bytes of ONE expert of `layer`, all roles. Layers differ (mixed quantization), experts of a layer do not."""
        return sum(t.expert_bytes for t in self.expert_tensors[layer])

    def total_expert_bytes(self) -> int:
        return sum(self.expert_bytes(l) * self.n_expert for l in self.moe_layers)

    def act_bytes(self) -> int:
        """One token's expert input or output, fp32: what crosses a link when an expert runs somewhere else."""
        return self.embd * 4

    def summary(self) -> dict:
        return {
            "architecture": self.architecture, "name": self.name, "fingerprint": self.fingerprint,
            "n_layers": self.n_layers, "moe_layers": _ranges(self.moe_layers), "n_expert": self.n_expert,
            "n_expert_used": self.n_expert_used, "n_shared": self.n_shared,
            "hash_routed_layers": self.hash_routed_layers, "mtp_layers": self.mtp_layers,
            "expert_bytes_by_layer": {str(l): self.expert_bytes(l) for l in self.moe_layers},
            "total_expert_gb": round(self.total_expert_bytes() / 1e9, 3),
            "dense_gb": round((sum(self.dense_bytes.values()) + sum(self.global_bytes.values())) / 1e9, 3),
            "kv_bytes_per_token": sum(self.kv_bytes_per_token.values()),
            "state_mib": round(sum(self.state_bytes.values()) / 2**20, 1),
            "quant_types": _count(self.quant_types.values()),
            "runtime_support": self.runtime_support, "notes": self.notes,
        }


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _ranges(xs) -> str:
    xs = sorted(xs)
    if not xs:
        return ""
    out, start, prev = [], xs[0], xs[0]
    for x in xs[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        if x is not None:
            start = prev = x
    return ",".join(out)


def tensor_bytes(type_name: str, shape) -> int:
    geom = BLOCK_GEOMETRY.get(type_name)
    if geom is None:
        raise ValueError(f"no block geometry for ggml type {type_name}")
    n = 1
    for d in shape:
        n *= d
    be, bb = geom
    if n % be:
        raise ValueError(f"{n} elements is not a whole number of {type_name} blocks")
    return n // be * bb


# ---------------------------------------------------------------------------------------------- the GGUF header
@dataclasses.dataclass
class Header:
    metadata: dict
    tensors: list        # [(name, shape, type_name, file_name, abs_offset)]
    files: list          # [{"name", "size"}]
    paths: list          # local paths (never written into a map)


def read_header(paths) -> Header:
    paths = [pathlib.Path(p) for p in paths]
    if len(paths) == 1:
        m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", paths[0].name)
        if m:
            n = int(m.group(3))
            paths = [paths[0].with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
    meta, tensors, files = {}, [], []
    for p in paths:
        g = GGUFFile(p)
        for k, v in g.metadata.items():
            if not k.startswith("split."):
                meta.setdefault(k, v)
        for t in g.tensors:
            tensors.append((t.name, tuple(t.shape), t.type_name, p.name, g.data_start + t.offset))
        files.append({"name": p.name, "size": p.stat().st_size})
    return Header(meta, tensors, files, paths)


def fingerprint(h: Header) -> str:
    """sha256 of the metadata (shard bookkeeping excluded) and the tensor table (name, type, shape; offsets excluded),
    so the same checkpoint split differently fingerprints the same. It identifies a LAYOUT AND QUANTIZATION, not the
    weight bytes: see `content_sample` for that."""
    d = hashlib.sha256()
    d.update(FINGERPRINT_METHOD.encode())
    d.update(json.dumps(h.metadata, sort_keys=True, separators=(",", ":"), default=str).encode())
    for name, shape, typ, _f, _o in sorted(h.tensors):
        d.update(f"{name}|{typ}|{','.join(map(str, shape))}\n".encode())
    return "sha256:" + d.hexdigest()


def content_sample(h: Header, desc: "ModelDescriptor", nbytes: int = 4096) -> str:
    """sha256 over the first `nbytes` of every expert tensor: catches two checkpoints that share a layout but not
    weights (a fine-tune that kept the metadata) for a few hundred KiB of reads."""
    by_name = {p.name: p for p in h.paths}
    d = hashlib.sha256()
    for l in desc.moe_layers:
        for t in desc.expert_tensors[l]:
            with open(by_name[t.file], "rb") as f:
                f.seek(t.offset)
                d.update(f.read(nbytes))
    return "sha256:" + d.hexdigest()


# ---------------------------------------------------------------------------------------------- adapters
class Adapter:
    """Architecture-independent part: find the routed-expert tensors, size the expert slices, put every other tensor
    under its layer (dense) or global. Subclasses add KV/state costs, routing quirks and runtime support."""
    arch = ""

    def __init__(self, h: Header):
        self.h, self.m = h, h.metadata
        self.a = self.m.get("general.architecture", "")

    def key(self, k, default=None):
        return self.m.get(f"{self.a}.{k}", default)

    def describe(self) -> ModelDescriptor:
        n_layers = int(self.key("block_count"))
        n_expert = int(self.key("expert_count"))
        experts: dict = {}
        dense: dict = {l: 0 for l in range(n_layers)}
        glob: dict = {}
        quant: dict = {}
        shapes_by_layer: dict = {}
        for name, shape, typ, fname, off in self.h.tensors:
            m = _EXP.match(name)
            if m:
                layer, role = int(m.group(1)), m.group(2)
                if shape[-1] != n_expert:
                    raise ValueError(f"{name}: last dimension {shape[-1]} is not expert_count {n_expert}")
                eb = tensor_bytes(typ, shape[:-1])
                experts.setdefault(layer, []).append(ExpertTensor(layer, role, name, typ, shape, eb, fname, off))
                quant[f"{layer}.{role}"] = typ
                continue
            b = tensor_bytes(typ, shape) if typ in BLOCK_GEOMETRY else 0
            mb = _BLK.match(name)
            if mb:
                dense[int(mb.group(1))] += b
                shapes_by_layer.setdefault(int(mb.group(1)), {})[name[mb.end():]] = shape
            else:
                glob[name] = b
        for l, ts in experts.items():
            roles = {t.role for t in ts}
            if roles not in ({"gate", "up", "down"}, {"gate_up", "down"}):
                raise ValueError(f"layer {l}: expert roles {sorted(roles)} are incomplete")
            ts.sort(key=lambda t: ROLES.index(t.role))
        self.shapes = shapes_by_layer
        d = ModelDescriptor(
            architecture=self.a, name=str(self.m.get("general.name", "")), n_layers=n_layers, n_expert=n_expert,
            n_expert_used=int(self.key("expert_used_count", 0)), n_shared=int(self.key("expert_shared_count", 0)),
            embd=int(self.key("embedding_length")), moe_layers=sorted(experts), expert_tensors=experts,
            dense_bytes=dense, global_bytes=glob, kv_bytes_per_token={l: 0.0 for l in range(n_layers)},
            state_bytes={l: 0 for l in range(n_layers)}, quant_types=quant,
            file_type=self.m.get("general.file_type"), fingerprint=fingerprint(self.h), files=list(self.h.files))
        self.costs(d)
        d.runtime_support = self.runtime_support(d)
        return d

    def costs(self, d: ModelDescriptor) -> None:
        d.notes.append("generic adapter: KV cache and recurrent state are NOT modelled; budget them explicitly")

    def runtime_support(self, d: ModelDescriptor) -> dict:
        return {"strata_native": False, "reason": f"no Strata layer implementation for '{self.a}'"}


class Qwen4ExpAdapter(Adapter):
    """Qwen3.8-Flash-Next (qwen4exp): the architecture the Strata engine implements. Costs ported from
    include/strata/plan/plan.hpp (INT8 KV grouped-64 on the 12 QSA layers, one pooled indexer key, fp32 GDN state)."""
    arch = "qwen4exp"

    def costs(self, d):
        interval = int(self.key("full_attention_interval", 4))
        for l in range(d.n_layers):
            if l % interval == interval - 1:
                kv = 2 * 2 * 256
                d.kv_bytes_per_token[l] = kv + kv // 64 * 2 + 128 / 4
            else:
                d.state_bytes[l] = 128 * 48 * 128 * 4 + 3 * 10240 * 4
        d.notes.append("KV/state costs ported from include/strata/plan/plan.hpp")

    def runtime_support(self, d):
        ok = d.n_layers == 48 and d.n_expert == 512
        return {"strata_native": ok,
                "reason": "the engine's geometry (include/strata/core/layout.hpp: 48 layers x 512 experts)" if ok
                else f"geometry {d.n_layers}x{d.n_expert} differs from the engine's fixed 48x512"}


class Glm5NextAdapter(Adapter):
    """GLM-5.3-Flash (glm5next). From its metadata: leading dense blocks, then MoE; per-layer attention is either MLA
    (head_count_kv[l] > 0: a compressed latent of kv_lora_rank per token, plus a pooled indexer key) or KDA linear
    attention (fixed recurrent state, sized from ssm_a's head count and kda.head_dim). The last block is the
    next-token (MTP) layer and its experts only run when MTP drafting does."""
    arch = "glm5next"

    def costs(self, d):
        kv_heads = self.key("attention.head_count_kv")
        if not isinstance(kv_heads, list):
            kv_heads = [kv_heads] * d.n_layers
        lora = int(self.key("attention.kv_lora_rank", 512))
        idx_k = int(self.key("attention.indexer.key_length", 128))
        kpool = int(self.key("attention.indexer.kpool", 1)) or 1
        hd = int(self.key("kda.head_dim", 128))
        conv = int(self.key("ssm.conv_kernel", 4))
        for l in range(d.n_layers):
            sh = self.shapes.get(l, {})
            if kv_heads[l]:
                d.kv_bytes_per_token[l] = lora * 2 + (idx_k * 2 / kpool if "indexer.attn_k.weight" in sh else 0)
            elif "ssm_a" in sh:
                heads = int(sh["ssm_a"][0])
                width = int(sh["ssm_conv1d_q.weight"][-1]) if "ssm_conv1d_q.weight" in sh else heads * hd
                d.state_bytes[l] = heads * hd * hd * 4 + 3 * (conv - 1) * width * 4
        nextn = int(self.key("nextn_predict_layers", 0))
        d.mtp_layers = list(range(d.n_layers - nextn, d.n_layers))
        d.notes.append("KV/state estimated from metadata (f16 latent, fp32 KDA state); not measured by a runtime")
        if d.mtp_layers:
            d.notes.append(f"layer(s) {d.mtp_layers} are the MTP block: their experts run only with MTP drafting")

    def runtime_support(self, d):
        return {"strata_native": False,
                "reason": "Strata has no glm5next layer (KDA linear attention, MLA with an indexer, 4-way hyper-"
                          "connections) and its engine geometry is fixed to qwen4exp; a placement map does not "
                          "supply kernels"}


class DeepSeek4Adapter(Adapter):
    """DeepSeek V4 Flash (deepseek4). Every block is MoE; the first `hash_layer_count` route by a token-id table
    (ffn_gate_tid2eid) instead of the learned router. Attention keeps one 512-wide KV head over a sliding window
    plus compressed entries (ctx / compress_ratio) and, on indexer layers, a 128-wide indexer key per entry."""
    arch = "deepseek4"

    def costs(self, d):
        ratios = self.key("attention.compress_ratios") or [0] * d.n_layers
        kdim = int(self.key("attention.key_length", 512)) * int(self.key("attention.head_count_kv", 1))
        window = int(self.key("attention.sliding_window", 0))
        idx_k = int(self.key("attention.indexer.key_length", 128))
        for l in range(d.n_layers):
            r = int(ratios[l]) if l < len(ratios) else 0
            d.state_bytes[l] = window * kdim * 2
            if r > 0:
                idx = idx_k * 2 if "indexer.proj.weight" in self.shapes.get(l, {}) else 0
                d.kv_bytes_per_token[l] = (kdim * 2 + idx) / r
        d.hash_routed_layers = [l for l in range(d.n_layers) if f"blk.{l}.ffn_gate_tid2eid.weight" in
                                {t[0] for t in self.h.tensors}]
        d.notes.append("KV estimated from metadata (f16, window + compressed entries); not measured by a runtime")
        if d.hash_routed_layers:
            d.notes.append(f"layers {_ranges(d.hash_routed_layers)} are hash-routed by token id: their expert "
                           "frequencies follow the token distribution, so stats must come from representative text")

    def runtime_support(self, d):
        return {"strata_native": False,
                "reason": "Strata has no deepseek4 layer (compressed sparse attention with an indexer, hash-routed "
                          "layers, hyper-connections) and its engine geometry is fixed to qwen4exp; a placement map "
                          "does not supply kernels"}


ADAPTERS = {a.arch: a for a in (Qwen4ExpAdapter, Glm5NextAdapter, DeepSeek4Adapter)}
# other spellings of the same architecture: Project Maya's GLM-5.3-Flash files say "glm5-next" (the standard GGUF name),
# the first llama.cpp port and unsloth's files "glm5next"; metadata keys follow each file's own spelling
ALIASES = {"glm5-next": "glm5next"}
ADAPTERS.update({alias: ADAPTERS[name] for alias, name in ALIASES.items()})


def canonical_arch(arch: str) -> str:
    return ALIASES.get(arch, arch)


def describe(paths, allow_generic: bool = False) -> tuple[ModelDescriptor, Header]:
    h = read_header(paths)
    arch = h.metadata.get("general.architecture", "")
    cls = ADAPTERS.get(arch)
    if cls is None:
        if not allow_generic:
            raise UnsupportedArchitecture(
                f"no adapter for architecture '{arch}' (known: {', '.join(sorted(ADAPTERS))}); "
                "--allow-generic plans it without KV/state costs")
        cls = Adapter
    d = cls(h).describe()
    if not d.moe_layers:
        raise ValueError(f"{arch}: no routed-expert tensors (blk.N.ffn_*_exps.weight) found")
    return d, h
