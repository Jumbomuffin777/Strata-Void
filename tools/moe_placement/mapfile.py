"""The shareable placement map: one JSON file, versioned, validated on import.

    format, version            "strata.moe-map", 1. A newer MAJOR version is refused; unknown keys are ignored.
    checkpoint                 fingerprint (gguf-layout-v1), optional content sample, architecture, name, file type,
                               file names and sizes (never paths)
    model                      the geometry the map was made for: layers, MoE layers, experts, top-k, shared
                               experts, hash-routed and MTP layers
    tensors                    per MoE layer and role: tensor name, ggml type, bytes of ONE expert's slice
    stats                      measured routing counts per layer (+ tokens, + optional weight mass) and their source
    placement (optional)       tiers, per-layer owner, per-expert tier and rank, physical order, segments, budgets,
                               cost-model parameters, expected hit rate
    runtime_support            informational only: recomputed from the adapter on every validation, never trusted

Sharing: a map's STATS transfer to any checkpoint of the same architecture and geometry (another quantization of
the same weights routes the same way, to first order); its PLACEMENT is only valid for the fingerprinted checkpoint
and a cluster with at least the recorded budgets. `validate` enforces exactly that split.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import FORMAT, FORMAT_VERSION
from .descriptor import FINGERPRINT_METHOD, ModelDescriptor
from .lookup import check_consistent
from .planner import Placement
from .stats import RoutingStats


class MapError(ValueError):
    pass


def build(d: ModelDescriptor, stats: RoutingStats, placement: Placement | None = None,
          content_sample: str | None = None, tool: str = "tools/moe_place.py") -> dict:
    m = {
        "format": FORMAT, "version": FORMAT_VERSION,
        "checkpoint": {"fingerprint": d.fingerprint, "fingerprint_method": FINGERPRINT_METHOD,
                       "architecture": d.architecture, "name": d.name, "file_type": d.file_type,
                       "files": d.files},
        "model": {"n_layers": d.n_layers, "moe_layers": d.moe_layers, "n_expert": d.n_expert,
                  "n_expert_used": d.n_expert_used, "n_shared": d.n_shared, "embd": d.embd,
                  "hash_routed_layers": d.hash_routed_layers, "mtp_layers": d.mtp_layers},
        "tensors": {str(l): [{"role": t.role, "name": t.name, "type": t.type, "expert_bytes": t.expert_bytes}
                             for t in d.expert_tensors[l]] for l in d.moe_layers},
        "stats": stats.to_json(),
        "runtime_support": d.runtime_support,
        "provenance": {"tool": tool, "notes": d.notes},
    }
    if content_sample:
        m["checkpoint"]["content_sample"] = content_sample
    if placement is not None:
        p = placement
        m["placement"] = {
            "tiers": p.devices,
            "layer_owner": {str(l): t for l, t in sorted(p.layer_owner.items())},
            "layers": {str(l): {"tier": p.assign[l], "rank": p.rank[l], "physical_order": p.physical_order[l],
                                "segments": p.segments[l]} for l in sorted(p.assign)},
            "budgets": p.budgets, "expected": p.expected, "params": p.params,
            "unmeasured_layers": p.unmeasured_layers, "remote_advice": p.remote_advice,
        }
    return m


def dumps(m: dict) -> str:
    """Deterministic text: sorted keys, per-layer arrays on one line each so a diff of two maps stays readable."""
    def enc(o, ind=0):
        pad = " " * ind
        if isinstance(o, dict):
            if not o:
                return "{}"
            items = [f'{pad} {json.dumps(k)}: {enc(v, ind + 1)}' for k, v in sorted(o.items())]
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            return "[\n" + ",\n".join(f"{pad} {enc(x, ind + 1)}" for x in o) + "\n" + pad + "]"
        return json.dumps(o, separators=(",", ":"))
    return enc(m) + "\n"


def save(m: dict, path) -> None:
    Path(path).write_text(dumps(m), encoding="utf-8")


def load(path) -> dict:
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MapError(f"{path}: not JSON ({e})")
    if m.get("format") != FORMAT:
        raise MapError(f"{path}: format is {m.get('format')!r}, not {FORMAT!r}")
    if int(m.get("version", 0)) > FORMAT_VERSION:
        raise MapError(f"{path}: map version {m['version']} is newer than this tool reads ({FORMAT_VERSION})")
    return m


def stats_of(m: dict) -> RoutingStats:
    md = m["model"]
    return RoutingStats.from_json(m["stats"], md["n_layers"], md["n_expert"])


def validate(m: dict, d: ModelDescriptor | None = None, *, stats_only: bool = False) -> list:
    """Raises MapError on anything that would make the map wrong for `d`; returns warnings.

    Without `d`: internal consistency only. With `d`: the geometry must match, and unless `stats_only` the
    fingerprint and every tensor's type and slice size must match too (a placement is checkpoint-specific)."""
    warn = []
    md = m["model"]
    n_expert, moe = md["n_expert"], md["moe_layers"]
    s = stats_of(m)
    try:
        s.validate(md["n_expert_used"], moe)
    except ValueError as e:
        raise MapError(f"stats: {e}")
    if sorted(int(k) for k in m["tensors"]) != sorted(moe):
        raise MapError("tensors: the layers listed differ from model.moe_layers")

    if d is not None:
        if m["checkpoint"]["architecture"] != d.architecture:
            raise MapError(f"map is for architecture {m['checkpoint']['architecture']!r}, checkpoint is "
                           f"{d.architecture!r}")
        for k, v in (("n_layers", d.n_layers), ("n_expert", d.n_expert), ("n_expert_used", d.n_expert_used),
                     ("moe_layers", d.moe_layers)):
            if md[k] != v:
                raise MapError(f"model.{k} is {md[k]} in the map and {v} in the checkpoint")
        same = m["checkpoint"]["fingerprint"] == d.fingerprint
        if not same:
            if not stats_only:
                raise MapError("the map's fingerprint does not match this checkpoint: its placement is for another "
                               "file (use its stats only and plan again)")
            warn.append("fingerprint differs: only the routing statistics are used")
        if same:
            for l in moe:
                mine = [(t.role, t.type, t.expert_bytes) for t in d.expert_tensors[l]]
                theirs = [(t["role"], t["type"], t["expert_bytes"]) for t in m["tensors"][str(l)]]
                if mine != theirs:
                    raise MapError(f"layer {l}: expert tensors {theirs} in the map, {mine} in the checkpoint")
        if m.get("runtime_support") != d.runtime_support:
            warn.append("runtime_support in the file differs from this engine's; the engine's applies: "
                        f"{d.runtime_support}")

    pl = m.get("placement")
    if pl is not None and not stats_only:
        layers = {int(k): v for k, v in pl["layers"].items()}
        if sorted(layers) != sorted(moe):
            raise MapError("placement: layers differ from model.moe_layers")
        n_tiers = len(pl["tiers"])
        try:
            check_consistent({l: v["tier"] for l, v in layers.items()},
                             {l: v["physical_order"] for l, v in layers.items()},
                             {l: v["segments"] for l, v in layers.items()}, n_expert, n_tiers)
        except ValueError as e:
            raise MapError(f"placement: {e}")
        ranks = sorted(r for v in layers.values() for r in v["rank"])
        if ranks != list(range(len(moe) * n_expert)):
            raise MapError("placement: ranks must number every (layer, expert) exactly once")
        used = [0] * n_tiers
        for l, v in layers.items():
            eb = sum(t["expert_bytes"] for t in m["tensors"][str(l)])
            for t in v["tier"]:
                used[t] += eb
        for t, b in enumerate(pl["budgets"]):
            if used[t] != b["expert_used"]:
                raise MapError(f"placement: {b['tier']} holds {used[t]} B of experts, the budget says "
                               f"{b['expert_used']}")
            if used[t] > b["expert_budget"]:
                raise MapError(f"placement: {b['tier']} holds {used[t]} B of experts over its budget "
                               f"{b['expert_budget']}")
    return warn
