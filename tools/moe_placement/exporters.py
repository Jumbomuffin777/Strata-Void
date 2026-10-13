"""What today's runtimes can consume from a map.

Strata (qwen4exp only): the STRP expert profile `--expert-profile` reads (include/strata/core/expert_cache.hpp,
tools/make_profile.py), byte for byte the same format: `STRP`, `version, n_layers, n_expert, n_ranked, n_ranked`
(uint32), the ranked (uint16 layer, uint16 expert) pairs, then the n_layers x n_expert int32 rank table. The ranking
is the plan's tier order (GPU 0's experts, then GPU 1's, ..., then RAM), by priority within each tier, so
`--expert-cache N0` with N0 = the plan's GPU-0 count pre-loads exactly GPU 0's set.

The engine distributes the ranks after N0 over `--expert-cache-device1..3` its own way (striped or by layer,
`--expert-cache-remote-placement`), so the plan's per-GPU split beyond GPU 0 is a suggestion there, not a contract.
"""
from __future__ import annotations

import struct
from pathlib import Path

MAGIC, VERSION = b"STRP", 1


class ExportError(ValueError):
    pass


def strp_ranking(m: dict) -> list:
    pl = m.get("placement")
    if pl is None:
        raise ExportError("the map has no placement to export")
    pairs = []
    for l, v in pl["layers"].items():
        for e, (t, r) in enumerate(zip(v["tier"], v["rank"])):
            pairs.append((t, r, int(l), e))
    pairs.sort()
    return [(l, e) for _, _, l, e in pairs]


def tier_counts(m: dict) -> dict:
    pl = m["placement"]
    out = {name: 0 for name in pl["tiers"]}
    for v in pl["layers"].values():
        for t in v["tier"]:
            out[pl["tiers"][t]] += 1
    return out


def write_strp(m: dict, path, runtime_support: dict) -> dict:
    """Writes the profile; returns the per-tier counts for the matching --expert-cache arguments. Refuses unless the
    CHECKPOINT's adapter (not the map file) says the Strata engine runs it."""
    if not runtime_support.get("strata_native"):
        raise ExportError(f"the Strata engine cannot run this checkpoint: {runtime_support.get('reason')}")
    md = m["model"]
    nl, ne = md["n_layers"], md["n_expert"]
    if nl > 65535 or ne > 65535:
        raise ExportError("geometry does not fit the STRP format")
    ranked = strp_ranking(m)
    table = [-1] * (nl * ne)
    for r, (l, e) in enumerate(ranked):
        table[l * ne + e] = r
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, nl, ne, len(ranked), len(ranked)))
        for l, e in ranked:
            f.write(struct.pack("<HH", l, e))
        f.write(struct.pack(f"<{nl * ne}i", *table))
    tmp.replace(path)
    return tier_counts(m)


def read_strp(path) -> tuple[int, int, list]:
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise ExportError(f"{path}: not a STRP profile")
    _v, nl, ne, _slots, n = struct.unpack_from("<5I", blob, 4)
    return nl, ne, [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]
