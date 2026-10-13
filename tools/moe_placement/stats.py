"""Measured routing statistics: how often each (layer, expert) was routed, over how many tokens.

The ranking signal is the measured routing frequency, never weight magnitude: a large-norm expert the router
rarely picks is still cold. Sources:
  * Strata's `--dump-routing FILE` trace (int32 layer, int32 k, k int32 expert ids, k float32 weights per record),
    the same file tools/make_profile.py reads;
  * the `stats` section of a placement map (a shared / community map carries its counts, not just its ranking).
"""
from __future__ import annotations

import dataclasses
import struct
from pathlib import Path


class StatsError(ValueError):
    pass


@dataclasses.dataclass
class RoutingStats:
    n_layers: int
    n_expert: int
    tokens: dict          # layer -> tokens observed at that layer
    counts: dict          # layer -> [n_expert] times routed
    mass: dict            # layer -> [n_expert] summed routing weight (empty when the source has none)
    source: str = ""

    @classmethod
    def empty(cls, n_layers, n_expert, source=""):
        return cls(n_layers, n_expert, {}, {}, {}, source)

    def freq(self, layer: int, expert: int) -> float:
        """P(expert routed | a token reaches layer). 0 for a layer with no observations."""
        t = self.tokens.get(layer, 0)
        return self.counts[layer][expert] / t if t else 0.0

    def add(self, layer: int, ids, weights=None) -> None:
        if layer not in self.counts:
            self.counts[layer] = [0] * self.n_expert
            self.tokens[layer] = 0
        self.tokens[layer] += 1
        c = self.counts[layer]
        for i, e in enumerate(ids):
            if not 0 <= e < self.n_expert:
                raise StatsError(f"layer {layer}: expert id {e} out of range 0..{self.n_expert - 1}")
            c[e] += 1
            if weights is not None:
                m = self.mass.setdefault(layer, [0.0] * self.n_expert)
                m[e] += float(weights[i])

    def merge(self, other: "RoutingStats") -> None:
        if (other.n_layers, other.n_expert) != (self.n_layers, self.n_expert):
            raise StatsError("cannot merge statistics of different geometries")
        for l, c in other.counts.items():
            mine = self.counts.setdefault(l, [0] * self.n_expert)
            self.tokens[l] = self.tokens.get(l, 0) + other.tokens[l]
            for e, v in enumerate(c):
                mine[e] += v
        for l, m in other.mass.items():
            mine = self.mass.setdefault(l, [0.0] * self.n_expert)
            for e, v in enumerate(m):
                mine[e] += v
        self.source = "+".join(s for s in (self.source, other.source) if s)

    def validate(self, n_expert_used: int, moe_layers) -> None:
        """Counts must be possible: non-negative, per-layer total at most tokens x top-k, only on MoE layers."""
        moe = set(moe_layers)
        for l, c in self.counts.items():
            if l not in moe:
                raise StatsError(f"statistics for layer {l}, which has no routed experts")
            if len(c) != self.n_expert or any(v < 0 for v in c):
                raise StatsError(f"layer {l}: counts must be {self.n_expert} non-negative integers")
            if n_expert_used and sum(c) > self.tokens[l] * n_expert_used:
                raise StatsError(f"layer {l}: {sum(c)} routings over {self.tokens[l]} tokens exceeds top-"
                                 f"{n_expert_used}")

    def to_json(self) -> dict:
        out = {"source": self.source,
               "layers": {str(l): {"tokens": self.tokens[l], "counts": self.counts[l]} for l in sorted(self.counts)}}
        for l in sorted(self.mass):
            out["layers"][str(l)]["weight_mass"] = [round(v, 6) for v in self.mass[l]]
        return out

    @classmethod
    def from_json(cls, j: dict, n_layers: int, n_expert: int) -> "RoutingStats":
        s = cls.empty(n_layers, n_expert, j.get("source", ""))
        for k, v in j.get("layers", {}).items():
            l = int(k)
            s.tokens[l] = int(v["tokens"])
            s.counts[l] = [int(x) for x in v["counts"]]
            if "weight_mass" in v:
                s.mass[l] = [float(x) for x in v["weight_mass"]]
        return s


def read_strata_trace(path, n_layers: int, n_expert: int) -> RoutingStats:
    blob = Path(path).read_bytes()
    s = RoutingStats.empty(n_layers, n_expert, f"strata-trace:{Path(path).name}")
    off = 0
    while off + 8 <= len(blob):
        layer, k = struct.unpack_from("<ii", blob, off)
        off += 8
        if k <= 0 or off + 8 * k > len(blob):
            raise StatsError(f"{path}: malformed record at byte {off - 8}")
        ids = struct.unpack_from(f"<{k}i", blob, off)
        w = struct.unpack_from(f"<{k}f", blob, off + 4 * k)
        off += 8 * k
        if not 0 <= layer < n_layers:
            raise StatsError(f"{path}: layer {layer} out of range")
        s.add(layer, ids, w)
    if off != len(blob):
        raise StatsError(f"{path}: {len(blob) - off} trailing bytes")
    return s


def write_strata_trace(path, records) -> None:
    """records: iterable of (layer, ids, weights). The inverse of read_strata_trace (tests, synthetic traces)."""
    with open(path, "wb") as f:
        for layer, ids, w in records:
            f.write(struct.pack("<ii", layer, len(ids)))
            f.write(struct.pack(f"<{len(ids)}i", *ids))
            f.write(struct.pack(f"<{len(ids)}f", *w))
