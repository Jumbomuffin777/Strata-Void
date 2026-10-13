"""Canonical expert id -> where its bytes are. The reordering is a STORAGE permutation and nothing else.

Per MoE layer the plan gives `physical_order` P (physical index -> canonical id) and contiguous `segments`
(tier, start, count). The lookup keeps the inverse Q (canonical id -> physical index), so for a routed expert e:

    phys            = Q[layer][e]
    tier, start     = the segment containing phys
    index_in_tier   = phys - start                  byte offset = tier_base[layer] + index_in_tier * expert_bytes

Everything that is a function of the expert's IDENTITY stays canonical: the router's logits and bias
(exp_probs_b), a hash-routed layer's token -> expert table, the top-k selection and its order, the routing weights,
and the order the expert outputs are summed in. Only the address the weights are read from goes through Q. That is
what keeps the output bit-identical to the canonical layout: the same products are summed in the same order.

In the Strata engine this is the residency table ExpertCache keeps (`slot_of(layer, expert)`,
include/strata/core/expert_cache.hpp): a [n_layers x n_expert] int32 indexed by the canonical id, -1 = host arena,
with the host arena itself in canonical order (ExpertLayout::blob_offset). A plan maps onto it directly: the GPU
segments are slots, the RAM segment is the canonical arena, and Q restricted to a GPU segment is the slot index.
"""
from __future__ import annotations


class LookupError_(ValueError):
    pass


class ExpertLookup:
    def __init__(self, n_expert: int, physical_order: dict, segments: dict, n_tiers: int):
        self.n_expert = n_expert
        self.inv, self.seg_of, self.segments = {}, {}, {}
        for l, po in physical_order.items():
            l = int(l)
            if sorted(po) != list(range(n_expert)):
                raise LookupError_(f"layer {l}: physical_order is not a permutation of 0..{n_expert - 1}")
            q = [0] * n_expert
            for phys, e in enumerate(po):
                q[e] = phys
            segs = [tuple(s) for s in (segments[l] if l in segments else segments[str(l)])]
            pos = 0
            tiers = []
            for t, start, count in segs:
                if start != pos or count <= 0 or not 0 <= t < n_tiers or (tiers and t <= tiers[-1]):
                    raise LookupError_(f"layer {l}: segments must be contiguous, non-empty and in tier order")
                tiers.append(t)
                pos += count
            if pos != n_expert:
                raise LookupError_(f"layer {l}: segments cover {pos} of {n_expert} experts")
            tier_of_phys = [0] * n_expert
            for t, start, count in segs:
                for i in range(start, start + count):
                    tier_of_phys[i] = (t, start)
            self.inv[l], self.seg_of[l], self.segments[l] = q, tier_of_phys, segs

    def locate(self, layer: int, expert: int) -> tuple[int, int]:
        """(tier, index within that tier's run of this layer) for canonical `expert`."""
        if not 0 <= expert < self.n_expert:
            raise LookupError_(f"expert {expert} out of range")
        phys = self.inv[layer][expert]
        t, start = self.seg_of[layer][phys]
        return t, phys - start

    def tier_table(self, layer: int, tier: int) -> list:
        """Strata's residency-table form for one tier: canonical id -> index in that tier, or -1."""
        out = [-1] * self.n_expert
        for e in range(self.n_expert):
            t, i = self.locate(layer, e)
            if t == tier:
                out[e] = i
        return out


def check_consistent(assign: dict, physical_order: dict, segments: dict, n_expert: int, n_tiers: int) -> None:
    """The plan's three views of one fact (per-expert tier, physical order, segments) must agree."""
    lk = ExpertLookup(n_expert, physical_order, segments, n_tiers)
    for l, a in assign.items():
        l = int(l)
        for e in range(n_expert):
            if lk.locate(l, e)[0] != a[e]:
                raise LookupError_(f"layer {l} expert {e}: assigned to tier {a[e]} but stored in tier "
                                   f"{lk.locate(l, e)[0]}")
