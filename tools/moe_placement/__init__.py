"""MoE expert placement: model adapters, a versioned placement map, a cost-model planner and the lookup it implies.

Stdlib only (it runs where setup's tests run: no numpy, no GPU, no model download). Read docs/MOE_PLACEMENT.md first.

    descriptor.py  GGUF header -> ModelDescriptor, through one adapter per architecture
    stats.py       measured routing counts (Strata --dump-routing traces, or counts carried in a map)
    planner.py     cluster spec + descriptor + stats -> placement, under explicit byte budgets
    lookup.py      canonical expert id -> (tier, physical index); the reordering is only ever a storage permutation
    mapfile.py     the shareable map: export, import, compatibility validation
    exporters.py   what today's runtimes can consume (Strata's STRP profile)
"""

FORMAT = "strata.moe-map"
FORMAT_VERSION = 1
CLUSTER_FORMAT = "strata.moe-cluster"
CLUSTER_VERSION = 1
