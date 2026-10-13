"""MoE expert placement maps: inspect a checkpoint, plan a placement, validate a map, export what a runtime reads.

    python tools/moe_place.py inspect MODEL.gguf                       # header only: adapter, sizes, fingerprint
    python tools/moe_place.py plan MODEL.gguf --cluster CLUSTER.json (--trace T ... | --stats-from MAP)
                                   [--allow-unmeasured] [--content-sample] --out MAP.json
    python tools/moe_place.py validate MAP.json [--gguf MODEL.gguf [--stats-only]] [--cluster CLUSTER.json]
    python tools/moe_place.py export-strp MAP.json --gguf MODEL.gguf --out expert-profile.bin

A split checkpoint is named by its first shard; the others are found beside it. Nothing here reads weights except
--content-sample (4 KiB per expert tensor). docs/MOE_PLACEMENT.md has the format, the cost model and the limits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moe_placement import descriptor, exporters, mapfile, planner, stats  # noqa: E402


def _cluster(path):
    return planner.Cluster.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_inspect(a):
    d, _ = descriptor.describe([a.gguf], a.allow_generic)
    print(json.dumps(d.summary(), indent=1))


def cmd_plan(a):
    d, h = descriptor.describe([a.gguf], a.allow_generic)
    s = stats.RoutingStats.empty(d.n_layers, d.n_expert)
    for t in a.trace or []:
        s.merge(stats.read_strata_trace(t, d.n_layers, d.n_expert))
    for mp in a.stats_from or []:
        other = mapfile.load(mp)
        for w in mapfile.validate(other, d, stats_only=True):
            print(f"note: {mp}: {w}", file=sys.stderr)
        s.merge(mapfile.stats_of(other))
    s.validate(d.n_expert_used, d.moe_layers)
    p = planner.plan(d, s, _cluster(a.cluster), allow_unmeasured=a.allow_unmeasured)
    m = mapfile.build(d, s, p, descriptor.content_sample(h, d) if a.content_sample else None)
    mapfile.validate(m, d)
    mapfile.save(m, a.out)
    e = p.expected
    print(f"wrote {a.out}: {d.architecture}, {len(d.moe_layers)} MoE layers x {d.n_expert} experts")
    for b in p.budgets:
        print(f"  {b['tier']:>12}: experts {b['expert_used'] / 1e9:7.2f} of {b['expert_budget'] / 1e9:7.2f} GB  "
              f"(dense {b['dense'] / 1e9:.2f}, KV {b['kv'] / 1e9:.2f}, state {b['state'] / 1e9:.2f}, reserve "
              f"{b['reserve'] / 1e9:.2f})  activations {e['activation_share'][b['tier']]:.1%}")
    print(f"  modelled expert time {e['modelled_expert_us_per_token'] / 1e3:.2f} ms/token (a model, not a measurement)")
    if p.unmeasured_layers:
        print(f"  unmeasured layers (prior, not data): {p.unmeasured_layers}")
    if not d.runtime_support.get("strata_native"):
        print(f"  runtime: {d.runtime_support['reason']}")


def cmd_validate(a):
    m = mapfile.load(a.map)
    d = descriptor.describe([a.gguf], a.allow_generic)[0] if a.gguf else None
    for w in mapfile.validate(m, d, stats_only=a.stats_only):
        print("warning:", w)
    if a.cluster:
        if d is None:
            raise SystemExit("--cluster needs --gguf (budgets depend on the checkpoint's dense and KV sizes)")
        if "placement" not in m:
            raise SystemExit("the map has no placement to check against a cluster")
        planner.check_fits(m["placement"], d, _cluster(a.cluster))
    print("ok:", a.map, "(placement checked)" if "placement" in m and not a.stats_only else "(stats checked)")


def cmd_export_strp(a):
    m = mapfile.load(a.map)
    d = descriptor.describe([a.gguf])[0]
    mapfile.validate(m, d)
    counts = exporters.write_strp(m, a.out, d.runtime_support)
    tiers = list(counts.items())
    args = [f"--expert-profile {a.out}", f"--expert-cache {tiers[0][1]}"]
    args += [f"--expert-cache-device{i} {n}" for i, (_, n) in enumerate(tiers[1:-1], 1)]
    print(f"wrote {a.out}; engine arguments: {' '.join(args)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("gguf")
    p.add_argument("--allow-generic", action="store_true")
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("plan")
    p.add_argument("gguf")
    p.add_argument("--cluster", required=True)
    p.add_argument("--trace", action="append", help="a Strata --dump-routing trace (repeatable)")
    p.add_argument("--stats-from", action="append", help="take routing counts from another map (repeatable)")
    p.add_argument("--allow-unmeasured", action="store_true", help="uniform prior for layers without statistics")
    p.add_argument("--allow-generic", action="store_true")
    p.add_argument("--content-sample", action="store_true", help="also hash 4 KiB of every expert tensor")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("validate")
    p.add_argument("map")
    p.add_argument("--gguf")
    p.add_argument("--stats-only", action="store_true", help="check only what is shareable across checkpoints")
    p.add_argument("--cluster")
    p.add_argument("--allow-generic", action="store_true")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("export-strp")
    p.add_argument("map")
    p.add_argument("--gguf", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export_strp)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (ValueError, OSError) as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
