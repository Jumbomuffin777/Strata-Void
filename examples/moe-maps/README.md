# MoE placement map examples

- `synthetic-glm5next.map.json`: a complete map (statistics and placement) for a tiny synthetic `glm5next`
  checkpoint (4 MoE layers x 16 experts) on a two-GPU cluster. Made by the test suite; regenerate with
  `python tools/test_moe_placement.py --write-example`. Check it with
  `python tools/moe_place.py validate examples/moe-maps/synthetic-glm5next.map.json`.
- `cluster.example.json`: the cluster-spec format. One engine host with two GPUs and bounded RAM, one helper host
  over a 5 GbE-class link. The bandwidth figures are placeholders: measure your own.

Format, cost model and limits: [docs/MOE_PLACEMENT.md](../../docs/MOE_PLACEMENT.md).
