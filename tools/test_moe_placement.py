"""Tests for tools/moe_placement: synthetic and deterministic; no GPU, no model download, stdlib only.

    python -m unittest tools.test_moe_placement -v        (or: python tools/test_moe_placement.py)

The checkpoints are tiny GGUF files written here (header + zero data), read back through tools/gguf_reader.py, so
the adapters see exactly the structure a real file has.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from moe_placement import descriptor, exporters, lookup, mapfile, planner, stats  # noqa: E402

F16, Q8_0, F32 = 1, 8, 0
_BYTES = {F16: (1, 2), Q8_0: (32, 34), F32: (1, 4)}


# ------------------------------------------------------------------------------------ a minimal GGUF writer
def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(k, v) -> bytes:
    if isinstance(v, bool):
        return _s(k) + struct.pack("<IB", 7, v)
    if isinstance(v, int):
        return _s(k) + struct.pack("<Ii", 5, v)
    if isinstance(v, float):
        return _s(k) + struct.pack("<If", 6, v)
    if isinstance(v, str):
        return _s(k) + struct.pack("<I", 8) + _s(v)
    if isinstance(v, list):
        return _s(k) + struct.pack("<IIQ", 9, 5, len(v)) + struct.pack(f"<{len(v)}i", *v)
    raise TypeError(v)


def write_gguf(path, meta: dict, tensors: list, align: int = 32, fill: bool = False) -> None:
    """tensors: [(name, shape, type_id)]. The data region is zeros (a sparse file), or random bytes with `fill`."""
    offs, pos = [], 0
    for _, shape, t in tensors:
        offs.append(pos)
        n = math.prod(shape)
        be, bb = _BYTES[t]
        pos += (n // be * bb + align - 1) // align * align
    out = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(meta))
    out += b"".join(_kv(k, v) for k, v in meta.items())
    for (name, shape, t), o in zip(tensors, offs):
        out += _s(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        out += struct.pack("<IQ", t, o)
    out += b"\0" * ((-len(out)) % align)
    with open(path, "wb") as f:
        f.write(out)
        if fill:
            f.write(random.Random(path.name).randbytes(pos))
        f.truncate(len(out) + pos)


def glm_like(n_expert=16, n_layers=5, embd=64, ff=32):
    """A tiny glm5next: block 0 dense, blocks 1..4 MoE, block 4 the MTP block; MLA on 1, 3, 4; KDA on 0, 2."""
    kvh = [0, 1, 0, 1, 1][:n_layers]
    meta = {"general.architecture": "glm5next", "general.name": "tiny-glm", "general.file_type": 1,
            "glm5next.block_count": n_layers, "glm5next.embedding_length": embd, "glm5next.expert_count": n_expert,
            "glm5next.expert_used_count": 2, "glm5next.expert_shared_count": 1,
            "glm5next.attention.head_count_kv": kvh, "glm5next.attention.kv_lora_rank": 16,
            "glm5next.attention.indexer.key_length": 8, "glm5next.attention.indexer.kpool": 4,
            "glm5next.kda.head_dim": 8, "glm5next.ssm.conv_kernel": 4, "glm5next.nextn_predict_layers": 1}
    t = [("token_embd.weight", [embd, 100], F16), ("output.weight", [embd, 100], F16)]
    for l in range(n_layers):
        t.append((f"blk.{l}.attn_norm.weight", [embd], F32))
        if kvh[l]:
            t.append((f"blk.{l}.attn_kv_a_mqa.weight", [embd, 16], F16))
            t.append((f"blk.{l}.indexer.attn_k.weight", [embd, 8], F16))
        else:
            t.append((f"blk.{l}.ssm_a", [4], F32))
            t.append((f"blk.{l}.ssm_conv1d_q.weight", [4, 1, 32], F32))
        if l == 0:
            t.append((f"blk.{l}.ffn_gate.weight", [embd, 128], F16))
            continue
        t.append((f"blk.{l}.ffn_gate_inp.weight", [embd, n_expert], F32))
        t.append((f"blk.{l}.ffn_gate_shexp.weight", [embd, ff], F16))
        for role in ("gate", "up"):
            t.append((f"blk.{l}.ffn_{role}_exps.weight", [embd, ff, n_expert], F16))
        t.append((f"blk.{l}.ffn_down_exps.weight", [ff, embd, n_expert], F16))
    return meta, t


def qwen_like():
    """qwen4exp at the engine's real geometry (48 x 512), with one-block expert slices to stay small."""
    meta = {"general.architecture": "qwen4exp", "general.name": "tiny-qwen", "qwen4exp.block_count": 48,
            "qwen4exp.embedding_length": 32, "qwen4exp.expert_count": 512, "qwen4exp.expert_used_count": 10}
    t = []
    for l in range(48):
        t.append((f"blk.{l}.attn_norm.weight", [32], F32))
        for role in ("gate", "up", "down"):
            t.append((f"blk.{l}.ffn_{role}_exps.weight", [32, 1, 512], Q8_0))
    return meta, t


def zipf_stats(d, tokens=400, seed=7, layers=None):
    """Routing drawn from a per-layer Zipf over a shuffled expert order: realistic skew, deterministic."""
    rng = random.Random(seed)
    s = stats.RoutingStats.empty(d.n_layers, d.n_expert, "synthetic-zipf")
    for l in layers if layers is not None else d.moe_layers:
        order = list(range(d.n_expert))
        rng.shuffle(order)
        w = [1.0 / (i + 1) ** 1.1 for i in range(d.n_expert)]
        for _ in range(tokens):
            ids, pool = [], list(range(d.n_expert))
            for _k in range(d.n_expert_used):
                i = rng.choices(pool, weights=[w[j] for j in pool])[0]
                pool.remove(i)
                ids.append(order[i])
            s.add(l, ids, [1.0 / d.n_expert_used] * d.n_expert_used)
    return s


def cluster(gpu_bytes=(300_000, 200_000), ram=10**7, context=64, cpu=20.0, links=None, extra_hosts=None):
    return planner.Cluster.from_json({
        "format": "strata.moe-cluster", "version": 1, "context_tokens": context, "sequences": 1,
        "hosts": [{"name": "local", "ram_bytes": ram, "ram_reserve_bytes": 1000, "cpu_expert_GBps": cpu,
                   "devices": [{"name": f"gpu{i}", "vram_bytes": b, "reserve_bytes": 1000, "mem_GBps": 400,
                                "link_GBps": 16} for i, b in enumerate(gpu_bytes)]}] + (extra_hosts or []),
        "links": links or []})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, maker=glm_like, name="m.gguf", **kw):
        meta, t = maker(**kw)
        p = self.dir / name
        write_gguf(p, meta, t)
        return descriptor.describe([p])[0], p


# ------------------------------------------------------------------------------------ adapters
class TestAdapters(Base):
    def test_glm_descriptor(self):
        d, _ = self.model()
        self.assertEqual(d.moe_layers, [1, 2, 3, 4])
        self.assertEqual(d.expert_bytes(1), 3 * 64 * 32 * 2)          # gate + up + down, F16
        self.assertEqual(d.mtp_layers, [4])
        self.assertGreater(d.kv_bytes_per_token[1], 0)                # MLA
        self.assertEqual(d.kv_bytes_per_token[2], 0)                  # KDA: state, no KV
        self.assertGreater(d.state_bytes[2], 0)
        self.assertGreater(d.dense_bytes[1], 0)                       # router + shared expert stay dense
        self.assertFalse(d.runtime_support["strata_native"])
        t = d.expert_tensors[2][1]
        self.assertEqual(t.slice_offset(3) - t.slice_offset(2), t.expert_bytes)

    def test_fingerprint_ignores_sharding(self):
        meta, t = glm_like()
        one = self.dir / "one.gguf"
        write_gguf(one, meta, t)
        half = len(t) // 2
        for i, part in enumerate((t[:half], t[half:]), 1):
            write_gguf(self.dir / f"two-{i:05d}-of-00002.gguf",
                       dict(meta, **{"split.no": i - 1, "split.count": 2}), part)
        a = descriptor.describe([one])[0]
        b = descriptor.describe([self.dir / "two-00001-of-00002.gguf"])[0]
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertEqual(a.total_expert_bytes(), b.total_expert_bytes())

    def test_fingerprint_sees_quantization(self):
        meta, t = glm_like()
        t2 = [(n, s, Q8_0 if "exps" in n else ty) for n, s, ty in t]
        write_gguf(self.dir / "a.gguf", meta, t)
        write_gguf(self.dir / "b.gguf", meta, t2)
        a = descriptor.describe([self.dir / "a.gguf"])[0]
        b = descriptor.describe([self.dir / "b.gguf"])[0]
        self.assertNotEqual(a.fingerprint, b.fingerprint)

    def test_unknown_architecture(self):
        meta, t = glm_like()
        meta = {k.replace("glm5next", "mystery"): v for k, v in meta.items()}
        meta["general.architecture"] = "mystery"
        write_gguf(self.dir / "x.gguf", meta, t)
        with self.assertRaises(descriptor.UnsupportedArchitecture):
            descriptor.describe([self.dir / "x.gguf"])
        d, _ = descriptor.describe([self.dir / "x.gguf"], allow_generic=True)
        self.assertFalse(d.runtime_support["strata_native"])
        self.assertTrue(any("NOT modelled" in n for n in d.notes))

    def test_qwen_is_the_engine_geometry(self):
        d, _ = self.model(qwen_like)
        self.assertTrue(d.runtime_support["strata_native"])
        self.assertEqual(sum(1 for v in d.kv_bytes_per_token.values() if v), 12)    # QSA layers 3, 7, ... 47


# ------------------------------------------------------------------------------------ statistics
class TestStats(Base):
    def test_trace_round_trip(self):
        recs = [(1, [3, 5], [0.6, 0.4]), (2, [0, 1], [0.5, 0.5]), (1, [3, 7], [0.7, 0.3])]
        p = self.dir / "t.bin"
        stats.write_strata_trace(p, recs)
        s = stats.read_strata_trace(p, 5, 16)
        self.assertEqual(s.tokens, {1: 2, 2: 1})
        self.assertEqual(s.counts[1][3], 2)
        self.assertAlmostEqual(s.freq(1, 3), 1.0)
        self.assertAlmostEqual(s.mass[1][3], 1.3, places=5)

    def test_rejects_impossible_counts(self):
        s = stats.RoutingStats.empty(5, 16)
        s.add(1, [1, 2])
        s.counts[1][4] += 5                       # 7 routings over 1 token at top-2
        with self.assertRaises(stats.StatsError):
            s.validate(2, [1, 2, 3, 4])
        with self.assertRaises(stats.StatsError):
            s.add(1, [16])
        t = stats.RoutingStats.empty(5, 16)
        t.add(0, [1, 2])
        with self.assertRaises(stats.StatsError):
            t.validate(2, [1, 2, 3, 4])           # layer 0 is dense


# ------------------------------------------------------------------------------------ the planner
class TestPlanner(Base):
    def setUp(self):
        super().setUp()
        self.d, self.path = self.model()
        self.s = zipf_stats(self.d)

    def test_deterministic(self):
        a = mapfile.dumps(mapfile.build(self.d, self.s, planner.plan(self.d, self.s, cluster())))
        b = mapfile.dumps(mapfile.build(self.d, self.s, planner.plan(self.d, self.s, cluster())))
        self.assertEqual(a, b)

    def test_budgets_hold_and_hot_experts_win(self):
        p = planner.plan(self.d, self.s, cluster())
        for b in p.budgets:
            self.assertLessEqual(b["expert_used"], b["expert_budget"], b["tier"])
            self.assertGreaterEqual(b["expert_budget"], 0)
        ram = len(p.devices) - 1
        for l in self.d.moe_layers:
            if l in self.d.mtp_layers:
                continue
            on_gpu = [self.s.freq(l, e) for e in range(16) if p.assign[l][e] != ram]
            in_ram = [self.s.freq(l, e) for e in range(16) if p.assign[l][e] == ram]
            if on_gpu and in_ram:                 # within a layer, nothing in RAM is hotter than the GPU set
                self.assertGreaterEqual(min(on_gpu), max(in_ram))
        # activation share beats byte share: the point of ranking by measured frequency
        gpu_bytes = sum(b["expert_used"] for b in p.budgets[:-1])
        self.assertGreater(p.expected["vram_hit_rate"], gpu_bytes / self.d.total_expert_bytes())

    def test_kv_and_context_take_from_experts(self):
        small = planner.plan(self.d, self.s, cluster(context=64))
        big = planner.plan(self.d, self.s, cluster(context=1024))
        self.assertGreater(big.budgets[0]["kv"], small.budgets[0]["kv"])
        self.assertLess(big.budgets[0]["expert_budget"], small.budgets[0]["expert_budget"])

    def test_refuses_when_fixed_costs_do_not_fit(self):
        with self.assertRaisesRegex(planner.PlanError, "fixed costs exceed"):
            planner.plan(self.d, self.s, cluster(gpu_bytes=(2000, 2000)))

    def test_refuses_when_ram_is_too_small(self):
        with self.assertRaisesRegex(planner.PlanError, "no NVMe tier"):
            planner.plan(self.d, self.s, cluster(gpu_bytes=(100_000, 100_000), ram=200_000))

    def test_unmeasured_layers(self):
        s = zipf_stats(self.d, layers=[1, 2])     # 3 unmeasured, 4 is MTP
        with self.assertRaisesRegex(planner.PlanError, "no routing statistics"):
            planner.plan(self.d, s, cluster())
        p = planner.plan(self.d, s, cluster(), allow_unmeasured=True)
        self.assertEqual(p.unmeasured_layers, [3, 4])
        s2 = zipf_stats(self.d, layers=[1, 2, 3])
        p2 = planner.plan(self.d, s2, cluster(gpu_bytes=(900_000, 900_000)))
        self.assertEqual(p2.unmeasured_layers, [4])   # the MTP block alone may be unmeasured: it ranks coldest,
        self.assertTrue(all(t == len(p2.devices) - 1 for t in p2.assign[4]))     # in RAM even with VRAM to spare
        self.assertTrue(all(t < len(p2.devices) - 1 for l in (1, 2, 3) for e, t in enumerate(p2.assign[l])
                            if s2.counts[l][e]))

    def test_dense_split_and_peer_cost(self):
        c = cluster()
        c.dense_split = {"gpu0": "0-2", "gpu1": "3-4"}
        p = planner.plan(self.d, self.s, c)
        self.assertEqual([p.layer_owner[l] for l in range(5)], [0, 0, 0, 1, 1])
        self.assertEqual(p.budgets[0]["layers"], "0-2")
        c.dense_split = {"gpu0": "0-2"}
        with self.assertRaisesRegex(planner.PlanError, "without a GPU"):
            planner.plan(self.d, self.s, c)

    def test_remote_advice_follows_the_link(self):
        remote = [{"name": "far", "ram_bytes": 10**9, "cpu_expert_GBps": 20,
                   "devices": [{"name": "rgpu", "vram_bytes": 10**6, "mem_GBps": 400}]}]
        slow = cluster(gpu_bytes=(100_000, 100_000), cpu=0.001, extra_hosts=remote,
                       links=[{"hosts": ["local", "far"], "GBps": 0.0001, "rtt_us": 10**6}])
        fast = cluster(gpu_bytes=(100_000, 100_000), cpu=0.001, extra_hosts=remote,
                       links=[{"hosts": ["local", "far"], "GBps": 10, "rtt_us": 1, "transport": "llama.cpp-rpc"}])
        self.assertEqual(planner.plan(self.d, self.s, slow).remote_advice[0]["layers"], [])
        adv = planner.plan(self.d, self.s, fast).remote_advice[0]
        self.assertTrue(adv["layers"])
        self.assertEqual(adv["transport"], "llama.cpp-rpc")
        self.assertIn("no cross-host transport", adv["executable_by"])


# ------------------------------------------------------------------------------------ numerical correctness
def _silu(x):
    return x / (1.0 + math.exp(-x))


def _expert(w, x):
    g = [sum(a * b for a, b in zip(row, x)) for row in w["gate"]]
    u = [sum(a * b for a, b in zip(row, x)) for row in w["up"]]
    h = [_silu(a) * b for a, b in zip(g, u)]
    return [sum(a * b for a, b in zip(row, h)) for row in w["down"]]


def moe_forward(x, router, bias, fetch, k, scale, hash_ids=None):
    """DeepSeek/GLM-style routing: sigmoid scores, a selection bias that does not enter the weights, normalized
    top-k weights. `fetch(e)` returns canonical expert e's weights - from whatever storage. Accumulation runs in
    top-k order, which is a function of canonical ids only."""
    scores = [1.0 / (1.0 + math.exp(-sum(a * b for a, b in zip(row, x)))) for row in router]
    if hash_ids is not None:
        top = list(hash_ids)
    else:
        top = sorted(range(len(scores)), key=lambda e: (-(scores[e] + bias[e]), e))[:k]
    tot = sum(scores[e] for e in top)
    out = [0.0] * len(x)
    for e in top:
        y = _expert(fetch(e), x)
        w = scores[e] / tot * scale
        out = [o + w * v for o, v in zip(out, y)]
    return out, top


class TestCorrectness(Base):
    def test_reordered_storage_is_bit_identical(self):
        rng = random.Random(3)
        n_expert, embd, ff, k = 16, 8, 4, 3

        def mat(r, c):
            return [[rng.uniform(-1, 1) for _ in range(c)] for _ in range(r)]
        canonical = [{"gate": mat(ff, embd), "up": mat(ff, embd), "down": mat(embd, ff)} for _ in range(n_expert)]
        router = mat(n_expert, embd)
        bias = [rng.uniform(-0.5, 0.5) for _ in range(n_expert)]

        d, _ = self.model(n_expert=n_expert)
        p = planner.plan(d, zipf_stats(d, tokens=200), cluster(gpu_bytes=(150_000, 100_000)))
        lk = lookup.ExpertLookup(n_expert, p.physical_order, p.segments, len(p.devices))
        for layer in d.moe_layers:
            # build the physical stores exactly as a runtime would: one array per tier, in physical order
            tiers = {t: [canonical[e] for e in p.physical_order[layer][start:start + n]]
                     for t, start, n in p.segments[layer]}
            self.assertNotEqual(p.physical_order[layer], list(range(n_expert)))   # it really was reordered

            def fetch(e, layer=layer, tiers=tiers):
                t, i = lk.locate(layer, e)
                return tiers[t][i]
            for trial in range(20):
                x = [rng.uniform(-2, 2) for _ in range(embd)]
                want, want_top = moe_forward(x, router, bias, lambda e: canonical[e], k, 2.5)
                got, got_top = moe_forward(x, router, bias, fetch, k, 2.5)
                self.assertEqual(want_top, got_top)
                self.assertEqual(want, got)       # exact equality, not a tolerance
                tid = [rng.randrange(n_expert) for _ in range(k)]     # a hash-routed layer's token -> expert row
                self.assertEqual(moe_forward(x, router, bias, lambda e: canonical[e], k, 1.5, tid)[0],
                                 moe_forward(x, router, bias, fetch, k, 1.5, tid)[0])

    def test_lookup_matches_strata_residency_form(self):
        d, _ = self.model()
        p = planner.plan(d, zipf_stats(d), cluster())
        lk = lookup.ExpertLookup(d.n_expert, p.physical_order, p.segments, len(p.devices))
        for l in d.moe_layers:
            table = lk.tier_table(l, 0)
            on0 = sorted(e for e in range(d.n_expert) if p.assign[l][e] == 0)
            self.assertEqual(sorted(e for e, s in enumerate(table) if s >= 0), on0)
            self.assertEqual(sorted(s for s in table if s >= 0), list(range(len(on0))))

    def test_lookup_rejects_bad_tables(self):
        with self.assertRaises(lookup.LookupError_):
            lookup.ExpertLookup(4, {1: [0, 1, 1, 3]}, {1: [[0, 0, 4]]}, 2)
        with self.assertRaises(lookup.LookupError_):
            lookup.ExpertLookup(4, {1: [0, 1, 2, 3]}, {1: [[0, 0, 2], [1, 3, 2]]}, 2)
        with self.assertRaises(lookup.LookupError_):
            lookup.ExpertLookup(4, {1: [0, 1, 2, 3]}, {1: [[1, 0, 2], [0, 2, 2]]}, 2)


# ------------------------------------------------------------------------------------ the map file
class TestMapFile(Base):
    def setUp(self):
        super().setUp()
        self.d, self.path = self.model()
        self.s = zipf_stats(self.d)
        self.p = planner.plan(self.d, self.s, cluster())
        self.m = mapfile.build(self.d, self.s, self.p)
        self.f = self.dir / "m.map.json"
        mapfile.save(self.m, self.f)

    def test_round_trip(self):
        m = mapfile.load(self.f)
        self.assertEqual(mapfile.validate(m, self.d), [])
        self.assertEqual(mapfile.dumps(m), self.f.read_text())
        s = mapfile.stats_of(m)
        self.assertEqual(s.counts, self.s.counts)

    def test_tampering_is_caught(self):
        def bad(mut, msg):
            m = copy.deepcopy(self.m)
            mut(m)
            with self.assertRaisesRegex(mapfile.MapError, msg):
                mapfile.validate(m, self.d)
        L = "2"
        bad(lambda m: m["placement"]["layers"][L]["physical_order"].__setitem__(0, 99), "permutation")
        bad(lambda m: m["placement"]["layers"][L]["tier"].__setitem__(
            m["placement"]["layers"][L]["physical_order"][0], 2), "assigned to tier")
        bad(lambda m: m["placement"]["budgets"][0].__setitem__("expert_budget", 0), "over its budget")
        bad(lambda m: m["placement"]["layers"][L]["rank"].__setitem__(0, 10**6), "ranks")
        bad(lambda m: m["checkpoint"].__setitem__("fingerprint", "sha256:00"), "fingerprint")
        bad(lambda m: m["checkpoint"].__setitem__("architecture", "deepseek4"), "architecture")
        bad(lambda m: m["tensors"][L][0].__setitem__("expert_bytes", 1), "expert tensors")
        bad(lambda m: m["stats"]["layers"]["1"]["counts"].__setitem__(0, 10**9), "stats")

    def test_newer_version_refused(self):
        m = json.loads(self.f.read_text())
        m["version"] = 99
        (self.dir / "new.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(mapfile.MapError, "newer"):
            mapfile.load(self.dir / "new.json")

    def test_community_stats_transfer_to_another_quantization(self):
        meta, t = glm_like()
        write_gguf(self.dir / "q8.gguf", meta, [(n, s, Q8_0 if "exps" in n else ty) for n, s, ty in t])
        d8 = descriptor.describe([self.dir / "q8.gguf"])[0]
        m = mapfile.load(self.f)
        with self.assertRaisesRegex(mapfile.MapError, "fingerprint"):
            mapfile.validate(m, d8)                                  # the placement does not transfer
        warn = mapfile.validate(m, d8, stats_only=True)              # the statistics do
        self.assertTrue(any("only the routing statistics" in w for w in warn))
        p8 = planner.plan(d8, mapfile.stats_of(m), cluster())
        self.assertEqual(mapfile.validate(mapfile.build(d8, mapfile.stats_of(m), p8), d8), [])

    def test_fits_another_cluster(self):
        planner.check_fits(self.m["placement"], self.d, cluster())
        planner.check_fits(self.m["placement"], self.d, cluster(gpu_bytes=(900_000, 900_000)))
        with self.assertRaisesRegex(planner.PlanError, "leaves"):
            planner.check_fits(self.m["placement"], self.d, cluster(context=1024))
        with self.assertRaisesRegex(planner.PlanError, "tiers"):
            planner.check_fits(self.m["placement"], self.d, cluster(gpu_bytes=(500_000,)))

    def test_runtime_support_is_not_trusted_from_the_file(self):
        m = copy.deepcopy(self.m)
        m["runtime_support"] = {"strata_native": True, "reason": "trust me"}
        warn = mapfile.validate(m, self.d)
        self.assertTrue(any("engine's applies" in w for w in warn))
        with self.assertRaises(exporters.ExportError):
            exporters.write_strp(m, self.dir / "p.bin", self.d.runtime_support)


# ------------------------------------------------------------------------------------ export to the engine
class TestStrpExport(Base):
    def test_strp_matches_make_profile_and_tiers(self):
        d, _ = self.model(qwen_like)
        s = zipf_stats(d, tokens=60)
        eb = d.expert_bytes(0)
        fixed = 200 * 2**20                                            # 36 GDN layers of state, and KV
        c = cluster(gpu_bytes=(eb * 3000 + fixed, eb * 1500 + fixed), ram=10**8, context=16)
        p = planner.plan(d, s, c)
        m = mapfile.build(d, s, p)
        out = self.dir / "expert-profile.bin"
        counts = exporters.write_strp(m, out, d.runtime_support)
        import make_profile                                        # the engine-side tool reads it unchanged
        ranked = make_profile.read_profile(out)
        self.assertEqual(len(ranked), 48 * 512)
        self.assertEqual(len(set(ranked)), 48 * 512)
        n0 = counts["gpu0"]
        self.assertGreater(n0, 0)
        self.assertEqual({tuple(x) for x in ranked[:n0]},
                         {(l, e) for l in range(48) for e in range(512) if p.assign[l][e] == 0})
        self.assertEqual(exporters.read_strp(out)[2], ranked)
        ref = self.dir / "ref.bin"
        make_profile.write_profile(ref, ranked, 512)
        self.assertEqual(out.read_bytes(), ref.read_bytes())       # byte for byte what --expert-profile reads


# ------------------------------------------------------------------------------------ hot/cold GGUF rewrite
class TestHotColdRewrite(Base):
    def test_rewrite_is_exact(self):
        from moe_placement import gguf_hotcold
        from gguf_reader import GGUFFile
        meta, t = glm_like()
        src = self.dir / "src.gguf"
        write_gguf(src, meta, t, fill=True)
        cold = {"1": [3, 0, 15], "3": list(range(8, 16)), "4": [7]}       # layer 2 stays all-hot
        dst = self.dir / "hc.gguf"
        gguf_hotcold.write(src, cold, dst, progress=None)
        a, b = GGUFFile(src), GGUFFile(dst)
        self.assertEqual(a.metadata, b.metadata)
        ra, rb = src.read_bytes(), dst.read_bytes()
        ta = {x.name: x for x in a.tensors}
        tb = {x.name: x for x in b.tensors}

        def data(g, raw, x):
            return raw[g.data_start + x.offset: g.data_start + x.offset + descriptor.tensor_bytes(x.type_name, x.shape)
                       if x.type_name != "I32" else g.data_start + x.offset + 4 * math.prod(x.shape)]
        for name, x in ta.items():
            if name in tb:
                self.assertEqual(data(a, ra, x), data(b, rb, tb[name]), name)
        for layer, c in cold.items():
            self.assertNotIn(f"blk.{layer}.ffn_gate_exps.weight", tb)
            hot = [e for e in range(16) if e not in c]
            mh = struct.unpack("<16i", data(b, rb, tb[f"blk.{layer}.ffn_exps_map_hot.weight"]))
            mc = struct.unpack("<16i", data(b, rb, tb[f"blk.{layer}.ffn_exps_map_cold.weight"]))
            for role in ("gate", "up", "down"):
                orig = data(a, ra, ta[f"blk.{layer}.ffn_{role}_exps.weight"])
                sb = len(orig) // 16
                parts = {p: data(b, rb, tb[f"blk.{layer}.ffn_{role}_exps_{p}.weight"]) for p in ("hot", "cold")}
                self.assertEqual(len(parts["hot"]), sb * (len(hot) + 1))
                self.assertEqual(parts["hot"][-sb:], bytes(sb))
                self.assertEqual(parts["cold"][-sb:], bytes(sb))
                for e in range(16):                       # every canonical expert is found exactly where its map says
                    p, m, n = ("hot", mh, len(hot)) if e in hot else ("cold", mc, len(c))
                    self.assertEqual(parts[p][m[e] * sb:(m[e] + 1) * sb], orig[e * sb:(e + 1) * sb])
                    other = mc if p == "hot" else mh
                    self.assertEqual(other[e], len(c) if p == "hot" else len(hot))   # -> the other part's zero expert
        self.assertIn("blk.2.ffn_gate_exps.weight", tb)
        d = descriptor.describe([dst], allow_generic=True)        # still a well-formed GGUF
        self.assertEqual(d[0].architecture, "glm5next")


    def test_per_slot_zero_experts_give_distinct_ids(self):
        """CUDA layers: absent experts go to zero expert `slot`, so no token repeats a hot id."""
        from moe_placement import gguf_hotcold
        from gguf_reader import GGUFFile
        meta, t = glm_like()
        src = self.dir / "src.gguf"
        write_gguf(src, meta, t, fill=True)
        cold = {"1": [3, 0, 15, 9], "2": [1, 2]}
        dst = self.dir / "slots.gguf"
        gguf_hotcold.write(src, cold, dst, progress=None, slot_zero_layers={1})
        b = GGUFFile(dst)
        rb = dst.read_bytes()
        tb = {x.name: x for x in b.tensors}
        k = 2                                                       # glm_like: expert_used_count = 2
        self.assertEqual(tb["blk.1.ffn_exps_map_hot.weight"].shape, [1, 16, k])
        self.assertEqual(tb["blk.1.ffn_gate_exps_hot.weight"].shape[-1], 16 - 4 + k)
        self.assertEqual(tb["blk.1.ffn_exps_map_cold.weight"].shape, [1, 16])   # cold (CPU) keeps one zero expert
        self.assertEqual(tb["blk.2.ffn_exps_map_hot.weight"].shape, [1, 16])    # non-CUDA layer unchanged
        x = tb["blk.1.ffn_exps_map_hot.weight"]
        raw = rb[b.data_start + x.offset: b.data_start + x.offset + 4 * 16 * k]
        table = [struct.unpack_from("<16i", raw, 64 * j) for j in range(k)]   # table[slot][expert]
        n_hot = 12
        for sel in itertools.combinations(range(16), k):       # every distinct top-k a router can produce
            for order in itertools.permutations(sel):
                ids = [table[j][e] for j, e in enumerate(order)]
                self.assertEqual(len(set(ids)), k, (order, ids))
                for j, e in enumerate(order):
                    self.assertEqual(ids[j] == n_hot + j, e in cold["1"])
        g = tb["blk.1.ffn_gate_exps_hot.weight"]
        sb = descriptor.tensor_bytes(g.type_name, g.shape[:-1])
        zeros = rb[b.data_start + g.offset + n_hot * sb: b.data_start + g.offset + (n_hot + k) * sb]
        self.assertEqual(zeros, bytes(k * sb))


# ------------------------------------------------------------------------------------ the shipped example
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "moe-maps" / "synthetic-glm5next.map.json"


def example_map(directory: Path) -> str:
    meta, t = glm_like()
    write_gguf(directory / "tiny-glm.gguf", meta, t)
    d = descriptor.describe([directory / "tiny-glm.gguf"])[0]
    s = zipf_stats(d)
    return mapfile.dumps(mapfile.build(d, s, planner.plan(d, s, cluster())))


class TestExample(Base):
    def test_example_map_is_current_and_valid(self):
        self.assertEqual(EXAMPLE.read_text(encoding="utf-8"), example_map(self.dir),
                         "regenerate: python tools/test_moe_placement.py --write-example")
        self.assertEqual(mapfile.validate(mapfile.load(EXAMPLE)), [])
        c = json.loads((EXAMPLE.parent / "cluster.example.json").read_text(encoding="utf-8"))
        planner.Cluster.from_json(c)


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-example"]:
        with tempfile.TemporaryDirectory() as t:
            EXAMPLE.write_text(example_map(Path(t)), encoding="utf-8")
        print("wrote", EXAMPLE)
    else:
        unittest.main()
