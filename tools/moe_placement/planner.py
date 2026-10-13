"""Cluster spec + ModelDescriptor + RoutingStats -> Placement, under explicit byte budgets.

THE COST MODEL (decode, one token, per routed expert activation; microseconds):

    on the GPU that owns the layer     bytes / mem_GBps
    on another GPU of the same host    bytes / mem_GBps  +  2 * act_bytes / link_GBps  +  hop_us
    in host RAM (computed by the CPU)  bytes / cpu_expert_GBps

An expert's VALUE on a device is `p * (cost_ram - cost_device)`: its measured routing frequency times the time it
saves per activation against the RAM tier. Devices are filled greedily by value per byte (the knapsack density),
which is exact for equal-size items and within one item of optimal otherwise; experts of one layer are equal-size.
Ties break on (layer, expert, device), so the same inputs always give the same plan.

THE BUDGETS, per GPU: capacity - runtime reserve - the dense weights of the layers it owns (attention, router,
shared experts, norms, any dense FFN) - their KV cache at `context_tokens x sequences` - their fixed recurrent state.
Only what is left holds routed experts. Host RAM is bounded the same way and must hold every expert no GPU takes:
there is no NVMe tier (unqualified), so a plan that does not fit is refused, not spilled.

Cross-host devices never receive individual experts: no Strata transport moves expert rows between hosts. They get an
advisory per-layer comparison (`remote_advice`) against the network cost, for runtimes that place whole tensors on a
remote backend (llama.cpp RPC).
"""
from __future__ import annotations

import dataclasses
import re

from . import CLUSTER_FORMAT, CLUSTER_VERSION
from .descriptor import ModelDescriptor
from .stats import RoutingStats

HOST = "host"          # the device-name suffix for a host's RAM tier: "<host>:ram"


class PlanError(ValueError):
    pass


_UNITS = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12,
          "KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40}


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if abs(n) >= 10**7 else f"{n} B"


def parse_bytes(v) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"\s*([0-9.]+)\s*([A-Za-z]*)\s*", str(v))
    if not m or m.group(2).upper() not in _UNITS:
        raise PlanError(f"cannot read a byte size from {v!r}")
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()])


@dataclasses.dataclass
class Device:
    name: str
    host: str
    vram: int
    reserve: int
    mem_GBps: float
    link_GBps: float


@dataclasses.dataclass
class Host:
    name: str
    ram: int
    ram_reserve: int
    cpu_expert_GBps: float
    devices: list


@dataclasses.dataclass
class Link:
    a: str
    b: str
    GBps: float
    rtt_us: float
    transport: str


@dataclasses.dataclass
class Cluster:
    hosts: list
    links: list
    context_tokens: int
    sequences: int
    hop_us: float = 10.0
    dense_split: object = "auto"

    @property
    def primary(self) -> Host:
        return self.hosts[0]

    @classmethod
    def from_json(cls, j: dict) -> "Cluster":
        if j.get("format") != CLUSTER_FORMAT:
            raise PlanError(f"cluster spec: format must be '{CLUSTER_FORMAT}'")
        if int(j.get("version", 0)) > CLUSTER_VERSION:
            raise PlanError(f"cluster spec version {j['version']} is newer than this tool ({CLUSTER_VERSION})")
        hosts = []
        for h in j["hosts"]:
            devs = [Device(d["name"], h["name"], parse_bytes(d["vram_bytes"]), parse_bytes(d.get("reserve_bytes", 0)),
                           float(d["mem_GBps"]), float(d.get("link_GBps", 16))) for d in h.get("devices", [])]
            hosts.append(Host(h["name"], parse_bytes(h["ram_bytes"]), parse_bytes(h.get("ram_reserve_bytes", 0)),
                              float(h["cpu_expert_GBps"]), devs))
        if not hosts or not hosts[0].devices:
            raise PlanError("cluster spec: the first host runs the engine and needs at least one GPU")
        names = [d.name for h in hosts for d in h.devices] + [h.name for h in hosts]
        if len(set(names)) != len(names):
            raise PlanError("cluster spec: device and host names must be unique")
        links = [Link(l["hosts"][0], l["hosts"][1], float(l["GBps"]), float(l.get("rtt_us", 0)),
                      l.get("transport", "none")) for l in j.get("links", [])]
        return cls(hosts, links, int(j.get("context_tokens", 4096)), int(j.get("sequences", 1)),
                   float(j.get("hop_us", 10.0)), j.get("dense_split", "auto"))


@dataclasses.dataclass
class Placement:
    devices: list            # tier names, index = device id in `assign`; the primary host's GPUs, then its RAM
    layer_owner: dict        # layer -> GPU index holding the layer's dense weights and KV
    assign: dict             # layer -> [tier index per canonical expert]
    rank: dict               # layer -> [global priority rank per canonical expert], 0 = most valuable
    physical_order: dict     # layer -> [canonical expert id at each physical index]
    segments: dict           # layer -> [[tier, start, count]] contiguous physical ranges, in tier order
    budgets: list            # per tier: capacity / fixed costs / expert budget / expert bytes used
    expected: dict           # activation-weighted hit rates and modelled expert time per token
    params: dict             # the cost-model inputs, recorded with the plan
    unmeasured_layers: list
    remote_advice: list


def _split_layers(c: Cluster, d: ModelDescriptor) -> dict:
    gpus = c.primary.devices
    if isinstance(c.dense_split, dict):
        owner = {}
        for i, g in enumerate(gpus):
            spec = c.dense_split.get(g.name)
            for part in (spec or "").split(","):
                if part:
                    lo, _, hi = part.partition("-")
                    for l in range(int(lo), int(hi or lo) + 1):
                        owner[l] = i
        missing = [l for l in range(d.n_layers) if l not in owner]
        if missing:
            raise PlanError(f"dense_split leaves layers {missing} without a GPU")
        return owner
    usable = [max(g.vram - g.reserve, 0) for g in gpus]
    total = sum(usable)
    if total <= 0:
        raise PlanError("no usable VRAM on the primary host")
    owner, cum, i = {}, 0.0, 0
    bounds = []
    for u in usable:
        cum += u / total
        bounds.append(cum)
    for l in range(d.n_layers):
        x = (l + 0.5) / d.n_layers
        while i < len(bounds) - 1 and x > bounds[i]:
            i += 1
        owner[l] = i
    return owner


def budgets_for(d: ModelDescriptor, c: Cluster) -> tuple[list, dict, list]:
    """(tier names, layer -> owning GPU, per-tier budgets): fixed costs first, what is left holds experts."""
    gpus = c.primary.devices
    ram_tier = len(gpus)
    names = [g.name for g in gpus] + [f"{c.primary.name}:ram"]
    owner = _split_layers(c, d)
    budgets = []
    for i, g in enumerate(gpus):
        owned = [l for l in range(d.n_layers) if owner[l] == i]
        dense = sum(d.dense_bytes[l] for l in owned)
        kv = int(sum(d.kv_bytes_per_token[l] for l in owned) * c.context_tokens * c.sequences)
        state = sum(d.state_bytes[l] for l in owned) * c.sequences
        budgets.append({"tier": g.name, "capacity": g.vram, "reserve": g.reserve, "dense": dense, "kv": kv,
                        "state": state, "layers": f"{min(owned)}-{max(owned)}" if owned else ""})
    last = owner[d.n_layers - 1]
    for name, b in d.global_bytes.items():
        if name == "token_embd.weight":     # a row lookup: llama.cpp and Strata both keep it in host memory
            continue
        budgets[last]["dense"] += b
    host = c.primary
    embd_host = d.global_bytes.get("token_embd.weight", 0)
    budgets.append({"tier": names[ram_tier], "capacity": host.ram, "reserve": host.ram_reserve, "dense": embd_host,
                    "kv": 0, "state": 0, "layers": ""})
    for b in budgets:
        b["expert_budget"] = b["capacity"] - b["reserve"] - b["dense"] - b["kv"] - b["state"]
        b["expert_used"] = 0
        if b["expert_budget"] < 0:
            raise PlanError(f"{b['tier']}: fixed costs exceed capacity by {gb(-b['expert_budget'])} (reserve "
                            f"{gb(b['reserve'])}, dense {gb(b['dense'])}, KV {gb(b['kv'])} at {c.context_tokens} "
                            f"tokens x {c.sequences}, state {gb(b['state'])}); reduce "
                            "context_tokens or sequences, or move layers with dense_split")
    return names, owner, budgets


def check_fits(placement_json: dict, d: ModelDescriptor, c: Cluster) -> None:
    """A map's placement on THIS cluster: same tiers and layer owners, and every tier's experts within the budget
    this cluster leaves (its own capacities, reserves, context and sequences)."""
    names, owner, budgets = budgets_for(d, c)
    if names != placement_json["tiers"]:
        raise PlanError(f"the map's tiers {placement_json['tiers']} are not this cluster's {names}")
    if {str(l): t for l, t in owner.items()} != placement_json["layer_owner"]:
        raise PlanError("the map's layer owners differ from this cluster's dense_split")
    for mine, theirs in zip(budgets, placement_json["budgets"]):
        if theirs["expert_used"] > mine["expert_budget"]:
            raise PlanError(f"{mine['tier']}: the map puts {gb(theirs['expert_used'])} of experts there, "
                            f"this cluster leaves {gb(mine['expert_budget'])}")


def plan(d: ModelDescriptor, stats: RoutingStats, c: Cluster, allow_unmeasured: bool = False) -> Placement:
    gpus = c.primary.devices
    n_gpu = len(gpus)
    ram_tier = n_gpu
    host = c.primary
    names, owner, budgets = budgets_for(d, c)

    # ---- frequencies: measured, or an explicit uniform prior
    unmeasured, p = [], {}
    k = d.n_expert_used or 1
    for l in d.moe_layers:
        if stats.tokens.get(l):
            p[l] = [stats.freq(l, e) for e in range(d.n_expert)]
        elif l in d.mtp_layers:
            p[l] = [0.0] * d.n_expert          # the MTP block runs only when drafting: coldest until measured
            unmeasured.append(l)
        elif allow_unmeasured:
            p[l] = [k / d.n_expert] * d.n_expert
            unmeasured.append(l)
        else:
            raise PlanError(f"no routing statistics for MoE layers {_short([l for l in d.moe_layers if not stats.tokens.get(l)])}"
                            "; measure them, or pass allow_unmeasured for a uniform prior")

    # ---- the cost model
    act = d.act_bytes()

    def us(nbytes, GBps):
        return nbytes / (GBps * 1e3)

    def cost(l, tier):
        b = d.expert_bytes(l)
        if tier == ram_tier:
            return us(b, host.cpu_expert_GBps)
        g = gpus[tier]
        t = us(b, g.mem_GBps)
        if tier != owner[l]:
            t += 2 * us(act, min(g.link_GBps, gpus[owner[l]].link_GBps)) + c.hop_us
        return t

    cand = []
    for l in d.moe_layers:
        b = d.expert_bytes(l)
        ram_cost = cost(l, ram_tier)
        for t in range(n_gpu):
            save = ram_cost - cost(l, t)
            if save <= 0:
                continue
            for e in range(d.n_expert):
                if p[l][e] > 0:                    # an expert never routed saves nothing anywhere
                    cand.append((-(p[l][e] * save) / b, -p[l][e], l, e, t))
    cand.sort()

    assign = {l: [-1] * d.n_expert for l in d.moe_layers}
    rank = {l: [-1] * d.n_expert for l in d.moe_layers}
    nxt = 0
    for _, _, l, e, t in cand:
        if assign[l][e] >= 0:
            continue
        b = d.expert_bytes(l)
        if budgets[t]["expert_used"] + b > budgets[t]["expert_budget"]:
            continue
        budgets[t]["expert_used"] += b
        assign[l][e] = t
        rank[l][e] = nxt
        nxt += 1
    rest = sorted((-p[l][e], l, e) for l in d.moe_layers for e in range(d.n_expert) if assign[l][e] < 0)
    for _, l, e in rest:
        assign[l][e] = ram_tier
        rank[l][e] = nxt
        nxt += 1
        budgets[ram_tier]["expert_used"] += d.expert_bytes(l)
    if budgets[ram_tier]["expert_used"] > budgets[ram_tier]["expert_budget"]:
        over = budgets[ram_tier]["expert_used"] - budgets[ram_tier]["expert_budget"]
        raise PlanError(f"{names[ram_tier]}: the experts no GPU holds need {gb(over)} more than the bounded "
                        "RAM budget; there is no NVMe tier (unqualified), so the plan is refused")

    # ---- physical order: tiers in device order, within a tier by priority
    order, segs = {}, {}
    for l in d.moe_layers:
        po = sorted(range(d.n_expert), key=lambda e: (assign[l][e], rank[l][e]))
        order[l] = po
        s, start = [], 0
        for t in range(len(names)):
            n = sum(1 for e in po if assign[l][e] == t)
            if n:
                s.append([t, start, n])
                start += n
        segs[l] = s

    # ---- what the plan expects
    tot = sum(sum(p[l]) for l in d.moe_layers) or 1.0
    by_tier = [0.0] * len(names)
    t_us = 0.0
    for l in d.moe_layers:
        for e in range(d.n_expert):
            by_tier[assign[l][e]] += p[l][e]
            t_us += p[l][e] * cost(l, assign[l][e])
    expected = {"activation_share": {names[i]: round(v / tot, 6) for i, v in enumerate(by_tier)},
                "vram_hit_rate": round(sum(by_tier[:n_gpu]) / tot, 6),
                "modelled_expert_us_per_token": round(t_us, 1),
                "note": "modelled from the cost parameters, not measured"}
    params = {"context_tokens": c.context_tokens, "sequences": c.sequences, "hop_us": c.hop_us,
              "act_bytes": act, "cpu_expert_GBps": host.cpu_expert_GBps,
              "gpus": {g.name: {"mem_GBps": g.mem_GBps, "link_GBps": g.link_GBps} for g in gpus}}
    pl = Placement(names, owner, assign, rank, order, segs, budgets, expected, params, unmeasured, [])
    pl.remote_advice = remote_advice(d, p, pl, c, cost)
    return pl


def remote_advice(d, p, pl, c, cost) -> list:
    """For each GPU on ANOTHER host reachable by a link: the MoE layers whose whole expert set would cost less there
    (remote compute + one activation round trip per token) than where this plan keeps them, most saving first, as
    many as that GPU's VRAM holds. Advisory: executing it needs a runtime that places whole tensors remotely."""
    out = []
    for link in c.links:
        if c.primary.name not in (link.a, link.b):
            continue
        other = link.b if link.a == c.primary.name else link.a
        host = next((h for h in c.hosts if h.name == other), None)
        if host is None:
            continue
        rt = 2 * d.act_bytes() / (link.GBps * 1e3) + link.rtt_us
        for g in host.devices:
            rows = []
            for l in d.moe_layers:
                here = sum(p[l][e] * cost(l, pl.assign[l][e]) for e in range(d.n_expert))
                there = sum(p[l]) * d.expert_bytes(l) / (g.mem_GBps * 1e3) + rt
                if there < here:
                    rows.append((here - there, l, d.expert_bytes(l) * d.n_expert))
            rows.sort(key=lambda r: (-r[0], r[1]))
            room, picked = g.vram - g.reserve, []
            for save, l, nbytes in rows:
                if nbytes <= room:
                    room -= nbytes
                    picked.append({"layer": l, "saves_us_per_token": round(save, 1)})
            out.append({"device": g.name, "host": other, "transport": link.transport,
                        "round_trip_us_per_layer": round(rt, 1), "layers": picked,
                        "executable_by": "a runtime with whole-layer remote expert tensors (llama.cpp RPC); "
                                         "native Strata has no cross-host transport"})
    return out


def _short(xs) -> str:
    from .descriptor import _ranges
    return _ranges(xs)
