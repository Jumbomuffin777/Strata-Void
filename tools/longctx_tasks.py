#!/usr/bin/env python3
"""Long-context evaluation tasks with known answers: the same question over material of a chosen size.

    python tools/longctx_tasks.py --tokens 8000,16000,32000,64000,128000 --out bench/longctx_tasks.json

Each task is one request: a long body of realistic documents (contracts of many vendors, quarterly reports of many
business units, the configuration reference of a large system) and a question whose answer needs facts from
documents far apart in the material - so a step that reads only part of it cannot answer alone.  The documents are
generated (seeded, deterministic) so the right answer is known exactly; checks: "numbers" (every expected figure
must appear in the answer) and "contains_all" (every listed name must be mentioned).  Sizes are approximate, in the
tokens of the Qwen3.x tokenizer (measured: about 5.2 / 4.8 / 4.3 characters a token for the three kinds).
"""
from __future__ import annotations

import argparse
import json
import random

ADJ = ["Northern", "Blue", "Granite", "Silver", "Rapid", "Harbor", "Summit", "Atlas", "Cedar", "Orion", "Pioneer",
       "Vertex", "Lumen", "Keystone", "Meridian", "Redwood", "Quantum", "Beacon", "Falcon", "Evergreen", "Iron",
       "Crystal", "Delta", "Echo", "Nimbus", "Polar", "Sterling", "Zenith", "Maple", "Coral"]
NOUN = ["Systems", "Logistics", "Analytics", "Networks", "Labs", "Holdings", "Partners", "Works", "Data", "Solutions",
        "Dynamics", "Industries", "Software", "Cloud", "Security", "Robotics", "Energy", "Health", "Media", "Foods"]
SERVICES = ["cloud hosting", "payroll processing", "fleet telematics", "managed security monitoring", "data warehousing",
            "customer support outsourcing", "document storage", "network maintenance", "travel booking",
            "facilities cleaning", "legal research", "translation services", "email marketing", "office supplies"]
BOILER = [
    "The parties agree that this Agreement constitutes the entire understanding between them with respect to its "
    "subject matter and supersedes all prior negotiations, representations and agreements, whether written or oral.",
    "Each party shall keep confidential all non-public information disclosed by the other party, using at least the "
    "same degree of care it uses to protect its own confidential information, and no less than reasonable care.",
    "Neither party shall be liable for any failure or delay in performance caused by events beyond its reasonable "
    "control, including acts of nature, war, terrorism, labor disputes, or failures of public utilities.",
    "All notices under this Agreement shall be in writing and delivered by courier or electronic mail to the "
    "addresses set out in the order form, and shall be effective upon receipt.",
    "The Supplier shall comply with all applicable laws and regulations, including anti-corruption, export control "
    "and data protection laws, in the performance of the Services.",
    "Invoices are payable within thirty days of receipt. Disputed amounts shall be notified in writing within "
    "fifteen days, and the undisputed portion shall be paid when due.",
    "This Agreement shall be governed by the laws of the jurisdiction stated in the order form, and the courts of "
    "that jurisdiction shall have exclusive jurisdiction over any dispute arising out of it.",
    "The Supplier shall maintain insurance coverage appropriate to the Services with reputable insurers, and shall "
    "provide certificates of insurance upon reasonable request.",
]


def _names(rng, n):
    out, seen = [], set()
    while len(out) < n:
        nm = f"{rng.choice(ADJ)} {rng.choice(NOUN)}"
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


def contracts(target_tokens: int, seed: int) -> dict:
    """Vendor contracts.  Q: which vendors may be terminated for convenience on less than 60 days' notice, their total
    annual fees, and which contract has a liability cap below its annual fees."""
    rng = random.Random(seed)
    per = 707                                                    # tokens per contract (measured, Qwen3.x tokenizer)
    n = max(6, target_tokens // per)
    names = _names(rng, n)
    docs, short, low_cap = [], [], []
    assert n <= len(ADJ) * len(NOUN)
    # exactly 3 vendors (spread out) have a notice period under 60 days; exactly 2 have a cap below the fees
    short_idx = sorted(rng.sample(range(n), 3))
    while True:
        cap_idx = sorted(rng.sample(range(n), 2))
        if not set(cap_idx) & set(short_idx):
            break
    total = 0
    for i, nm in enumerate(names):
        fees = rng.randrange(40, 900) * 1000
        notice = rng.choice([15, 30, 45]) if i in short_idx else rng.choice([60, 90, 120, 180])
        cap = fees // 2 if i in cap_idx else fees * rng.choice([2, 3, 5])
        if i in short_idx:
            short.append(nm)
            total += fees
        if i in cap_idx:
            low_cap.append(nm)
        svc = rng.choice(SERVICES)
        sla = rng.choice(["99.5%", "99.9%", "99.95%", "99.0%"])
        body = [f"# Contract {i + 1}: Master Services Agreement with {nm}",
                f"\n## 1. Services\n{nm} (the Supplier) shall provide {svc} to the Customer as described in the order "
                f"form, with an availability commitment of {sla} measured monthly.",
                f"\n## 2. Fees\nThe annual fees for the Services are ${fees:,} payable quarterly in advance. "
                + rng.choice(BOILER),
                f"\n## 3. Term and termination\nThe initial term is {rng.choice([1, 2, 3])} years. The Customer may "
                f"terminate this Agreement for convenience by giving {notice} days' written notice. Either party may "
                "terminate for material breach not cured within thirty days of notice. " + rng.choice(BOILER),
                f"\n## 4. Limitation of liability\nEach party's aggregate liability under this Agreement shall not "
                f"exceed ${cap:,}, except for breaches of confidentiality and indemnification obligations. "
                + rng.choice(BOILER)]
        for k in range(5, 9):
            body.append(f"\n## {k}. {rng.choice(['Confidentiality', 'Compliance', 'Notices', 'Insurance', 'Governing law', 'Payment terms', 'Force majeure'])}\n"
                        + " ".join(rng.sample(BOILER, 3)))
        docs.append("\n".join(body))
    q = ("You are reviewing the vendor contracts above for the Customer. Answer precisely: (1) Which vendors can the "
         "Customer terminate for convenience with LESS than 60 days' notice? (2) What are the total annual fees of "
         "exactly those vendors? (3) Which contracts have a liability cap LOWER than their own annual fees? Give the "
         "vendor names and the figures, then a short recommendation.")
    text = "VENDOR CONTRACTS\n\n" + "\n\n---\n\n".join(docs) + "\n\n" + q
    return {"prompt": text, "check": {"type": "longctx", "contains_all": short + low_cap,
                                      "numbers": [total], "short_notice": short, "low_cap": low_cap,
                                      "total_fees": total}}


def reports(target_tokens: int, seed: int) -> dict:
    """Quarterly reports of business units.  Q: total Q4 revenue, the unit with the highest Q4 operating margin, and
    the units whose Q4 revenue fell from Q3."""
    rng = random.Random(seed + 7)
    per = 210
    n = max(4, target_tokens // per)
    units = [f"{nm} Division" for nm in _names(rng, n)]
    docs = []
    total_q4, best, best_m, fell = 0, None, -1.0, []
    fell_idx = set(rng.sample(range(n), 3))
    for i, u in enumerate(units):
        q3 = rng.randrange(20, 400) * 100_000
        q4 = int(q3 * (rng.uniform(0.80, 0.97) if i in fell_idx else rng.uniform(1.01, 1.25))) // 10_000 * 10_000
        if i in fell_idx:
            fell.append(u)
        cost = int(q4 * rng.uniform(0.70, 0.93)) // 10_000 * 10_000
        margin = (q4 - cost) / q4
        if margin > best_m + 1e-9:
            best_m, best = margin, u
        total_q4 += q4
        head = rng.choice(["steady demand", "a new enterprise contract", "pricing pressure", "a supply delay",
                           "seasonal effects", "a product launch", "currency headwinds"])
        docs.append(
            f"# Quarterly report: {u}\n\n## Summary\nIn the fourth quarter the {u} reported revenue of ${q4:,} against "
            f"${q3:,} in the third quarter, driven by {head}. Operating costs in the fourth quarter were ${cost:,}.\n\n"
            f"## Operations\nHeadcount ended the quarter at {rng.randrange(40, 900)}. "
            + " ".join(rng.sample(["The unit completed its migration to the new ERP system.",
                                   "Customer satisfaction scores improved for the second consecutive quarter.",
                                   "Two senior managers joined from competitors.",
                                   "A warehouse consolidation is planned for next year.",
                                   "The unit renegotiated three supplier contracts.",
                                   "Inventory turns improved slightly compared with last year.",
                                   "The sales pipeline for the next two quarters is described as healthy."], 4))
            + "\n\n## Outlook\nManagement expects " + rng.choice(["moderate growth", "flat revenue",
                                                                   "a recovery in demand", "continued pressure"])
            + " next quarter. " + " ".join(rng.sample(BOILER, 2)))
    q = ("Using the quarterly reports above, answer precisely: (1) the TOTAL fourth-quarter revenue of all units "
         "together; (2) which unit had the HIGHEST fourth-quarter operating margin ((revenue - operating costs) / "
         "revenue), and that margin; (3) which units' fourth-quarter revenue was LOWER than their third-quarter "
         "revenue. Show the key figures.")
    text = "BUSINESS UNIT REPORTS\n\n" + "\n\n---\n\n".join(docs) + "\n\n" + q
    return {"prompt": text, "check": {"type": "longctx", "contains_all": [best] + fell, "numbers": [total_q4],
                                      "best_margin_unit": best, "best_margin": round(best_m * 100, 1),
                                      "fell": fell, "total_q4": total_q4}}


def config_ref(target_tokens: int, seed: int) -> dict:
    """A system's configuration reference.  Q: a value that is defined in one component in terms of a parameter of
    another, far away - the answer needs both sections."""
    rng = random.Random(seed + 13)
    per = 282
    n = max(5, target_tokens // per)
    comps = [f"{nm.split()[0].lower()}-{nm.split()[1].lower()}-{rng.choice(['gateway', 'scheduler', 'cache', 'indexer', 'broker', 'store', 'router', 'worker', 'auditor', 'billing'])}"
             for nm in _names(rng, min(n, len(ADJ) * len(NOUN)))]
    comps = list(dict.fromkeys(comps))
    n = len(comps)
    a, b = sorted(rng.sample(range(n), 2))
    base = rng.choice([250, 400, 750, 1200])
    mult = rng.choice([3, 4, 6])
    docs = []
    for i, c in enumerate(comps):
        params = []
        for k in range(6):
            p = f"{c.split('-')[-1]}.{rng.choice(['pool_size', 'retry_limit', 'flush_interval_ms', 'max_batch', 'queue_depth', 'ttl_s', 'shards'])}_{k}"
            params.append(f"- `{p}` (default {rng.randrange(1, 5000)}): " + rng.choice(
                ["controls how many items are processed together.", "limits retries before an error is raised.",
                 "sets the interval between background flushes.", "bounds the memory used by the queue.",
                 "defines how long cached entries live."]))
        if i == a:
            params.append(f"- `{c}.base_timeout_ms` (default {base}): the base timeout every dependent component "
                          "builds on.")
        if i == b:
            params.append(f"- `{c}.request_timeout_ms`: set to {mult} times `{comps[a]}.base_timeout_ms` (see the "
                          f"{comps[a]} section); not configurable separately.")
        docs.append(f"# Component: {c}\n\n## Purpose\nThe {c} " + rng.choice(
            ["accepts external requests and forwards them.", "schedules background jobs.", "caches hot objects.",
             "indexes documents for search.", "brokers messages between services.", "stores durable records."])
            + " " + " ".join(rng.sample(BOILER, 1)) + "\n\n## Parameters\n" + "\n".join(params)
            + "\n\n## Operations\n" + " ".join(rng.sample(BOILER, 2)))
    q = (f"Using the configuration reference above: what is the effective default request timeout of the "
         f"{comps[b]}, in milliseconds, and which parameters (of which components) determine it? Also list three "
         "parameters from different components that an operator would most likely tune for throughput, with their "
         "defaults.")
    text = "CONFIGURATION REFERENCE\n\n" + "\n\n---\n\n".join(docs) + "\n\n" + q
    return {"prompt": text, "check": {"type": "longctx", "contains_all": [comps[a], comps[b]],
                                      "numbers": [base * mult], "timeout_ms": base * mult}}


KINDS = {"contracts": contracts, "reports": reports, "config": config_ref}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8000,16000,32000,64000,128000")
    ap.add_argument("--kinds", default="contracts,reports,config")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = []
    for kind in a.kinds.split(","):
        for t in (int(x) for x in a.tokens.split(",")):
            d = KINDS[kind](t, a.seed + t)
            tasks.append({"id": f"{kind}-{t // 1000}k", "category": f"long-context {kind}", "decomposable": True,
                          "tokens_target": t, "prompt": d["prompt"], "check": d["check"]})
    json.dump(tasks, open(a.out, "w"), indent=1)
    for t in tasks:
        print(t["id"], len(t["prompt"]), "chars ~", int(len(t["prompt"]) / 4.8), "tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
