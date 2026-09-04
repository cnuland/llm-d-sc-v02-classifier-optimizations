"""Targeted synthetic data for ONE boundary: sensitivity INTERNAL/CONFIDENTIAL.

Why this boundary and not more data generally. §51's rule says corpus scaling
pays only where juror agreement is low; sensitivity's agreement is low, but §55
showed its error sits on rows where jurors AGREED, so undirected volume is the
wrong instrument. §57 localised it: a frozen probe on JUST this pair, classes
balanced, no interference, reaches only 82.9% within-train 5-fold CV. The
training corpus does not cleanly encode the distinction, so more of the same
corpus cannot teach it.

Minimal pairs are the instrument that does. Two requests identical in topic,
length and tone, differing only in the feature that moves the tier, force the
decision boundary onto that feature instead of onto register or vocabulary.
Anti-cues attack the mirror failure: CONFIDENTIAL content dressed in routine
operational language, and INTERNAL content dressed in boardroom language.

Every generated item is blind-relabelled by a second model that never sees the
proposed label; disagreements are dropped rather than resolved, because a
minimal pair whose own generator and checker disagree is exactly the ambiguous
kind this is meant to remove.
"""
import json, random, sys, collections, hashlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy, load_jsonl
import llmkit as L
from sdg import gen_pairs, gen_anticue, blind_relabel, key

SIG = "sensitivity"
A, B = "INTERNAL", "CONFIDENTIAL"
GEN = "claude-sonnet-5"
CHECK = "claude-opus-5"
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 8
labels = load_taxonomy(SIG)["labels"]

DOMAINS = [
 "quarterly close and financial reporting", "vendor and procurement management",
 "product roadmap and release planning", "pricing and discount approvals",
 "customer success and account escalations", "security operations and incident response",
 "M&A diligence and integration planning", "headcount and org design",
 "partner and channel agreements", "manufacturing capacity planning",
 "clinical trial operations", "loan portfolio and credit review",
 "marketing campaign planning", "IT asset and licence management",
 "board and investor communications", "internal audit findings",
 "data platform migration", "regulatory filings and submissions",
]

# guard against colliding with any eval row
evkeys = set()
for f in ("sensitivity-entsec-gold.jsonl", "sensitivity-entsec-contested.jsonl",
          "sensitivity-enterprise-gold.jsonl", "sensitivity-real-gold.jsonl"):
    p = ROOT/"data/eval"/f
    if p.exists():
        evkeys |= {key(r["text"]) for r in load_jsonl(p)}

jobs = []
for d in DOMAINS:
    jobs.append(("pair", d, None))
    jobs.append(("anti", d, (B, A)))    # CONFIDENTIAL that looks INTERNAL
    jobs.append(("anti", d, (A, B)))    # INTERNAL that looks CONFIDENTIAL

def run(j):
    kind, dom, ab = j
    try:
        if kind == "pair":
            return gen_pairs(SIG, labels, A, B, dom, N_PER, GEN, f"bp:{dom}")
        tier, decoy = ab
        return gen_anticue(SIG, labels, tier, decoy, dom, N_PER, GEN, f"ba:{dom}:{tier}")
    except Exception as e:
        print(f"  skip {kind}/{dom}: {type(e).__name__}", flush=True)
        return []

print(f"generating: {len(jobs)} jobs x ~{N_PER} items")
items = [it for part in L.amap(run, jobs, workers=10, desc="gen") for it in (part or [])]
print(f"  raw items: {len(items)}")

seen, cand = set(), []
for it in items:
    t, k = it.get("tier") or it.get("label"), key(it["text"])
    if t not in (A, B) or k in evkeys or k in seen:
        continue
    seen.add(k); cand.append({"text": it["text"], "tier": t})
print(f"  after dedupe/eval-overlap filter: {len(cand)}")

# blind_relabel yields dicts {"label", "confidence"}, not bare strings.
# Comparing the dict to a tier string silently made agreement 0.0% on 576 items
# -- a "quality" number that was really a type error.
chk = [x["label"] for x in blind_relabel(SIG, labels, [c["text"] for c in cand],
                                         model=CHECK, batch=20, effort="medium")]
kept = [c for c, v in zip(cand, chk) if v == c["tier"]]
import collections as _c
print(f"  checker said: {dict(_c.Counter(chk))}")
agree = len(kept)/max(1, len(cand))
print(f"  blind-check agreement: {agree:.1%}  kept {len(kept)}")
print(f"  kept mix: {dict(collections.Counter(k['tier'] for k in kept))}")

out = ROOT/"data/train/sensitivity-boundary.jsonl"
with open(out, "w") as f:
    for k in kept:
        f.write(json.dumps({**k, "source": "boundary-sdg"}) + "\n")
print(f"  -> {out}")
