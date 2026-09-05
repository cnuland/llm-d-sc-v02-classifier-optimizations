"""Enumerate every CONTIGUOUS fold of each ordered taxonomy and rank by jury
agreement.

§98 established that model accuracy tracks jury agreement across seven
taxonomies, and §100 showed collapsing complexity to the two tiers the router
consumes was worth +3.06 at zero operational cost. Both of those taxonomies were
chosen by hand -- one because Praxis routes that way, one because it was the
obvious binary.

Nobody has checked whether a BETTER fold exists. Enumerating them is a few
seconds of arithmetic on vote data already on disk, and it either finds one or
confirms the hand-picked choices were right. Contiguous folds only: these ladders
are ordered, and a fold that groups PUBLIC with NEVER_EGRESS is not a routing
decision anyone would deploy.

Reports minority share alongside, because §98's other lesson is that agreement
can be inflated by a degenerate split -- a fold that is 97% one class will show
high agreement and predict nothing.
"""
import sys, itertools, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np
from evalkit import ROOT, load_jsonl

SIGS = {
 "complexity": (["SIMPLE","MEDIUM","COMPLEX","REASONING"],
                ["complexity-real-gold.jsonl","complexity-real-contested.jsonl"]),
 "cost":       (["MINIMAL","LOW","MODERATE","HIGH"],
                ["cost-real-gold.jsonl","cost-real-contested.jsonl"]),
 "sensitivity":(["PUBLIC","INTERNAL","CONFIDENTIAL","REGULATED","NEVER_EGRESS"],
                ["sensitivity-entsec-gold.jsonl","sensitivity-entsec-contested.jsonl"]),
}
for sig, (order, files) in SIGS.items():
    rows = [r for f in files if (ROOT/"data/eval"/f).exists()
            for r in load_jsonl(ROOT/"data/eval"/f) if r.get("votes")]
    if not rows: continue
    base = np.mean([len(set(r["votes"])) == 1 for r in rows])
    print(f"\n### {sig}  n={len(rows)}   full taxonomy agreement {base:.1%}")
    print(f"  {'fold':<46}{'groups':>7}{'agreement':>11}{'minority':>10}")
    out = []
    n = len(order)
    for k in range(1, n):                       # number of cut points
        for cuts in itertools.combinations(range(1, n), k):
            gid, g = {}, 0
            bounds = [0] + list(cuts) + [n]
            for a, b in zip(bounds, bounds[1:]):
                for t in order[a:b]: gid[t] = g
                g += 1
            agr = np.mean([len({gid[v] for v in r["votes"] if v in gid}) == 1 for r in rows])
            c = collections.Counter(gid[r["tier"]] for r in rows)
            minor = min(c.values())/sum(c.values())
            name = " | ".join("+".join(x[:4] for x in order[a:b]) for a, b in zip(bounds, bounds[1:]))
            out.append((agr, minor, g, name))
    for agr, minor, g, name in sorted(out, reverse=True)[:6]:
        flag = "   <- degenerate" if minor < 0.06 else ""
        print(f"  {name:<46}{g:7d}{agr:10.1%}{minor:9.1%}{flag}")
