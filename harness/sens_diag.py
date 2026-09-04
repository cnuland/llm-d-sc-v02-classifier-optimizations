"""Where does sensitivity lose? Separate RUBRIC ambiguity from MODEL error.

Two questions, deliberately kept apart:
  1. juror disagreement, normalised by how often a boundary is even AT RISK
     (the diagnostic that found complexity's MEDIUM/COMPLEX at 2.2x)
  2. model error conditioned on juror agreement -- if the model errs where all
     jurors AGREED, the rubric is not what is capping the score.
"""
import json, sys, collections, itertools, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

LAB = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "REGULATED", "NEVER_EGRESS"]

def load(p):
    return [json.loads(l) for l in open(p)]

rows = load("data/eval/sensitivity-entsec-gold.jsonl") + load("data/eval/sensitivity-entsec-contested.jsonl")
print(f"n={len(rows)} rows with juror votes\n")

# --- 1. disagreement per unordered pair, normalised by opportunity -----------
split = collections.Counter()
opp   = collections.Counter()
nsplit = 0
for r in rows:
    v = r.get("votes") or {}
    cast = list(v) if isinstance(v, list) else [lab for lab, n in v.items() for _ in range(int(n))]
    if len(cast) < 2:
        continue
    distinct = sorted(set(cast))
    # every pair of jurors is an OPPORTUNITY for the pairs of labels in play;
    # count opportunity as: this row could have split on any pair among the
    # labels its jurors actually considered plausible + its gold neighbours.
    for a, b in itertools.combinations(sorted(set(cast) | {r["tier"]}), 2):
        opp[(a, b)] += 1
    if len(distinct) > 1:
        nsplit += 1
        for a, b in itertools.combinations(distinct, 2):
            split[(a, b)] += 1

tot_s = sum(split.values()) or 1
tot_o = sum(opp.values()) or 1
print(f"rows where jurors split: {nsplit}/{len(rows)} = {nsplit/len(rows):.1%}\n")
print(f"{'boundary':34} {'splits':>7} {'%split':>7} {'%opp':>7} {'over-rep':>9}")
out = []
for pair, c in split.most_common():
    o = opp[pair]
    ps, po = c / tot_s, o / tot_o
    out.append((ps / po if po else 0, pair, c, ps, po))
for ratio, pair, c, ps, po in sorted(out, reverse=True):
    if c < 3: continue
    print(f"{pair[0]+'/'+pair[1]:34} {c:7d} {ps:6.1%} {po:6.1%} {ratio:8.2f}x")
