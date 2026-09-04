"""Fold the sensitivity ladder into the binary egress gate.

§74's motivation: of every taxonomy variant measured, the NEVER_EGRESS block/allow
decision is the only one whose JURY AGREEMENT reaches the high 90s (96.4%). It is
therefore the only one where a high-90s model score is a real target rather than
an artefact of noisy labels — and it happens to be the decision with the most at
stake.

Folding is done on the ORIGINAL tier, not on a model's prediction, and contested
rows are folded too: a row where jurors split PUBLIC/INTERNAL is unanimous once
both fold to ALLOW, which is exactly the agreement the merge buys.
"""
import sys, json, glob, collections, pathlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy

LAD = load_taxonomy("sensitivity")["labels"]
rank = {l: i for i, l in enumerate(LAD)}
THR = rank["NEVER_EGRESS"]
fold = lambda t: "BLOCK" if rank.get(t, -1) >= THR else "ALLOW"

def convert(src, dst):
    n = collections.Counter(); out = []
    for l in open(src):
        r = json.loads(l)
        t = r.get("tier") or r.get("label")
        if t not in rank: continue
        r = dict(r); r["tier"] = fold(t)
        if isinstance(r.get("votes"), list):
            r["votes"] = [fold(v) for v in r["votes"]]
        r.pop("soft_dist", None)          # distribution is over 5 tiers, now stale
        n[r["tier"]] += 1; out.append(r)
    pathlib.Path(dst).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"  {pathlib.Path(src).name:<42} -> {pathlib.Path(dst).name:<38} {dict(n)}")

print("train:")
for p in sorted(glob.glob(str(ROOT/"data/train/sensitivity-*.jsonl"))):
    b = pathlib.Path(p).name
    if any(x in b for x in ("balanced", "boundary", "greyzone")): continue
    convert(p, ROOT/"data/train"/b.replace("sensitivity-", "egress-"))
print("eval:")
for p in sorted(glob.glob(str(ROOT/"data/eval/sensitivity-*.jsonl"))):
    b = pathlib.Path(p).name
    convert(p, ROOT/"data/eval"/b.replace("sensitivity-", "egress-"))
