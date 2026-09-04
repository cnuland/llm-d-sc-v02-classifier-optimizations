"""Fold any signal's ladder into a binary DEPLOYED decision.

Generalises fold_egress.py after §75 measured all three signals' real routing
decisions. The folds here are branches that exist in the deployment, not
convenient regroupings: a classifier that says MEDIUM where the jury said SIMPLE
routes to the same backend and has made no error, but tier-exact accuracy counts
it as one.

Folding is done on the ORIGINAL tier and on the jury's votes, so a row where
jurors split SIMPLE/MEDIUM becomes unanimous once both fold to "small" -- that
recovered agreement is exactly what the fold buys, and it must be measured, not
assumed.
"""
import sys, json, glob, collections, pathlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT

TABLES = {
 "reasoning": ("complexity", {"SIMPLE":"NO","MEDIUM":"NO","COMPLEX":"NO","REASONING":"YES"}),
 "route":     ("complexity", {"SIMPLE":"SMALL","MEDIUM":"SMALL","COMPLEX":"LARGE","REASONING":"LARGE"}),
 "genlen":    ("cost",       {"MINIMAL":"SHORT","LOW":"SHORT","MODERATE":"LONG","HIGH":"LONG"}),
 "egress":    ("sensitivity",{"PUBLIC":"ALLOW","INTERNAL":"ALLOW","CONFIDENTIAL":"ALLOW",
                              "REGULATED":"ALLOW","NEVER_EGRESS":"BLOCK"}),
}
name = sys.argv[1]
src_sig, table = TABLES[name]
labels = sorted(set(table.values()))
pathlib.Path(ROOT/"classifiers").mkdir(exist_ok=True)
(ROOT/"classifiers"/f"{name}.json").write_text(json.dumps(
    {"labels": labels,
     "note": f"Binary deployed decision folded out of {src_sig}: {table}. "
             f"Derived taxonomy local to the accuracy project; genesis untouched. See FINDINGS 75."},
    indent=1) + "\n")

def convert(src, dst):
    n = collections.Counter(); out = []
    for l in open(src):
        r = json.loads(l)
        t = r.get("tier") or r.get("label")
        if t not in table: continue
        r = dict(r); r["tier"] = table[t]
        if isinstance(r.get("votes"), list):
            r["votes"] = [table[v] for v in r["votes"] if v in table] or [r["tier"]]
        r.pop("soft_dist", None)          # distribution was over the old label set
        n[r["tier"]] += 1; out.append(r)
    pathlib.Path(dst).write_text("".join(json.dumps(r)+"\n" for r in out))
    return n

for kind in ("train", "eval"):
    print(f"{kind}:")
    for p in sorted(glob.glob(str(ROOT/f"data/{kind}"/f"{src_sig}-*.jsonl"))):
        b = pathlib.Path(p).name
        if any(x in b for x in ("balanced","boundary","greyzone","pool","v2rubric")): continue
        n = convert(p, ROOT/f"data/{kind}"/b.replace(f"{src_sig}-", f"{name}-"))
        print(f"  {b.replace(src_sig+'-',''):<34} {dict(n)}")
