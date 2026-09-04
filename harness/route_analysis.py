"""Accuracy on the decision the ROUTER actually makes.

§74's finding, stated plainly: model accuracy tracked jury agreement within 2-3
points at every fold tested. The binding constraint is not the model, it is
whether the question has a decidable answer. So the useful question becomes:
which questions does llm-d-sc actually ASK?

The deployed Praxis route table does not consume four complexity tiers. It maps
SIMPLE/MEDIUM -> the small model and COMPLEX/REASONING -> the large one. A
classifier that says MEDIUM where the jury said SIMPLE has made NO routing error;
tier-exact accuracy counts it as one.

So this reports, for each signal, the shipped tier-exact number beside the
accuracy of the actual binary routing decision, with jury agreement on that same
decision as the ceiling. Every fold below corresponds to a real branch in the
deployment, not to a convenient regrouping.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# (signal, model, evals, [(decision name, {tier -> branch})])
SPECS = [
 ("complexity", "cx-v-resolved-seed11",
  ["complexity-real-gold.jsonl", "complexity-real-contested.jsonl"],
  [("route: small vs large  (Praxis table)",
    {"SIMPLE":"small","MEDIUM":"small","COMPLEX":"large","REASONING":"large"}),
   ("route: is reasoning needed",
    {"SIMPLE":"no","MEDIUM":"no","COMPLEX":"no","REASONING":"yes"})]),
 ("cost", "co-r-big-seed22",
  ["cost-real-gold.jsonl", "cost-real-contested.jsonl"],
  [("capacity: short vs long generation",
    {"MINIMAL":"short","LOW":"short","MODERATE":"long","HIGH":"long"}),
   ("capacity: reserve a big budget",
    {"MINIMAL":"no","LOW":"no","MODERATE":"no","HIGH":"yes"})]),
 ("sensitivity", "se-w-esc0.0-seed22",
  ["sensitivity-entsec-gold.jsonl", "sensitivity-entsec-contested.jsonl"],
  [("egress: block at NEVER_EGRESS",
    {"PUBLIC":"allow","INTERNAL":"allow","CONFIDENTIAL":"allow",
     "REGULATED":"allow","NEVER_EGRESS":"block"}),
   ("egress: block at REGULATED",
    {"PUBLIC":"allow","INTERNAL":"allow","CONFIDENTIAL":"allow",
     "REGULATED":"block","NEVER_EGRESS":"block"})]),
]

for sig, tag, evs, decisions in SPECS:
    rows = [r for f in evs for r in load_jsonl(ROOT/"data/eval"/f) if r.get("votes")]
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = [m.config.id2label[i] for i in range(m.config.num_labels)]
    pred = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            pred += [order[j] for j in m(**e).logits.argmax(-1).tolist()]
    y = [r["tier"] for r in rows]; votes = [r["votes"] for r in rows]
    print(f"\n### {sig}  ({tag}, n={len(rows)})")
    acc = np.mean([a == b for a, b in zip(y, pred)])
    agr = np.mean([len(set(v)) == 1 for v in votes])
    print(f"  {'decision':<42}{'jury agr':>10}{'accuracy':>10}{'majority':>10}")
    print(f"  {'TIER-EXACT (what has been reported)':<42}{agr:9.1%}{acc:10.2%}{'-':>10}")
    for name, table in decisions:
        yy = [table[a] for a in y]; pp = [table[a] for a in pred]
        a2 = np.mean([x == z for x, z in zip(yy, pp)])
        g2 = np.mean([len({table[v] for v in vs if v in table}) == 1 for vs in votes])
        maj = max(np.mean([x == c for x in yy]) for c in set(yy))
        print(f"  {name:<42}{g2:9.1%}{a2:10.2%}{maj:10.2%}")
