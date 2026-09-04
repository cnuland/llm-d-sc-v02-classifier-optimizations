"""What accuracy would a MERGED taxonomy actually deliver?

§69 showed high-90s tier-exact accuracy is unreachable against labels with 73-88%
inter-juror agreement. The honest alternative is not to keep grinding — it is to
ask whether the taxonomy is drawing a distinction the routing decision does not
use. That is a product question, but it should be answered with numbers, so this
reports for each candidate merge:

  agreement   what the three jurors' agreement becomes once the merged tiers are
              treated as one label -- the CEILING the merge creates
  accuracy    what the existing model scores under the merged labels, with no
              retraining -- a LOWER bound, since a model trained on the merged
              taxonomy would do better

A merge is only worth proposing where the boundary is both expensive (it eats
juror agreement) and cheap to lose (routing treats both tiers the same).
"""
import sys, json, itertools, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SPECS = {
 "complexity": ("cx-v-resolved-seed11",
                ["complexity-real-gold.jsonl", "complexity-real-contested.jsonl"],
                [("MEDIUM","COMPLEX"), ("SIMPLE","MEDIUM"), ("COMPLEX","REASONING")]),
 "cost":       ("co-r-big-seed22",
                ["cost-real-gold.jsonl", "cost-real-contested.jsonl"],
                [("LOW","MODERATE"), ("MINIMAL","LOW"), ("MODERATE","HIGH")]),
 "sensitivity":("se-w-esc0.0-seed22",
                ["sensitivity-entsec-gold.jsonl", "sensitivity-entsec-contested.jsonl"],
                [("INTERNAL","CONFIDENTIAL"), ("PUBLIC","INTERNAL"),
                 ("CONFIDENTIAL","REGULATED")]),
}

for sig, (tag, evs, merges) in SPECS.items():
    labels = load_taxonomy(sig)["labels"]
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
    y = [r["tier"] for r in rows]
    votes = [r["votes"] for r in rows]

    def score(fold):
        yy = [fold(a) for a in y]; pp = [fold(a) for a in pred]
        acc = np.mean([a == b for a, b in zip(yy, pp)])
        # unanimity under the folded label set
        agr = np.mean([len({fold(v) for v in vs}) == 1 for vs in votes])
        return acc, agr

    ident = lambda a: a
    a0, g0 = score(ident)
    print(f"\n### {sig}  ({tag}, n={len(rows)})")
    print(f"  {'taxonomy':<34}{'jury agreement':>16}{'model accuracy':>16}")
    print(f"  {'as shipped ('+str(len(labels))+' tiers)':<34}{g0:15.1%}{a0:16.2%}")
    for a, b in merges:
        fold = lambda t, a=a, b=b: (a+"+"+b) if t in (a, b) else t
        acc, agr = score(fold)
        print(f"  {'merge '+a+'+'+b:<34}{agr:15.1%}{acc:16.2%}")
    if sig == "sensitivity":
        rank = {l: i for i, l in enumerate(labels)}
        for gate in ("CONFIDENTIAL", "REGULATED", "NEVER_EGRESS"):
            fold = lambda t, g=gate: "BLOCK" if rank[t] >= rank[g] else "ALLOW"
            acc, agr = score(fold)
            print(f"  {'BINARY gate at '+gate:<34}{agr:15.1%}{acc:16.2%}")
