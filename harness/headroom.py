"""Is this signal capped by its LABELS or by the MODEL?

Generalises the §55 diagnostic that gave opposite answers on two signals in the
same corpus. Split the eval by whether the jury was unanimous:

  error concentrated on CONTESTED rows -> the definition is ambiguous; relabel,
      sharpen the rubric, or merge the tiers.
  error present on UNANIMOUS rows      -> the labels are fine and the model is
      the bottleneck; no relabelling campaign can touch it.

The "ceiling" column is what the model would score if every contested row were
labelled perfectly AND it got them all right -- an upper bound on the entire
label-quality axis, computable before spending anything on it.
"""
import sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sig, tag = sys.argv[1], sys.argv[2]
evs = sys.argv[3].split(",")
labels = load_taxonomy(sig)["labels"]
rows = [r for f in evs for r in load_jsonl(ROOT/"data/eval"/f) if r.get("votes")]
if not rows: sys.exit(f"no rows with votes in {evs}")

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

b = collections.defaultdict(lambda: [0, 0])
for r, p in zip(rows, pred):
    k = "unanimous" if len(set(r["votes"])) == 1 else "contested"
    b[k][1] += 1; b[k][0] += (p == r["tier"])
cu, nu = b["unanimous"]; cc, nc = b["contested"]; n = nu + nc
print(f"{sig} / {tag}   n={n}")
print(f"  unanimous  {cu:5d}/{nu:<5d} = {cu/max(nu,1):6.2%}")
print(f"  contested  {cc:5d}/{nc:<5d} = {cc/max(nc,1):6.2%}   ({nc/n:.1%} of the eval)")
print(f"  OVERALL    {cu+cc:5d}/{n:<5d} = {(cu+cc)/n:6.2%}")
print(f"  ceiling from perfect labels alone: {(cu+nc)/n:6.2%}")
print(f"  error on UNANIMOUS rows (untouchable by relabelling): {nu-cu} = {(nu-cu)/n:.2%} of the eval")
conf = collections.Counter((r["tier"], p) for r, p in zip(rows, pred)
                           if len(set(r["votes"])) == 1 and p != r["tier"])
print("  top unanimous-row confusions:")
for (g, p), c in conf.most_common(5):
    print(f"    {g:<12} -> {p:<12} {c}")
