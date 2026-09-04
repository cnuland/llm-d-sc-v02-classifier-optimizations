"""Is sensitivity capped by the RUBRIC or by the MODEL?

Decisive split: accuracy on rows where all 3 jurors AGREED vs rows where they
split. If the model is already near-perfect on unanimous rows, the remaining
error is concentrated where the definition itself is ambiguous -> rewrite the
rubric. If it errs on unanimous rows too, the definition is fine and the model
is the bottleneck -> a rubric rewrite cannot help.
"""
import sys, json, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tag = sys.argv[1] if len(sys.argv) > 1 else "se-u-big-seed11"
labels = load_taxonomy("sensitivity")["labels"]
rows = (load_jsonl(ROOT/"data/eval/sensitivity-entsec-gold.jsonl")
        + load_jsonl(ROOT/"data/eval/sensitivity-entsec-contested.jsonl"))

d = ROOT/"models"/tag
tok = AutoTokenizer.from_pretrained(str(d))
m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
order = [m.config.id2label[i] for i in range(m.config.num_labels)]

preds = []
with torch.no_grad():
    for i in range(0, len(rows), 64):
        b = [r["text"] for r in rows[i:i+64]]
        enc = tok(b, truncation=True, max_length=256, padding=True, return_tensors="pt")
        preds += [order[j] for j in m(**enc).logits.argmax(-1).tolist()]

buckets = collections.defaultdict(lambda: [0, 0])
for r, p in zip(rows, preds):
    k = "unanimous" if len(set(r["votes"])) == 1 else "contested"
    buckets[k][1] += 1
    buckets[k][0] += (p == r["tier"])

print(f"model {tag}   n={len(rows)}\n")
tot_c = tot_n = 0
for k in ("unanimous", "contested"):
    c, n = buckets[k]
    tot_c += c; tot_n += n
    print(f"  {k:<12} {c:4d}/{n:<4d} = {c/n:6.2%}")
print(f"  {'OVERALL':<12} {tot_c:4d}/{tot_n:<4d} = {tot_c/tot_n:6.2%}")

cu, nu = buckets["unanimous"]
print(f"\n  ceiling if contested rows were perfectly labelled: "
      f"{(cu + buckets['contested'][1]) / tot_n:.2%}")
print(f"  error on unanimous rows (model's own fault): {nu-cu} rows = {(nu-cu)/tot_n:.2%} of the set")

# where does the unanimous error go?
conf = collections.Counter()
for r, p in zip(rows, preds):
    if len(set(r["votes"])) == 1 and p != r["tier"]:
        conf[(r["tier"], p)] += 1
print("\n  unanimous-row confusions (gold -> pred):")
for (g, p), c in conf.most_common(8):
    print(f"    {g:<14} -> {p:<14} {c}")
