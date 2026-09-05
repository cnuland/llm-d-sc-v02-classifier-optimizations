"""Operating curves for the four published gates.

Every gate is currently reported at argmax, i.e. a 0.5 threshold. That is one
arbitrary point on a curve, and it is almost never the right one, because the two
error directions cost different amounts:

  route     SIMPLE sent to the large model wastes money; REASONING sent to the
            small model produces a worse answer. Asymmetric, and which way
            depends on the price gap.
  genlen    a SHORT request given a LONG budget wastes reserved capacity; the
            reverse causes truncation or a re-run.
  reasoning firing spuriously costs an expensive call; missing costs quality.
  egress    a false BLOCK stops legitimate work; a false ALLOW leaks a secret.
            Wildly asymmetric.

This prints the curve rather than a tuned number. **Picking the threshold that
maximises accuracy on the eval and reporting THAT accuracy would be fitting the
test set** -- the argmax row stays the headline, and the rest is for choosing an
operating point against a cost model this project does not have.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GATES = [("route",     "rt-af1-seed11", "cx2-real-gold-v2.jsonl",      "REASONING", "LARGE"),
         ("reasoning", "rs-ae1-seed11", "reasoning-real-gold-v2.jsonl","YES",       "YES"),
         ("genlen",    "gl-ae2-seed11", "genlen-real-gold-v2.jsonl",   "LONG",      "LONG"),
         ("egress",    "eg-ad1-cw-seed11","egress-entsec-gold.jsonl",  "BLOCK",     "BLOCK")]

for name, tag, evf, pos_eval, pos_model in GATES:
    p = ROOT/"data/eval"/evf
    if not p.exists(): print(f"{name}: eval missing"); continue
    rows = load_jsonl(p); y = np.array([r["tier"] == pos_eval for r in rows])
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = [m.config.id2label[i] for i in range(m.config.num_labels)]
    j = order.index(pos_model)
    s = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            s.append(torch.softmax(m(**e).logits, -1).numpy()[:, j])
    s = np.concatenate(s)
    base = max(y.mean(), 1-y.mean())
    print(f"\n### {name}  ({tag}, n={len(rows)}, positives={y.sum()} = {y.mean():.1%}, "
          f"majority baseline {base:.2%})")
    print(f"  {'thresh':>7}{'accuracy':>10}{'pos recall':>12}{'pos prec':>10}{'false-fire':>12}")
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        p_ = s >= th
        acc = np.mean(p_ == y)
        rec = p_[y].mean() if y.any() else float("nan")
        prec = y[p_].mean() if p_.any() else float("nan")
        ff = p_[~y].mean()
        star = "  <- argmax, the reported number" if th == 0.5 else ""
        print(f"  {th:7.1f}{acc:10.2%}{rec:12.2%}{prec:10.2%}{ff:12.2%}{star}")
