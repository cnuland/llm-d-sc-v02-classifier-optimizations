"""Does a containment threshold tuned on one sample HOLD on another?

Every gate figure in this report picks a threshold and reports the containment and
over-block it achieves on the same rows. A deployer cannot do that: they tune on
whatever data they have and then run on traffic they have not seen. The question
that actually matters is whether "set the threshold for 95% containment" delivers
95% containment on new rows -- or systematically overshoots, because the threshold
was fitted to the noise in the tuning sample.

Protocol: split the eval, choose the smallest bias reaching the target containment
on fold A, then MEASURE the containment and over-block that threshold actually
achieves on fold B. Swapped, over many random splits.

The gap between target and achieved is the number nobody has reported and the one
a security owner needs: it says how much containment margin to ask for in order to
get the containment you want.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TAG = sys.argv[1] if len(sys.argv) > 1 else "eg-ad1-cw-seed11"
REPEATS = 60
rows = load_jsonl(ROOT/"data/eval/egress-entsec-gold.jsonl")
y = np.array([r["tier"] == "BLOCK" for r in rows]); txt = [r["text"] for r in rows]
d = ROOT/"models"/TAG
tok = AutoTokenizer.from_pretrained(str(d))
m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
order = [m.config.id2label[i] for i in range(m.config.num_labels)]; j = order.index("BLOCK")
s = []
with torch.no_grad():
    for i in range(0, len(txt), 64):
        e = tok(txt[i:i+64], truncation=True, max_length=256, padding=True, return_tensors="pt")
        s.append(torch.softmax(m(**e).logits, -1).numpy()[:, j])
s = np.concatenate(s)

print(f"{TAG}  n={len(rows)}  BLOCK={y.sum()}\n")
print(f"  {'target':>8}{'achieved':>11}{'shortfall':>11}{'over-block':>12}{'P(miss target)':>16}")
rng = np.random.default_rng(0)
for target in (0.85, 0.90, 0.95):
    ach, ob, miss = [], [], 0
    for _ in range(REPEATS):
        idx = rng.permutation(len(y))
        for a, b in ((idx[:len(idx)//2], idx[len(idx)//2:]), (idx[len(idx)//2:], idx[:len(idx)//2])):
            cand = [t for t in np.unique(np.round(s[a], 3)) if (s[a] >= t)[y[a]].mean() >= target]
            if not cand: continue
            t = max(cand)                       # cheapest threshold that hits target on A
            p = s[b] >= t
            c = p[y[b]].mean()
            ach.append(c); ob.append(p[~y[b]].mean()); miss += (c < target)
    n = len(ach)
    print(f"  {target:8.0%}{np.mean(ach):11.2%}{np.mean(ach)-target:+11.2%}"
          f"{np.mean(ob):12.2%}{miss/max(1,n):15.0%}")
