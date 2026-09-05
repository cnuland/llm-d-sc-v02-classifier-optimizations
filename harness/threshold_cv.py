"""Honestly-selected operating thresholds, by 2-fold cross-validation.

`gate_curves.py` shows accuracy rising above argmax for every gate -- reasoning
reaches 98.14% at threshold 0.8 against 96.54% at 0.5. **Quoting that would be
fitting the test set**: the threshold was chosen because it maximised accuracy on
the same rows being scored.

The fix is the standard one. Split the eval in half, choose the threshold on fold
A, report accuracy on fold B, swap, average. Every reported number comes from rows
whose threshold never saw them. Repeated over several random splits, because a
single split of ~370 rows is itself noisy.

Reports the SELECTED thresholds too. If they scatter across folds, the curve was
noise and argmax should stand; if they cluster, the gain is real and costs one
config value.
"""
import sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GATES = [("route",     "rt-af1-seed11",   "cx2-real-gold-v2.jsonl",       "REASONING","LARGE"),
         ("reasoning", "rs-ae1-seed11",   "reasoning-real-gold-v2.jsonl", "YES",      "YES"),
         ("genlen",    "gl-ae2-seed11",   "genlen-real-gold-v2.jsonl",    "LONG",     "LONG"),
         ("egress",    "eg-ad1-cw-seed11","egress-entsec-gold.jsonl",     "BLOCK",    "BLOCK")]
GRID = np.arange(0.05, 0.96, 0.05)
REPEATS = 40

print(f"{'gate':<11}{'argmax':>9}{'CV-tuned':>10}{'delta':>8}{'thresholds chosen':>34}")
for name, tag, evf, pos_eval, pos_model in GATES:
    p = ROOT/"data/eval"/evf
    if not p.exists(): continue
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
    argmax_acc = np.mean((s >= 0.5) == y)

    held, chosen = [], []
    rng = np.random.default_rng(0)
    for rep in range(REPEATS):
        idx = rng.permutation(len(y))
        for a, b in ((idx[:len(idx)//2], idx[len(idx)//2:]), (idx[len(idx)//2:], idx[:len(idx)//2])):
            best_t = max(GRID, key=lambda t: np.mean((s[a] >= t) == y[a]))
            held.append(np.mean((s[b] >= best_t) == y[b]))
            chosen.append(best_t)
    c = collections.Counter(np.round(chosen, 2))
    top = ", ".join(f"{t}x{n}" for t, n in c.most_common(3))
    print(f"{name:<11}{argmax_acc:8.2%}{np.mean(held):10.2%}{np.mean(held)-argmax_acc:+8.2%}   {top}")
