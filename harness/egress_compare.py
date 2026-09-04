"""Binary egress gate vs the folded 5-way model, at MATCHED containment.

The binary model's headline is 95.33% accuracy on entsec-gold against an 82.89%
majority baseline. That clears the baseline by 12.4 points and is a real number.
It is also not the number that matters: BLOCK recall -- what fraction of
live-secret and privileged content is actually stopped -- is 85.12%, and the
5-way model folded onto the same decision contains ~90%.

Higher accuracy with lower containment is the trade this project has hit at every
turn, so §60's rule applies: sweep the decision threshold on BOTH models and read
over-block at equal containment. Only a model that dominates that curve is
actually better.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

rows = load_jsonl(ROOT/"data/eval/egress-entsec-gold.jsonl")
y = np.array([r["tier"] == "BLOCK" for r in rows])
texts = [r["text"] for r in rows]
print(f"entsec-gold n={len(rows)}   BLOCK={y.sum()} ({y.mean():.1%})   majority baseline={max(y.mean(),1-y.mean()):.2%}\n")

def block_score(tag, five_way):
    """P(BLOCK) per row. For the 5-way model that is P(NEVER_EGRESS)."""
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = [m.config.id2label[i] for i in range(m.config.num_labels)]
    j = order.index("NEVER_EGRESS" if five_way else "BLOCK")
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            e = tok(texts[i:i+64], truncation=True, max_length=256, padding=True, return_tensors="pt")
            out.append(torch.softmax(m(**e).logits, -1).numpy()[:, j])
    return np.concatenate(out)

MODELS = [("eg-ad1-cw-seed11", False, "binary egress"),
          ("se-w-esc0.0-seed22", True, "5-way folded"),
          ("se-u-big-seed11", True, "5-way v2 (shipped)")]
S = {n: block_score(t, f) for t, f, n in MODELS}

print(f"{'model':<22}{'argmax acc':>11}{'containment':>12}{'over-block':>11}")
for t, f, n in MODELS:
    p = S[n] >= 0.5
    print(f"{n:<22}{np.mean(p == y):10.2%}{p[y].mean():12.2%}{p[~y].mean():11.2%}")

print(f"\nover-block at matched containment (lower is better):")
print(f"  {'model':<22}" + "".join(f"{f'@{c:.0%}':>12}" for c in (0.85, 0.90, 0.95, 0.99)))
for t, f, n in MODELS:
    cells = ""
    for target in (0.85, 0.90, 0.95, 0.99):
        best = None
        for th in np.unique(np.round(S[n], 4)):
            p = S[n] >= th
            if p[y].mean() >= target:
                o = p[~y].mean()
                if best is None or o < best: best = o
        cells += f"{(f'{best:.2%}' if best is not None else '--'):>12}"
    print(f"  {n:<22}{cells}")
