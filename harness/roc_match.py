"""Compare two models at MATCHED containment, per gate.

§60's rule. Raw containment/over-block pairs are not comparable between models
that sit at different operating points: a model can look safer purely by being
more trigger-happy. Sweep a bias on the at-or-above-gate logits of BOTH models,
then read each one's over-block at the other's containment. A model that is
genuinely better dominates -- lower over-block at every containment it can reach.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
LAB = load_taxonomy("sensitivity")["labels"]; rank = {l: i for i, l in enumerate(LAB)}
rows = load_jsonl(ROOT/"data/eval/sensitivity-entsec-gold.jsonl")
y = [r["tier"] for r in rows]

def logits_of(tag):
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = [m.config.id2label[i] for i in range(m.config.num_labels)]
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            out.append(m(**e).logits.numpy())
    return np.concatenate(out), order

def curve(L, order, gate):
    thr = rank[gate]
    up = np.array([1.0 if rank[l] >= thr else 0.0 for l in order])
    hi = np.array([rank[a] >= thr for a in y])
    pts = []
    for b in np.arange(-4, 6.01, 0.05):
        P = (L + b*up).argmax(1)
        blk = np.array([rank[order[i]] >= thr for i in P])
        pts.append((blk[hi].mean(), blk[~hi].mean()))
    return pts

A, B = sys.argv[1], sys.argv[2]
LA, oA = logits_of(A); LB, oB = logits_of(B)
print(f"over-block at matched containment (lower is better)\n")
print(f"{'gate':<15}{'containment':>12}{A[:18]:>20}{B[:18]:>20}{'winner':>10}")
for g in ("CONFIDENTIAL", "REGULATED", "NEVER_EGRESS"):
    ca, cb = curve(LA, oA, g), curve(LB, oB, g)
    for target in (0.85, 0.90, 0.95):
        fa = [o for c, o in ca if c >= target]; fb = [o for c, o in cb if c >= target]
        oa = min(fa) if fa else None; ob = min(fb) if fb else None
        if oa is None and ob is None: continue
        sa = f"{oa:.2%}" if oa is not None else "unreachable"
        sb = f"{ob:.2%}" if ob is not None else "unreachable"
        w = A[:12] if (ob is None or (oa is not None and oa < ob)) else B[:12]
        print(f"{g:<15}{target:11.0%}{sa:>20}{sb:>20}{w:>10}")
