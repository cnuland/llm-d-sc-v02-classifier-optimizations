"""Ensemble two specific models and score BOTH accuracy and the gates.

§67 found bge and MiniLM own different gates -- bge the NEVER_EGRESS gate, MiniLM
the REGULATED gate. Complementary errors are the precondition for averaging to
pay, so this is the case where an ensemble is worth its second forward pass
rather than a reflex.

Scored at matched containment against each member, because §60 established that
raw containment numbers between different operating points are not comparable.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
LAB = load_taxonomy("sensitivity")["labels"]; rank = {l: i for i, l in enumerate(LAB)}
rows = load_jsonl(ROOT/"data/eval/sensitivity-entsec-gold.jsonl")
y = [r["tier"] for r in rows]

def probs_of(tag):
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = [m.config.id2label[i] for i in range(m.config.num_labels)]
    idx = [order.index(l) for l in LAB]        # re-order to canonical tier order
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            out.append(torch.softmax(m(**e).logits, -1).numpy()[:, idx])
    return np.concatenate(out)

tags = sys.argv[1:]
P = {t: probs_of(t) for t in tags}
# do the members actually err differently? averaging cannot fix shared errors
errs = {t: set(np.where(np.array([LAB[i] for i in P[t].argmax(1)]) != np.array(y))[0]) for t in tags}
for a in range(len(tags)):
    for b in range(a+1, len(tags)):
        A, B = errs[tags[a]], errs[tags[b]]
        print(f"error overlap {tags[a]} ~ {tags[b]}: Jaccard {len(A&B)/len(A|B):.3f} "
              f"({len(A)} / {len(B)} errors, {len(A&B)} shared)")

def report(name, prob):
    pred = [LAB[i] for i in prob.argmax(1)]
    line = f"{name:<34}{np.mean([a==b for a,b in zip(pred,y)]):8.2%}"
    for g in ("CONFIDENTIAL", "REGULATED", "NEVER_EGRESS"):
        thr = rank[g]
        hi = [rank[a]>=thr for a in y]; blk = [rank[a]>=thr for a in pred]
        line += f"{np.mean([b for b,h in zip(blk,hi) if h]):11.2%}{np.mean([b for b,h in zip(blk,hi) if not h]):9.2%}"
    print(line)

print(f"\n{'model':<34}{'acc':>8}" + "".join(f"{g[:4]+' cont':>11}{g[:4]+' OB':>9}" for g in ("CONFIDENTIAL","REGULATED","NEVER_EGRESS")))
for t in tags: report(t, P[t])
report("MEAN ensemble", np.mean([P[t] for t in tags], axis=0))
# tier-max: take the higher tier when members disagree -- the safety-biased merge
mx = np.zeros_like(P[tags[0]])
argmaxes = [P[t].argmax(1) for t in tags]
for i in range(len(rows)):
    j = max(a[i] for a in argmaxes)
    mx[i, j] = 1.0
report("TIER-MAX merge", mx)

# --- does the ensemble beat its best member at MATCHED containment? ----------
# Without this the table above is unreadable: the ensemble sits at a different
# operating point from every member, so "higher containment" may be nothing but
# a more trigger-happy threshold (§60).
print("\nover-block at matched containment (lower is better):")
ENS = np.mean([P[t] for t in tags], axis=0)
cands = {**{t: P[t] for t in tags}, "MEAN ensemble": ENS}
for g in ("CONFIDENTIAL", "REGULATED", "NEVER_EGRESS"):
    thr = rank[g]
    up = np.array([1.0 if rank[l] >= thr else 0.0 for l in LAB])
    hi = np.array([rank[a] >= thr for a in y])
    print(f"\n  {g}")
    print(f"    {'model':<34}" + "".join(f"{f'@{t:.0%}':>12}" for t in (0.85, 0.90, 0.95)))
    for name, pr in cands.items():
        L = np.log(np.clip(pr, 1e-9, 1))
        cells = ""
        for target in (0.85, 0.90, 0.95):
            best = None
            for b in np.arange(-4, 8.01, 0.05):
                idx = (L + b*up).argmax(1)
                blk = np.array([rank[LAB[i]] >= thr for i in idx])
                c, o = blk[hi].mean(), blk[~hi].mean()
                if c >= target and (best is None or o < best): best = o
            cells += f"{(f'{best:.2%}' if best is not None else '--'):>12}"
        print(f"    {name:<34}{cells}")
