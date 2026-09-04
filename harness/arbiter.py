"""Combine the 4-way model with binary specialists on the contested boundaries.

45 of complexity's 48 real-traffic errors fall on two pairs: SIMPLE/MEDIUM (26)
and MEDIUM/COMPLEX (19). A single 4-way softmax carves all three boundaries with
one shared representation; a binary classifier trained on one pair spends its
whole capacity there.

Intervention rule: consult a specialist only when the 4-way model's TOP TWO
classes are exactly that pair. That is the situation the specialist was trained
for, and it leaves every confident 4-way decision untouched -- so the arbiter can
only affect rows where the base model is genuinely torn between two adjacent
tiers.

Reported against the base model on the same rows, plus how often the arbiter
actually fired and how often it changed the answer. An arbiter that rarely fires
cannot help much however good it is, and that is worth knowing separately from
whether it is right when it does.
"""
import sys, collections, torch, numpy as np
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy, metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sig = "complexity"
base_tag = sys.argv[1] if len(sys.argv) > 1 else "cx-l1-distill"
evalname = sys.argv[2] if len(sys.argv) > 2 else "complexity-real-gold-v2"
labels = load_taxonomy(sig)["labels"]
rows = load_jsonl(ROOT / f"data/eval/{evalname}.jsonl")
y = [r["tier"] for r in rows]

def probs(tag, maxlen=None):
    d = ROOT / "models" / tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    ml = maxlen or (512 if ("512" in tag or "mbert" in tag) else 256)
    P = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            b = tok([r["text"] for r in rows[i:i+32]], truncation=True, max_length=ml,
                    padding=True, return_tensors="pt")
            P.append(m(**b).logits.softmax(-1).numpy())
    return np.vstack(P)

base = probs(base_tag)
spec = {}
for a, b in [("SIMPLE", "MEDIUM"), ("MEDIUM", "COMPLEX")]:
    tag = f"cx-m{1 if a=='SIMPLE' else 2}-pair-{a}-{b}"
    if (ROOT / "models" / tag / "config.json").exists():
        spec[(a, b)] = probs(tag)
        print(f"  loaded specialist {tag}")
if not spec:
    sys.exit("  no specialists trained yet")

li = {l: i for i, l in enumerate(labels)}
base_pred = [labels[i] for i in base.argmax(1)]
arb_pred, fired, changed = [], 0, 0
for i in range(len(rows)):
    order = base[i].argsort()[::-1]
    top2 = frozenset({labels[order[0]], labels[order[1]]})
    p = base_pred[i]
    for (a, b), P in spec.items():
        if top2 == frozenset({a, b}):
            fired += 1
            cand = a if P[i][li[a]] >= P[i][li[b]] else b
            if cand != p:
                changed += 1
            p = cand
            break
    arb_pred.append(p)

mb = metrics(y, base_pred, labels)
ma = metrics(y, arb_pred, labels)
print(f"\n  base      {base_tag:<22} acc={mb['accuracy']:.4f}  macroF1={mb['macro_f1']:.4f}")
print(f"  arbitered {'':<22} acc={ma['accuracy']:.4f}  macroF1={ma['macro_f1']:.4f}"
      f"   delta={ma['accuracy']-mb['accuracy']:+.4f}")
print(f"  arbiter fired on {fired}/{len(rows)} rows ({fired/len(rows):.1%}), "
      f"changed the answer on {changed}")
if changed:
    right = sum(1 for i in range(len(rows))
                if arb_pred[i] != base_pred[i] and arb_pred[i] == y[i])
    print(f"  of the {changed} changes, {right} were corrections and "
          f"{changed-right} were regressions")
