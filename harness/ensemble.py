"""Do the already-trained models disagree usefully enough to ensemble?

Free to try: no training, only inference. The interesting question is not
whether the ensemble scores higher -- averaging usually helps a little -- but
whether these particular models are DIVERSE. Models trained on the same corpus
with the same objective tend to make the same mistakes, in which case averaging
buys nothing and the apparent gain is noise at n=418.

So report pairwise error overlap alongside accuracy. If the members err on the
same rows, the ensemble cannot fix them and is not worth 3x the inference cost.
"""
import sys, itertools, collections, torch, numpy as np
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy, metrics, wilson
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sig = "complexity"
labels = load_taxonomy(sig)["labels"]
rows = load_jsonl(ROOT / f"data/eval/{sig}-real-gold.jsonl")
y = [r["tier"] for r in rows]
MEMBERS = ["cx-b4-soft-512-mbert", "cx-b2-soft-256", "cx-d2-everything", "cx-b1-hard-256"]

probs, accs = {}, {}
for tag in MEMBERS:
    d = ROOT / "models" / tag
    if not d.exists():
        print(f"  {tag}: missing, skipped"); continue
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            b = tok([r["text"] for r in rows[i:i+32]], truncation=True, max_length=512,
                    padding=True, return_tensors="pt")
            P.append(m(**b).logits.softmax(-1).numpy())
    probs[tag] = np.vstack(P)
    pred = [labels[j] for j in probs[tag].argmax(1)]
    accs[tag] = metrics(y, pred, labels)["accuracy"]
    print(f"  {tag:<26} acc={accs[tag]:.4f}")

names = list(probs)
print("\n  pairwise ERROR OVERLAP (share of one member's errors the other also makes):")
errs = {t: {i for i in range(len(rows)) if labels[probs[t][i].argmax()] != y[i]} for t in names}
for a, b in itertools.combinations(names, 2):
    inter = len(errs[a] & errs[b]); union = len(errs[a] | errs[b])
    print(f"    {a[:22]:<24} vs {b[:22]:<24} jaccard={inter/max(1,union):.3f}  "
          f"({inter} shared of {len(errs[a])}/{len(errs[b])})")

print("\n  ensembles (mean of softmax):")
for k in range(2, len(names) + 1):
    for combo in itertools.combinations(names, k):
        avg = np.mean([probs[t] for t in combo], axis=0)
        pred = [labels[j] for j in avg.argmax(1)]
        mm = metrics(y, pred, labels)
        best = max(accs[t] for t in combo)
        lo, hi = mm["wilson95"]
        flag = "" if mm["accuracy"] <= best else "  <-- beats every member"
        print(f"    {'+'.join(t[:14] for t in combo):<62} acc={mm['accuracy']:.4f} "
              f"[{lo:.3f},{hi:.3f}] best-member={best:.4f}{flag}")
allerr = set.intersection(*errs.values())
print(f"\n  rows EVERY member gets wrong: {len(allerr)}  "
      f"({len(allerr)/len(rows):.1%} of the eval — an ensemble cannot fix these)")
