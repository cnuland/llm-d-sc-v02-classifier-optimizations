"""Is a confusable pair LEARNABLE from the training corpus, or absent from it?

Post-hoc logit adjustment bought only +0.95 accuracy for -1.95 containment, so
sensitivity's INTERNAL/CONFIDENTIAL error is not a decision-threshold artifact --
moving the boundary just relocates errors. That leaves two possibilities:

  (a) the distinction IS in the data, and the 5-way model loses it to
      interference from the other three classes  -> a hierarchical/2-stage
      model or pair-specific capacity would recover it
  (b) the distinction is NOT reliably in the training data  -> no architecture
      fixes it, and the fix is data

A frozen-encoder linear probe on JUST that pair separates the two: it has no
interference from other classes and no fine-tuning, so its accuracy is a
lower bound on how much of the distinction the corpus actually carries.
Cheap enough to run while other jobs hold the CPU.
"""
import sys, json, collections, itertools
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np
from evalkit import ROOT, load_jsonl, embed
from sklearn.linear_model import LogisticRegression

sig = sys.argv[1] if len(sys.argv) > 1 else "sensitivity"
A, B = (sys.argv[2] if len(sys.argv) > 2 else "INTERNAL/CONFIDENTIAL").split("/")
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
ENC = "sentence-transformers/all-MiniLM-L6-v2"

TRAIN = {"sensitivity": ["sensitivity-v2.jsonl", "sensitivity-real.jsonl",
                         "sensitivity-enterprise.jsonl", "sensitivity-real-contested.jsonl"],
         "complexity":  ["complexity-v2.jsonl", "complexity-real.jsonl"],
         "cost":        ["cost-v2.jsonl", "cost-real.jsonl"]}[sig]

tr = collections.defaultdict(list)
for f in TRAIN:
    p = ROOT / "data/train" / f
    if not p.exists(): continue
    for l in open(p):
        r = json.loads(l)
        t = r.get("tier") or r.get("label")
        if t in (A, B): tr[t].append(r["text"])

rng = np.random.default_rng(0)
Xtxt, ytr = [], []
for t in (A, B):
    v = tr[t]
    if len(v) > CAP: v = [v[i] for i in rng.choice(len(v), CAP, replace=False)]
    Xtxt += v; ytr += [t]*len(v)

evs = []
for f in (f"{sig}-entsec-gold.jsonl", f"{sig}-entsec-contested.jsonl",
          f"{sig}-real-gold.jsonl", f"{sig}-real-gold-v2.jsonl"):
    p = ROOT/"data/eval"/f
    if p.exists(): evs += load_jsonl(p)
seen, ev = set(), []
for r in evs:
    if r["tier"] in (A, B) and r["text"] not in seen:
        seen.add(r["text"]); ev.append(r)

print(f"{sig}  {A} vs {B}")
print(f"  train {collections.Counter(ytr)}   eval n={len(ev)} {collections.Counter(r['tier'] for r in ev)}")
if len(ev) < 20 or len(set(ytr)) < 2:
    sys.exit("  not enough data")

Xtr = embed(ENC, Xtxt, normalize=True)
Xev = embed(ENC, [r["text"] for r in ev], normalize=True)
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
pred = clf.predict(Xev)
yev = [r["tier"] for r in ev]
acc = np.mean(pred == np.array(yev))
maj = max(collections.Counter(yev).values())/len(yev)
unan = [len(set(r.get("votes") or [r["tier"]])) == 1 for r in ev]
au = np.mean([p == g for p, g, u in zip(pred, yev, unan) if u]) if any(unan) else float("nan")
print(f"  frozen-encoder linear probe: {acc:.2%}   (majority baseline {maj:.2%}, unanimous-only {au:.2%})")

# --- is the gap DOMAIN SHIFT or intrinsic difficulty? -----------------------
# If the same probe separates the pair easily WITHIN the training corpus but
# fails on eval, the corpus carries a different distinction than the eval does
# -- a shortcut. If it fails in both, the pair is genuinely hard.
from sklearn.model_selection import cross_val_score
cv = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                     Xtr, ytr, cv=5, scoring="accuracy")
print(f"  within-TRAIN 5-fold CV      : {cv.mean():.2%} (+/- {cv.std():.2%})")
print(f"  -> gap train->eval          : {cv.mean()-acc:+.1%}")
