"""Why does a head trained on cost-train.jsonl LOSE to hand-written anchors?

Two candidate explanations:
  (i)  the training corpus encodes a DIFFERENT notion of 'cost' than the
       hand-authored held-out set -> label drift, a data problem;
  (ii) the head overfits 1416 rows -> a modelling problem.
Distinguish them: if the head is confidently wrong in a systematic direction
(a whole tier shifted), that is drift, not overfitting.
"""
import sys; sys.path.insert(0,"/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *
from sklearn.linear_model import LogisticRegression
import numpy as np

sig="cost"; tax=load_taxonomy(sig); labels=tax["labels"]
ho=heldout(sig); tr=load_jsonl(GENESIS/"training/data/cost-train.jsonl")
mid="cnuland/llm-d-sc-cost"
Xtr=unit(embed(mid,[r["text"] for r in tr]).astype(np.float64))
Xho=unit(embed(mid,[r["text"] for r in ho]).astype(np.float64))
y_tr=[r["tier"] for r in tr]; y_ho=[r["tier"] for r in ho]
clf=LogisticRegression(max_iter=4000,C=10.0,class_weight="balanced").fit(Xtr,y_tr)
pred=list(clf.predict(Xho))
m=metrics(y_ho,pred,labels)
print("confusion (rows=truth, cols=head prediction)")
print(f"{'':>10}"+"".join(f"{l:>10}" for l in labels))
for t in labels: print(f"{t:>10}"+"".join(f"{m['confusion'][t][p]:>10}" for p in labels))

# train-set self-consistency: does the corpus even separate cleanly?
from sklearn.model_selection import cross_val_score
cv=cross_val_score(LogisticRegression(max_iter=4000,C=10.0,class_weight="balanced"),Xtr,y_tr,cv=5)
print(f"\n5-fold CV accuracy WITHIN cost-train.jsonl: {cv.mean():.4f} (+/-{cv.std():.4f})")
print("  -> if this is high but held-out is low, the two label systems disagree.\n")

print("held-out examples the head gets wrong (truth -> predicted):")
for i,(t,p) in enumerate(zip(y_ho,pred)):
    if t!=p: print(f"  [{t:>8} -> {p:<8}] {ho[i]['text'][:96]}")
