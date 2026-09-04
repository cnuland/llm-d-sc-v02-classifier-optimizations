"""Does a STRONGER frozen encoder carry the INTERNAL/CONFIDENTIAL distinction?

pair_probe.py showed MiniLM-L6 caps at 82.9% within-train 5-fold CV on this pair
with balanced classes and zero interference from other tiers. That is the
easiest setting the problem can be posed in, so 82.9% is a representation bound,
not a training artifact.

§50 concluded "bigger is worse" for this signal, but that was measured on
FINE-TUNED models over a skewed 5-way corpus, where the extra capacity went to
the majority class. A frozen probe removes both confounds: same data, same
classifier, same balance -- only the representation changes. If a stronger
encoder lifts the bound, the earlier finding was about the fine-tuning regime,
not about what the encoder can represent.
"""
import sys, json, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np
from evalkit import ROOT, load_jsonl, embed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score

A, B = (sys.argv[1] if len(sys.argv) > 1 else "INTERNAL/CONFIDENTIAL").split("/")
CAP = 3000
ENCODERS = ["sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
            "intfloat/e5-base-v2"]
FILES = ["sensitivity-v2.jsonl", "sensitivity-real.jsonl",
         "sensitivity-enterprise.jsonl", "sensitivity-real-contested.jsonl"]

tr = collections.defaultdict(list)
for f in FILES:
    p = ROOT/"data/train"/f
    if not p.exists(): continue
    for l in open(p):
        r = json.loads(l); t = r.get("tier") or r.get("label")
        if t in (A, B): tr[t].append(r["text"])
rng = np.random.default_rng(0)
Xtxt, ytr = [], []
for t in (A, B):
    v = tr[t]
    if len(v) > CAP: v = [v[i] for i in rng.choice(len(v), CAP, replace=False)]
    Xtxt += v; ytr += [t]*len(v)

evs, seen, ev = [], set(), []
for f in (f"sensitivity-entsec-gold.jsonl", "sensitivity-entsec-contested.jsonl"):
    evs += load_jsonl(ROOT/"data/eval"/f)
for r in evs:
    if r["tier"] in (A, B) and r["text"] not in seen:
        seen.add(r["text"]); ev.append(r)
yev = [r["tier"] for r in ev]
maj = max(collections.Counter(yev).values())/len(yev)

print(f"{A} vs {B}   train {len(ytr)} balanced   eval n={len(ev)} (majority {maj:.1%})\n")
print(f"{'encoder':<42}{'CV(train)':>11}{'eval':>9}{'eval-bal':>11}{B+' rec':>12}")
print(f"{'(majority-class predictor)':<42}{'':>11}{maj:9.2%}{0.5:11.2%}{0.0:12.2%}")
for e in ENCODERS:
    try:
        Xtr = embed(e, Xtxt, normalize=True)
        Xev = embed(e, [r["text"] for r in ev], normalize=True)
    except Exception as ex:
        print(f"{e.split('/')[-1]:<42}   FAILED {type(ex).__name__}"); continue
    cv = cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"),
                         Xtr, ytr, cv=5).mean()
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    pr = clf.predict(Xev)
    acc = np.mean(pr == np.array(yev))
    bal = balanced_accuracy_score(yev, pr)
    rec = np.mean([p==g for p,g in zip(pr,yev) if g==B])
    print(f"{e.split(chr(47))[-1]:<42}{cv:10.2%}{acc:9.2%}{bal:11.2%}{rec:12.2%}")
