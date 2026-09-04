"""§57's frozen-pair probe, generalised to any signal and pair.

The question it answers is deliberately narrow: with the pair isolated, the
classes balanced and no fine-tuning, how much of the distinction does each
encoder's representation carry? That is an upper bound on what any 5-way
fine-tune of the same encoder can recover, and it is cheap enough to run on
every confusable pair rather than only the one that hurt most.
"""
import sys, json, collections, glob
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np
from evalkit import ROOT, load_jsonl, embed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score

sig, pair = sys.argv[1], sys.argv[2]
A, B = pair.split("/")
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
# §80: bge-base is the better classifier (+4.5) but MiniLM the better component
# (6.1 ms vs 21.9 ms). The interesting cell is a SMALL strong encoder -- does
# bge-class quality survive at MiniLM-class cost?
ENCODERS = ["sentence-transformers/all-MiniLM-L6-v2",   #  22M, 384d  -- shipped
            "BAAI/bge-small-en-v1.5",                   #  33M, 384d
            "intfloat/e5-small-v2",                     #  33M, 384d
            "thenlper/gte-small",                       #  33M, 384d
            "BAAI/bge-base-en-v1.5"]                    # 109M, 768d -- reference

tr = collections.defaultdict(list)
for p in glob.glob(str(ROOT/"data/train"/f"{sig}-*.jsonl")):
    if "balanced" in p: continue
    for l in open(p):
        r = json.loads(l); t = r.get("tier") or r.get("label")
        if t in (A, B): tr[t].append(r["text"])
rng = np.random.default_rng(0)
Xtxt, ytr = [], []
for t in (A, B):
    v = list(dict.fromkeys(tr[t]))
    if len(v) > CAP: v = [v[i] for i in rng.choice(len(v), CAP, replace=False)]
    Xtxt += v; ytr += [t]*len(v)

seen, ev = set(), []
for p in glob.glob(str(ROOT/"data/eval"/f"{sig}-*gold*.jsonl")) + \
         glob.glob(str(ROOT/"data/eval"/f"{sig}-*contested*.jsonl")):
    for r in load_jsonl(p):
        if r["tier"] in (A, B) and r["text"] not in seen:
            seen.add(r["text"]); ev.append(r)
yev = [r["tier"] for r in ev]
c = collections.Counter(yev)
if len(c) < 2 or min(c.values()) < 10 or len(set(ytr)) < 2:
    sys.exit(f"{sig} {pair}: too few eval rows {dict(c)}")
maj = max(c.values())/len(yev)

print(f"\n### {sig}  {A} vs {B}   train {collections.Counter(ytr)}   eval {dict(c)}")
print(f"{'encoder':<26}{'CV(train)':>11}{'eval':>9}{'eval-bal':>10}{'minor rec':>11}")
print(f"{'(majority predictor)':<26}{'':>11}{maj:9.2%}{0.5:10.2%}{0.0:11.2%}")
minor = min(c, key=c.get)
for e in ENCODERS:
    Xtr = embed(e, Xtxt, normalize=True); Xev = embed(e, [r["text"] for r in ev], normalize=True)
    cv = cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"), Xtr, ytr, cv=5).mean()
    pr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr).predict(Xev)
    acc = np.mean(pr == np.array(yev)); bal = balanced_accuracy_score(yev, pr)
    rec = np.mean([p == g for p, g in zip(pr, yev) if g == minor])
    print(f"{e.split('/')[-1]:<26}{cv:10.2%}{acc:9.2%}{bal:10.2%}{rec:11.2%}")
