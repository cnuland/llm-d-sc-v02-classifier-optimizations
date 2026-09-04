"""Frozen-encoder 5-way probe: how much of the task is in the representation?

Round X is fine-tuning bge on this signal and will take hours. A frozen probe
answers the cheaper version of the same question in minutes -- and it is a
FLOOR, not an estimate: fine-tuning adapts the encoder, so whatever a linear
layer over frozen features recovers, a fine-tune of the same backbone should
match or beat. If frozen bge already approaches the fine-tuned MiniLM, the
encoder swap is worth its latency; if it does not, Round X is unlikely to
rescue it.
"""
import sys, json, collections, glob
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np
from evalkit import ROOT, load_jsonl, load_taxonomy, embed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sig = sys.argv[1]
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
ENC = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"]
LABELS = load_taxonomy(sig)["labels"]
rank = {l: i for i, l in enumerate(LABELS)}

by = collections.defaultdict(list)
for p in glob.glob(str(ROOT/"data/train"/f"{sig}-*.jsonl")):
    if "balanced" in p: continue
    for l in open(p):
        r = json.loads(l); t = r.get("tier") or r.get("label")
        if t in rank: by[t].append(r["text"])
rng = np.random.default_rng(0)
Xtxt, ytr = [], []
for t in LABELS:
    v = list(dict.fromkeys(by[t]))
    if len(v) > CAP: v = [v[i] for i in rng.choice(len(v), CAP, replace=False)]
    Xtxt += v; ytr += [t]*len(v)

evfiles = {"sensitivity": ["sensitivity-entsec-gold.jsonl", "sensitivity-entsec-contested.jsonl"],
           "complexity":  ["complexity-real-gold.jsonl"],
           "cost":        ["cost-real-gold.jsonl"]}[sig]
ev = [r for f in evfiles if (ROOT/"data/eval"/f).exists() for r in load_jsonl(ROOT/"data/eval"/f)]
yev = [r["tier"] for r in ev]
print(f"{sig}: train {len(ytr)} (cap {CAP}/class)  eval {len(ev)}  {dict(collections.Counter(yev))}")
print(f"\n{'encoder (frozen + linear)':<30}{'acc':>9}{'balanced':>10}")
for e in ENC:
    Xtr = embed(e, Xtxt, normalize=True); Xev = embed(e, [r["text"] for r in ev], normalize=True)
    pr = LogisticRegression(max_iter=4000, class_weight="balanced").fit(Xtr, ytr).predict(Xev)
    print(f"{e.split('/')[-1]:<30}{np.mean(pr==np.array(yev)):8.2%}{balanced_accuracy_score(yev,pr):10.2%}")
