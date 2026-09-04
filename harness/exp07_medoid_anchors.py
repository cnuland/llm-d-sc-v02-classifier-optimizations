"""EXPERIMENT 07 -- anchors as MEDOIDS of labelled data. Still a config change.

Experiment 06 tried to pick anchors by maximising dev accuracy and overfitted
hard: dev +9.1 points, test -0.5, legacy held-out -23.8. A 48-slot discrete
search over 1,280 candidates against 209 dev rows fits the split.

So select without looking at any eval at all. Anchors should be the most
REPRESENTATIVE examples of each tier in the corpus we want to serve, which is a
clustering question, not a search question:

  k-medoids per label (k = the incumbent anchor count), farthest-point
  initialised then Lloyd-style reassignment, medoid = the real example closest
  to its cluster's mean.

Two properties matter. It cannot overfit the eval, because the eval is never
consulted. And when the pool is REAL traffic, the anchors land inside the real
distribution -- which is the register gap that costs the shipped models 25-44
points, closed with no retraining and no runtime change.
"""
import sys, json, collections, random
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *
import numpy as np

SIG = sys.argv[1] if len(sys.argv) > 1 else "complexity"
POOLS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["v2"]
tax = load_taxonomy(SIG); labels = tax["labels"]; K = tax.get("top_k", 3)
mid, rev = tax["model_repo"], tax.get("model_revision")

rows = []
for p in POOLS:
    f = ROOT / f"data/train/{SIG}-{p}.jsonl"
    if f.exists():
        r = load_jsonl(f); rows += r; print(f"  pool {p}: {len(r)}")
    else:
        print(f"  pool {p}: MISSING")
if not rows:
    sys.exit("no pool")


def kmedoids(X, k, seed=0, iters=25):
    """Farthest-point init then Lloyd reassignment; medoid = closest real point
    to the cluster mean. Cosine geometry, so X must be unit-normalised."""
    rnd = np.random.default_rng(seed)
    n = len(X)
    if n <= k:
        return list(range(n))
    cur = [int(rnd.integers(n))]
    while len(cur) < k:                      # farthest-point: spread the seeds
        d = 1 - (X @ X[cur].T).max(axis=1)
        d[cur] = -1
        cur.append(int(d.argmax()))
    for _ in range(iters):
        assign = (X @ X[cur].T).argmax(axis=1)
        new = []
        for j in range(k):
            mem = np.where(assign == j)[0]
            if len(mem) == 0:
                new.append(cur[j]); continue
            mu = X[mem].mean(0)
            mu /= max(1e-9, np.linalg.norm(mu))
            new.append(int(mem[(X[mem] @ mu).argmax()]))
        if new == cur:
            break
        cur = new
    return cur


by = collections.defaultdict(list)
for r in rows:
    by[r["tier"]].append(r["text"])
chosen = {}
for l in labels:
    txt = by[l]
    if not txt:
        print(f"  !! no candidates for {l}; keeping incumbents")
        chosen[l] = list(tax["anchors"][l]); continue
    k = len(tax["anchors"][l])
    V = unit(embed(mid, txt, revision=rev).astype(np.float64))
    idx = kmedoids(V, k, seed=abs(hash(l)) % 10000)
    chosen[l] = [txt[i] for i in idx]
    print(f"  {l}: {len(txt)} candidates -> {len(chosen[l])} medoid anchors")

inc_txt = [t for l in labels for t in tax["anchors"][l]]
inc_lab = [l for l in labels for _ in tax["anchors"][l]]
new_txt = [t for l in labels for t in chosen[l]]
new_lab = [l for l in labels for _ in chosen[l]]
IV, NV = embed(mid, inc_txt, revision=rev), embed(mid, new_txt, revision=rev)

sets = [("heldout-v1", heldout(SIG))]
for nm, f in [("real-gold", f"{SIG}-real-gold"),
              ("real-contested", f"{SIG}-real-contested"),
              ("enterprise-gold", f"{SIG}-enterprise-gold")]:
    p = ROOT / f"data/eval/{f}.jsonl"
    if p.exists():
        sets.append((nm, load_jsonl(p)))

print(f"\n  {'eval':<18}{'incumbent':>20}{'medoid':>20}{'delta':>9}")
for nm, rs in sets:
    qv = embed(mid, [r["text"] for r in rs], revision=rev)
    y = [r["tier"] for r in rs]
    pi, _ = anchor_topk_predict(qv, IV, inc_lab, labels, K)
    pn, _ = anchor_topk_predict(qv, NV, new_lab, labels, K)
    a, b = metrics(y, pi, labels), metrics(y, pn, labels)
    print(f"  {nm:<18}{a['accuracy']:.4f}/F1={a['macro_f1']:.3f}"
          f"{b['accuracy']:>11.4f}/F1={b['macro_f1']:.3f}"
          f"{b['accuracy']-a['accuracy']:>+9.4f}")

out = ROOT / f"data/anchors-{SIG}-medoid-{'-'.join(POOLS)}.json"
json.dump({**{k: tax[k] for k in ("classifier_id","signal","model_repo",
                                  "model_revision","method","top_k","labels")},
           "taxonomy_revision": tax["taxonomy_revision"] + "-medoid",
           "anchors": chosen}, open(out, "w"), indent=1)
print(f"  -> {out}")
