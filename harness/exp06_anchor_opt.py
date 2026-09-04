"""EXPERIMENT 06 -- choose better ANCHORS. No retraining, no runtime change.

llm-d-sc stores anchors as plain text and embeds them at load
(taxonomy.rs: anchors: BTreeMap<String, Vec<String>>), so the anchor set is
ordinary configuration. Everything else in this project needs a new model on the
Hub or a change to the Rust scorer; this needs a new `classifiers/<sig>.json`.

Experiment 08 also showed the anchors are doing more work than they get credit
for: the embed arm held 0.950 on hand-authored prompts while a softmax head on
the same corpus fell to 0.550, because scoring against human-written anchor text
pins the decision boundary to human-written prompts. If anchors are that load-
bearing, choosing them deliberately should pay.

Method: greedy forward selection. Start empty, repeatedly add the candidate that
most improves accuracy on a DEV split, stop at the incumbent size. Greedy is the
right tool here -- anchor-topk-mean is not differentiable in the anchor set, the
objective is exactly the metric we care about, and k is small.

Selection uses DEV only; the reported number is on a TEST split the search never
saw. Otherwise this measures the search, not the anchors.
"""
import sys, json, random, collections, pathlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *
import numpy as np

SIG = sys.argv[1] if len(sys.argv) > 1 else "complexity"
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
tax = load_taxonomy(SIG); labels = tax["labels"]; K = tax.get("top_k", 3)
mid = MODEL or tax["model_repo"]; rev = None if MODEL else tax.get("model_revision")

# candidate anchor pool: real labelled traffic + synthetic, plus the incumbents
cands = []
for f in [f"data/train/{SIG}-real.jsonl", f"data/train/{SIG}-v2.jsonl"]:
    p = ROOT / f
    if p.exists():
        cands += load_jsonl(p)
if not cands:
    sys.exit(f"no candidate pool for {SIG}; build training data first")
for l in labels:
    cands += [{"text": t, "tier": l, "source": "incumbent"} for t in tax["anchors"][l]]

# dev/test split of the honest eval; selection never sees test
gold = load_jsonl(ROOT / f"data/eval/{SIG}-real-gold.jsonl")
rnd = random.Random(20260903); rnd.shuffle(gold)
half = len(gold) // 2
dev, test = gold[:half], gold[half:]

per_label = collections.defaultdict(list)
for c in cands:
    per_label[c["tier"]].append(c["text"])
# cap the pool: greedy is O(pool * k * |dev|) and 20k candidates buys nothing
POOL = 320
pool_txt, pool_lab = [], []
for l in labels:
    xs = per_label[l]; rnd.shuffle(xs)
    for t in xs[:POOL]:
        pool_txt.append(t); pool_lab.append(l)
print(f"{SIG}: model={mid} pool={len(pool_txt)} dev={len(dev)} test={len(test)} k={K}")

PV = unit(embed(mid, pool_txt, revision=rev).astype(np.float64))
DV = unit(embed(mid, [r["text"] for r in dev], revision=rev).astype(np.float64))
TV = unit(embed(mid, [r["text"] for r in test], revision=rev).astype(np.float64))
dev_y = [r["tier"] for r in dev]; test_y = [r["tier"] for r in test]
S_dev, S_test = DV @ PV.T, TV @ PV.T          # precompute every similarity once
li = {l: i for i, l in enumerate(labels)}


def score(S, chosen):
    """Per-label top-k mean over the chosen anchor columns."""
    out = np.full((S.shape[0], len(labels)), -np.inf)
    for l, idxs in chosen.items():
        if not idxs:
            continue
        c = S[:, idxs]; k = min(K, c.shape[1])
        out[:, li[l]] = np.sort(c, axis=1)[:, -k:].mean(axis=1)
    return out


def acc(S, chosen, y):
    sc = score(S, chosen)
    if not np.isfinite(sc).any():
        return 0.0
    pred = [labels[i] for i in sc.argmax(1)]
    return sum(a == b for a, b in zip(pred, y)) / len(y)


target = {l: len(tax["anchors"][l]) for l in labels}
chosen = {l: [] for l in labels}
by_label_idx = {l: [i for i, x in enumerate(pool_lab) if x == l] for l in labels}

total = sum(target.values())
for step in range(total):
    best = (-1.0, None, None)
    for l in labels:
        if len(chosen[l]) >= target[l]:
            continue
        for i in by_label_idx[l]:
            if i in chosen[l]:
                continue
            chosen[l].append(i)
            a = acc(S_dev, chosen, dev_y)
            chosen[l].pop()
            if a > best[0]:
                best = (a, l, i)
    if best[1] is None:
        break
    chosen[best[1]].append(best[2])
    if (step + 1) % 10 == 0 or step == total - 1:
        print(f"  step {step+1}/{total}  dev={best[0]:.4f}", flush=True)

# incumbent baseline, same model, same scoring
inc_txt = [t for l in labels for t in tax["anchors"][l]]
inc_lab = [l for l in labels for _ in tax["anchors"][l]]
IV = embed(mid, inc_txt, revision=rev)
for nm, rows in [("dev", dev), ("test", test),
                 ("heldout-v1", heldout(SIG))]:
    qv = embed(mid, [r["text"] for r in rows], revision=rev)
    p_inc, _ = anchor_topk_predict(qv, IV, inc_lab, labels, K)
    m_inc = metrics([r["tier"] for r in rows], p_inc, labels)
    Q = unit(qv.astype(np.float64)) @ PV.T
    sc = score(Q, chosen)
    p_new = [labels[i] for i in sc.argmax(1)]
    m_new = metrics([r["tier"] for r in rows], p_new, labels)
    print(f"  {nm:<12} incumbent={m_inc['accuracy']:.4f}/F1={m_inc['macro_f1']:.4f}"
          f"   optimised={m_new['accuracy']:.4f}/F1={m_new['macro_f1']:.4f}"
          f"   delta={m_new['accuracy']-m_inc['accuracy']:+.4f}")

out = ROOT / f"data/anchors-{SIG}-optimised.json"
json.dump({"classifier_id": tax["classifier_id"], "signal": tax["signal"],
           "taxonomy_revision": tax["taxonomy_revision"] + "-opt",
           "model_repo": mid, "model_revision": rev or "", "method": "anchor-topk-mean",
           "top_k": K, "labels": labels,
           "anchors": {l: [pool_txt[i] for i in chosen[l]] for l in labels}},
          open(out, "w"), indent=1)
print(f"  -> {out}")
