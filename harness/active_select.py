"""Active learning: spend the labelling budget where the model is uncertain.

Every previous round labelled a RANDOM slice of traffic. But the errors are not
randomly distributed -- 78% are adjacent-tier confusions, concentrated on
SIMPLE/MEDIUM and MEDIUM/COMPLEX. Random labelling therefore buys mostly
prompts the model already gets right.

This scores the unlabelled pool with the current best model and keeps the ones
it is least sure about, measured by MARGIN: the gap between the top two softmax
probabilities. A small margin means the model is genuinely torn between two
tiers, which is precisely the boundary the confusion matrix says it cannot see.

Two guards, because uncertainty sampling has two well-known failure modes:

  * The selection is capped per predicted tier, so a single ambiguous region
    cannot swallow the whole budget and skew the class prior.
  * A fraction of the budget stays RANDOM. Pure uncertainty sampling drifts the
    training distribution away from real traffic, and the class prior is part of
    what the model has to learn -- real sensitivity traffic really is ~85%
    PUBLIC, and a corpus that forgets that will over-fire the rare tiers.
"""
from __future__ import annotations
import sys, json, argparse, collections, random
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from sdg import key

ap = argparse.ArgumentParser()
ap.add_argument("signal")
ap.add_argument("--model", required=True, help="path under models/")
ap.add_argument("--budget", type=int, default=6000)
ap.add_argument("--random-frac", type=float, default=0.30)
ap.add_argument("--maxlen", type=int, default=256)
a = ap.parse_args()

labels = load_taxonomy(a.signal)["labels"]
pool = load_jsonl(ROOT / "data/eval/wildchat_pool.jsonl")

# exclude everything already spent on eval or training
seen = set()
for f in [f"data/eval/{a.signal}-real-gold.jsonl",
          f"data/eval/{a.signal}-real-contested.jsonl",
          f"data/train/{a.signal}-real.jsonl",
          f"data/train/{a.signal}-real-contested.jsonl"]:
    p = ROOT / f
    if p.exists():
        seen |= {key(r["text"]) for r in load_jsonl(p)}
cand = [r for r in pool if key(r["text"]) not in seen]
print(f"pool={len(pool)}  already spent={len(seen)}  candidates={len(cand)}")
if not cand:
    sys.exit("no unlabelled candidates left; harvest more first")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / a.model))
m = AutoModelForSequenceClassification.from_pretrained(str(ROOT / "models" / a.model))
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m.to(dev).eval()

margins, preds = [], []
with torch.no_grad():
    for i in range(0, len(cand), 128):
        b = tok([r["text"] for r in cand[i:i+128]], truncation=True,
                max_length=a.maxlen, padding=True, return_tensors="pt").to(dev)
        pr = m(**b).logits.softmax(-1).cpu().numpy()
        top2 = np.sort(pr, axis=1)[:, -2:]
        margins += (top2[:, 1] - top2[:, 0]).tolist()
        preds += [labels[j] for j in pr.argmax(1)]
        if i % 12800 == 0:
            print(f"  scored {i}/{len(cand)}", flush=True)

n_rand = int(a.budget * a.random_frac)
n_unc = a.budget - n_rand
per_tier = max(1, n_unc // len(labels))

order = np.argsort(margins)                      # smallest margin first
chosen, taken = [], collections.Counter()
for i in order:
    t = preds[i]
    if taken[t] >= per_tier:
        continue
    chosen.append(i); taken[t] += 1
    if len(chosen) >= n_unc:
        break
rest = [i for i in range(len(cand)) if i not in set(chosen)]
random.Random(20260903).shuffle(rest)
chosen += rest[:n_rand]

out = ROOT / f"data/train/{a.signal}-active-pool.jsonl"
with open(out, "w") as fh:
    for i in chosen:
        fh.write(json.dumps({"text": cand[i]["text"], "src": "wildchat",
                             "margin": round(float(margins[i]), 5),
                             "model_pred": preds[i]}) + "\n")
sel = [margins[i] for i in chosen[:n_unc]]
print(f"selected {len(chosen)}  ({n_unc} uncertainty + {n_rand} random)")
print(f"  uncertainty margins: min={min(sel):.4f} median={np.median(sel):.4f} max={max(sel):.4f}")
print(f"  pool-wide median margin for comparison: {np.median(margins):.4f}")
print(f"  predicted-tier mix of the uncertain set: {dict(taken)}")
print(f"  -> {out}")
