"""Acquisition by DISAGREEMENT, not uncertainty.

Margin-based selection cannot reach this model's remaining errors: 32 of its 48
residual mistakes carry confidence above 0.90. It is confidently wrong, so the
prompts that would teach it most have the *highest* margin, not the lowest.
That is exactly why the previous active round lifted contested accuracy (+3.4,
where the model is unsure) and left gold untouched.

So invert the search. Take the prompts the model is MOST sure about -- its blind
spot -- screen them with an independent cheap judge, and keep the ones where the
two disagree. A confident model contradicted by a competent judge is either a
real error or a genuinely hard case; both are worth a label, and neither is
findable by margin.

Budget note: screening is deliberately spent only on the high-confidence
region. Screening the low-margin region would rediscover what the previous round
already labelled.
"""
from __future__ import annotations
import sys, json, argparse, collections, random
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from sdg import blind_relabel, key

ap = argparse.ArgumentParser()
ap.add_argument("signal")
ap.add_argument("--model", required=True)
ap.add_argument("--screen", type=int, default=20000, help="how many confident prompts to screen")
ap.add_argument("--budget", type=int, default=6000, help="how many disagreements to keep")
ap.add_argument("--min-margin", type=float, default=0.90)
ap.add_argument("--maxlen", type=int, default=512)
a = ap.parse_args()

labels = load_taxonomy(a.signal)["labels"]
pool = load_jsonl(ROOT / "data/eval/wildchat_pool.jsonl")
spent = set()
for f in [f"data/eval/{a.signal}-real-gold.jsonl", f"data/eval/{a.signal}-real-contested.jsonl",
          f"data/train/{a.signal}-real.jsonl", f"data/train/{a.signal}-real-contested.jsonl",
          f"data/train/{a.signal}-active.jsonl", f"data/train/{a.signal}-active-contested.jsonl"]:
    p = ROOT / f
    if p.exists():
        spent |= {key(r["text"]) for r in load_jsonl(p)}
cand = [r for r in pool if key(r["text"]) not in spent]
print(f"pool={len(pool)}  spent={len(spent)}  candidates={len(cand)}")

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
        s = np.sort(pr, axis=1)
        margins += (s[:, -1] - s[:, -2]).tolist()
        preds += [labels[j] for j in pr.argmax(1)]
        if i % 25600 == 0:
            print(f"  scored {i}/{len(cand)}", flush=True)

conf_idx = [i for i, mg in enumerate(margins) if mg >= a.min_margin]
random.Random(20260903).shuffle(conf_idx)
screen_idx = conf_idx[:a.screen]
print(f"  confident (margin>={a.min_margin}): {len(conf_idx)}  -> screening {len(screen_idx)}")

judged = blind_relabel(a.signal, labels, [cand[i]["text"] for i in screen_idx],
                       model="claude-haiku-4-5-20251001", batch=25, effort="low")
dis = [(i, preds[i], j["label"], margins[i])
       for i, j in zip(screen_idx, judged) if j["label"] and j["label"] != preds[i]]
print(f"  screen disagreed with the model on {len(dis)}/{len(screen_idx)} = "
      f"{len(dis)/max(1,len(screen_idx)):.1%} of CONFIDENT prompts")

# cap per model-predicted tier so one region cannot dominate
per = max(1, a.budget // len(labels))
taken, chosen = collections.Counter(), []
for i, mp, jp, mg in sorted(dis, key=lambda x: -x[3]):   # most confident first
    if taken[mp] >= per:
        continue
    chosen.append((i, mp, jp, mg)); taken[mp] += 1
    if len(chosen) >= a.budget:
        break

out = ROOT / f"data/train/{a.signal}-disagree-pool.jsonl"
with open(out, "w") as fh:
    for i, mp, jp, mg in chosen:
        fh.write(json.dumps({"text": cand[i]["text"], "src": "wildchat",
                             "margin": round(float(mg), 5),
                             "model_pred": mp, "screen_pred": jp}) + "\n")
print(f"  selected {len(chosen)}  (mean margin {np.mean([c[3] for c in chosen]):.4f} "
      f"— i.e. the model was SURE and still contradicted)")
print(f"  model-predicted tier mix: {dict(taken)}")
print(f"  common disagreement pairs: "
      f"{collections.Counter(f'{mp}->{jp}' for _,mp,jp,_ in chosen).most_common(6)}")
print(f"  -> {out}")
