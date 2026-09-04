"""Re-score every trained model against the REFINED gold, and report the shift.

The point is not just a higher number. Refining labels changes the measuring
stick, so the only honest presentation shows both readings side by side plus
the new ceiling, and states how much of any gain came from the model versus
from the ruler.

Also recomputes the ensemble, since the members' errors move when the labels do.
"""
import sys, json, glob, itertools, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy, metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sig = sys.argv[1] if len(sys.argv) > 1 else "complexity"
tags = sys.argv[2].split(",") if len(sys.argv) > 2 else None
labels = load_taxonomy(sig)["labels"]

v1 = load_jsonl(ROOT / f"data/eval/{sig}-real-gold.jsonl")
p2 = ROOT / f"data/eval/{sig}-real-gold-v2.jsonl"
if not p2.exists():
    sys.exit(f"no refined gold yet at {p2.name}")
v2 = load_jsonl(p2)
by1 = {r["text"]: r["tier"] for r in v1}
changed = sum(1 for r in v2 if r.get("tier_v1") and r["tier"] != r["tier_v1"])
print(f"{sig}: v1 gold n={len(v1)}   refined gold n={len(v2)}   "
      f"labels changed {changed} ({changed/len(v2):.1%})   "
      f"dropped as non-unanimous {len(v1)-len(v2)}")

if tags is None:
    tags = sorted({p.split("/")[-2] for p in glob.glob(str(ROOT/"models"/"*"/"config.json"))})
    tags = [t for t in tags if t.startswith(("cx-", "co-", "se-", "complexity", "cost", "sensitivity"))]

probs, rows_v2 = {}, [r["text"] for r in v2]
y2 = [r["tier"] for r in v2]
y1_on_v2 = [by1.get(t) for t in rows_v2]
print(f"\n  {'model':<30}{'v1 gold':>10}{'refined':>10}{'delta':>9}")
for tag in tags:
    d = ROOT / "models" / tag
    if not (d / "config.json").exists():
        continue
    try:
        tok = AutoTokenizer.from_pretrained(str(d))
        m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
        if [m.config.id2label[i] for i in range(m.config.num_labels)] != labels:
            continue
    except Exception:
        continue
    ml = 512 if "512" in tag or "mbert" in tag else 256
    P = []
    with torch.no_grad():
        for i in range(0, len(rows_v2), 32):
            b = tok(rows_v2[i:i+32], truncation=True, max_length=ml,
                    padding=True, return_tensors="pt")
            P.append(m(**b).logits.softmax(-1).numpy())
    probs[tag] = np.vstack(P)
    pred = [labels[j] for j in probs[tag].argmax(1)]
    a2 = metrics(y2, pred, labels)["accuracy"]
    pair = [(p, g) for p, g in zip(pred, y1_on_v2) if g]
    a1 = sum(p == g for p, g in pair) / max(1, len(pair))
    print(f"  {tag[:29]:<30}{a1:>10.4f}{a2:>10.4f}{a2-a1:>+9.4f}")

if len(probs) >= 2:
    print("\n  ensembles on refined gold:")
    best = None
    for k in range(2, min(4, len(probs)) + 1):
        for combo in itertools.combinations(probs, k):
            avg = np.mean([probs[t] for t in combo], axis=0)
            mm = metrics(y2, [labels[j] for j in avg.argmax(1)], labels)
            if best is None or mm["accuracy"] > best[0]:
                best = (mm["accuracy"], combo, mm["wilson95"])
    lo, hi = best[2]
    print(f"    best: {'+'.join(t[:16] for t in best[1])}")
    print(f"          acc={best[0]:.4f}  95% CI [{lo:.3f},{hi:.3f}]")
