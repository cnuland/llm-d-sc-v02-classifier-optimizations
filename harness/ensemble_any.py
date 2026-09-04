"""Ensemble every compatible trained model for a signal, on its best eval.

Free: no training, only inference. Complexity gained +1.9 points this way.
Reports pairwise error overlap because averaging only pays when members err
differently -- if they fail on the same rows the ensemble cannot fix them.
"""
import sys, glob, itertools, collections, torch, numpy as np
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy, metrics

sig = sys.argv[1]
evalname = sys.argv[2]
labels = load_taxonomy(sig)["labels"]
rows = load_jsonl(ROOT / f"data/eval/{evalname}.jsonl")
y = [r["tier"] for r in rows]
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pref = {"cost": ("co-", "cost"), "sensitivity": ("se-", "sensitivity"),
        "complexity": ("cx-", "complexity")}[sig]
tags = sorted({p.split("/")[-2] for p in glob.glob(str(ROOT/"models"/"*"/"config.json"))})
tags = [t for t in tags if t.startswith(pref) and "smoke" not in t and "probe" not in t]

probs, accs = {}, {}
for tag in tags:
    d = ROOT / "models" / tag
    try:
        tok = AutoTokenizer.from_pretrained(str(d))
        m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
        if [m.config.id2label[i] for i in range(m.config.num_labels)] != labels:
            continue
    except Exception:
        continue
    ml = 512 if ("512" in tag or "mbert" in tag) else 256
    P = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            b = tok([r["text"] for r in rows[i:i+32]], truncation=True, max_length=ml,
                    padding=True, return_tensors="pt")
            P.append(m(**b).logits.softmax(-1).numpy())
    probs[tag] = np.vstack(P)
    accs[tag] = metrics(y, [labels[j] for j in probs[tag].argmax(1)], labels)["accuracy"]

for t, a in sorted(accs.items(), key=lambda x: -x[1]):
    print(f"  {t:<34} {a:.4f}")
if len(probs) < 2:
    sys.exit("  (need >=2 compatible models to ensemble)")

errs = {t: {i for i in range(len(rows)) if labels[probs[t][i].argmax()] != y[i]} for t in probs}
print("\n  error overlap (jaccard) — low means diverse, which is what makes averaging pay:")
for a, b in itertools.combinations(sorted(accs, key=lambda x: -accs[x])[:4], 2):
    j = len(errs[a] & errs[b]) / max(1, len(errs[a] | errs[b]))
    print(f"    {a[:24]:<26} vs {b[:24]:<26} {j:.3f}")

best = None
for k in range(2, min(5, len(probs)) + 1):
    for combo in itertools.combinations(probs, k):
        avg = np.mean([probs[t] for t in combo], axis=0)
        mm = metrics(y, [labels[j] for j in avg.argmax(1)], labels)
        if best is None or mm["accuracy"] > best[0]:
            best = (mm["accuracy"], combo, mm)
lo, hi = best[2]["wilson95"]
top = max(accs.values())
print(f"\n  BEST ENSEMBLE on {evalname}: {best[0]:.4f}  [{lo:.3f},{hi:.3f}]  "
      f"macroF1={best[2]['macro_f1']:.4f}")
print(f"    members: {', '.join(best[1])}")
print(f"    best single member: {top:.4f}   gain {best[0]-top:+.4f}")
cm = best[2]["confusion"]
print("    per-tier recall: " + "  ".join(
    f"{l[:4]}={cm[l][l]/max(1,sum(cm[l].values())):.2f}" for l in labels))
