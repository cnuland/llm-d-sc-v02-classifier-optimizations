"""Post-hoc logit adjustment: free accuracy from fixing prior mismatch.

Training corpora are assembled from several sources and their class prior does
not match the eval's. A model trained on a 25/25/25/25 corpus and evaluated on
traffic that is 55% MEDIUM is systematically biased, and no amount of extra
training fixes a mismatch that lives in the decision threshold rather than the
representation.

Logit adjustment is the standard correction: add log(p_target / p_train) to each
class logit. It requires no retraining.

Fitted on a DEV half of the eval and scored on the unseen TEST half, because
fitting per-class offsets on the same rows they are scored on measures the
fitting, not the correction. Reports both halves so overfitting is visible.
"""
import sys, glob, json, itertools, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy, metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sig, evalname, tag = sys.argv[1], sys.argv[2], sys.argv[3]
labels = load_taxonomy(sig)["labels"]
rows = load_jsonl(ROOT / f"data/eval/{evalname}.jsonl")
y = [r["tier"] for r in rows]
d = ROOT / "models" / tag
tok = AutoTokenizer.from_pretrained(str(d))
m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
ml = 512 if ("512" in tag or "mbert" in tag) else 256
L = []
with torch.no_grad():
    for i in range(0, len(rows), 32):
        b = tok([r["text"] for r in rows[i:i+32]], truncation=True, max_length=ml,
                padding=True, return_tensors="pt")
        L.append(m(**b).logits.numpy())
L = np.vstack(L)

idx = np.arange(len(rows)); rng = np.random.default_rng(20260903); rng.shuffle(idx)
dev, test = idx[:len(idx)//2], idx[len(idx)//2:]

def acc(sel, off):
    p = (L[sel] + off).argmax(1)
    return sum(labels[a] == y[i] for a, i in zip(p, sel)) / len(sel)

base_dev, base_test = acc(dev, 0), acc(test, 0)

# coordinate ascent on per-class offsets, fitted on DEV only
off = np.zeros(len(labels))
for _ in range(6):
    for c in range(len(labels)):
        best, bo = acc(dev, off), off[c]
        for delta in np.linspace(-2.0, 2.0, 41):
            trial = off.copy(); trial[c] = delta
            a = acc(dev, trial)
            if a > best:
                best, bo = a, delta
        off[c] = bo

print(f"  {tag} on {evalname}")
print(f"    dev  {base_dev:.4f} -> {acc(dev, off):.4f}   (fitted here — expect a gain)")
print(f"    TEST {base_test:.4f} -> {acc(test, off):.4f}   (unseen — this is the real result)")
print(f"    offsets: " + "  ".join(f"{l[:4]}={o:+.2f}" for l, o in zip(labels, off)))
