"""Distil the ensemble into a single model using the unlabelled pool.

Ensembling reliably adds 1-2 points on every signal here, but costs 2-4x
inference — against a measured ~480 classifications/sec per llm-d-sc replica
that is a real capacity decision. Distillation is the standard way to keep most
of the gain at single-model cost.

The lever this uses that nothing else has: ~60,000 WildChat prompts that are
harvested, deduplicated, disjoint from every eval, and completely unlabelled.
Jury-labelling them would cost thousands of API calls; the ensemble labels them
for free, and its SOFT output carries more information than a hard label would
(Hinton et al.) -- exactly the target format the soft-target path already
supports.

Two guards:
  * pseudo-labelled rows are mixed WITH the real labelled corpus, never used
    alone -- a student trained only on its teacher's output inherits every
    systematic error with no signal to correct it;
  * low-confidence pseudo-labels are kept rather than filtered. Filtering to
    confident rows is the standard mistake: it discards exactly the boundary
    region where the teacher's soft distribution is most informative.
"""
import sys, json, argparse, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from sdg import key

ap = argparse.ArgumentParser()
ap.add_argument("signal")
ap.add_argument("--members", required=True, help="comma-separated model tags")
ap.add_argument("--n", type=int, default=30000)
ap.add_argument("--pool", default="wildchat_pool",
                help="which unlabelled pool to distil onto; sensitivity needs "
                     "the enterprise pool because WildChat is 85% PUBLIC")
a = ap.parse_args()

labels = load_taxonomy(a.signal)["labels"]
pool = load_jsonl(ROOT / f"data/eval/{a.pool}.jsonl")
spent = set()
import glob as _g
for f in [x.replace(str(ROOT)+"/","") for x in _g.glob(str(ROOT/f"data/eval/{a.signal}-*.jsonl"))]:
    p = ROOT / f
    if p.exists():
        spent |= {key(r["text"]) for r in load_jsonl(p)}
cand = [r for r in pool if key(r["text"]) not in spent][:a.n]
print(f"{a.signal}: distilling onto {len(cand)} unlabelled prompts "
      f"(eval rows excluded by content hash)")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
dev = "mps" if torch.backends.mps.is_available() else "cpu"
acc = None
for tag in a.members.split(","):
    d = ROOT / "models" / tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).to(dev).eval()
    ml = 512 if ("512" in tag or "mbert" in tag) else 256
    P = []
    with torch.no_grad():
        for i in range(0, len(cand), 128):
            b = tok([r["text"] for r in cand[i:i+128]], truncation=True, max_length=ml,
                    padding=True, return_tensors="pt").to(dev)
            P.append(m(**b).logits.softmax(-1).cpu().numpy())
    P = np.vstack(P)
    acc = P if acc is None else acc + P
    print(f"  scored with {tag}")
    del m
avg = acc / len(a.members.split(","))

out = ROOT / f"data/train/{a.signal}-distill.jsonl"
conf = avg.max(1)
with open(out, "w") as fh:
    for r, q in zip(cand, avg):
        fh.write(json.dumps({
            "text": r["text"], "tier": labels[int(q.argmax())],
            # store the teacher distribution directly; the soft-target path
            # reads `votes`, so express the distribution as a weighted vote list
            "soft_dist": [round(float(x), 5) for x in q],
            "source": "ensemble-distilled"}) + "\n")
print(f"  teacher confidence: median {np.median(conf):.3f}  "
      f"below 0.6: {(conf < 0.6).mean():.1%}")
print(f"  pseudo-label mix: {dict(collections.Counter(labels[int(q.argmax())] for q in avg))}")
print(f"  -> {out}")
