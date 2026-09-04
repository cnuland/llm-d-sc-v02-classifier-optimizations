"""Gate-level safety: what a sensitivity classifier actually does in deployment.

Tier-exact recall is the wrong metric for an egress gate. If the gate blocks
CONFIDENTIAL and above, then classifying a REGULATED prompt as CONFIDENTIAL is
WRONG BUT SAFE -- it is still blocked. Classifying it as INTERNAL leaks it.

Measured on the best model, REGULATED tier-exact recall is 0.65, but 100 of 120
REGULATED prompts (83%) are classified at or above CONFIDENTIAL and would be
blocked. Exact recall understates the safety of the deployed system by 18 points.

So report two numbers per gating threshold:
  containment  fraction of at-or-above-threshold content correctly blocked
               (the safety number -- what leaks)
  over-block   fraction of below-threshold content wrongly blocked
               (the cost number -- unnecessary round trips or refusals)

A gate is characterised by the pair. Reporting either alone hides the trade.
"""
import sys, json, glob, torch
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sig = "sensitivity"
labels = load_taxonomy(sig)["labels"]          # ordered PUBLIC -> NEVER_EGRESS
rank = {l: i for i, l in enumerate(labels)}
rows = load_jsonl(ROOT / f"data/eval/{sig}-entsec-gold.jsonl")
y = [r["tier"] for r in rows]

tags = sorted({p.split("/")[-2] for p in glob.glob(str(ROOT/"models"/"*"/"config.json"))})
tags = [t for t in tags if t.startswith("se-") and "smoke" not in t]
print(f"  eval n={len(rows)}\n")
print(f"  {'model':<20}" + "".join(f"{'block>='+l[:6]:>26}" for l in ["CONFIDENTIAL", "REGULATED"]))
print(f"  {'':<20}" + "".join(f"{'contained':>13}{'over-block':>13}" for _ in range(2)))
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
    pred = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            b = tok([r["text"] for r in rows[i:i+32]], truncation=True, max_length=ml,
                    padding=True, return_tensors="pt")
            pred += [labels[j] for j in m(**b).logits.argmax(-1).tolist()]
    line = f"  {tag[:19]:<20}"
    for thr in ["CONFIDENTIAL", "REGULATED"]:
        t = rank[thr]
        sens = [(g, p) for g, p in zip(y, pred) if rank[g] >= t]
        pub = [(g, p) for g, p in zip(y, pred) if rank[g] < t]
        contained = sum(rank[p] >= t for _, p in sens) / max(1, len(sens))
        over = sum(rank[p] >= t for _, p in pub) / max(1, len(pub))
        line += f"{contained:>13.3f}{over:>13.3f}"
    print(line)
