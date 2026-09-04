"""Containment/over-block for a set of models, at every gate. §55's rule: an
accuracy gain on sensitivity is not a result until the safety number is beside it.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
LAB = load_taxonomy("sensitivity")["labels"]; rank = {l: i for i, l in enumerate(LAB)}
rows = load_jsonl(ROOT/"data/eval/sensitivity-entsec-gold.jsonl")
y = [r["tier"] for r in rows]
print(f"n={len(rows)}  (entsec-gold, the published eval)\n")
print(f"{'model':<24}{'acc':>8}" + "".join(f"{g[:4]+' cont':>12}{g[:4]+' OB':>10}" for g in ("CONFIDENTIAL","REGULATED","NEVER_EGRESS")))
for tag in sys.argv[1:]:
    d = ROOT/"models"/tag
    if not (d/"config.json").exists(): print(f"{tag:<24}  (missing)"); continue
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = [m.config.id2label[i] for i in range(m.config.num_labels)]
    pred = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            pred += [order[j] for j in m(**e).logits.argmax(-1).tolist()]
    line = f"{tag:<24}{np.mean([a==b for a,b in zip(pred,y)]):8.2%}"
    for g in ("CONFIDENTIAL","REGULATED","NEVER_EGRESS"):
        thr = rank[g]
        hi = [rank[a]>=thr for a in y]; blk = [rank[a]>=thr for a in pred]
        line += f"{np.mean([b for b,h in zip(blk,hi) if h]):12.2%}{np.mean([b for b,h in zip(blk,hi) if not h]):10.2%}"
    print(line)
