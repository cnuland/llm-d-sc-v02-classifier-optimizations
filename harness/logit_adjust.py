"""Post-hoc logit adjustment for the train/eval prior mismatch (Menon et al.,
ICLR 2021, 'Long-tail learning via logit adjustment').

§55 found the model's dominant unanimous-row error is INTERNAL escalated upward.
The class priors say why without invoking the loss at all:

    tier          train%   eval%   ratio
    INTERNAL       35.1%   51.2%   1.46x   under-represented in training
    CONFIDENTIAL   14.4%    5.8%   0.40x   2.5x OVER-represented in training

A softmax head trained on one prior and evaluated on another is miscalibrated by
exactly log(p_eval/p_train) per class. That correction has NO free parameter, so
applying it is not fitting the eval set -- it is removing a known bias. A tau
sweep is shown alongside only to check the theoretical tau=1 is not a fluke;
tau=1 is the headline because it is the only one not chosen with the answer in
hand.

Reports accuracy AND gate containment, since §55's whole point is that
sensitivity's errors are directional and accuracy alone would hide a safety
regression.
"""
import sys, json, collections, pathlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tag = sys.argv[1] if len(sys.argv) > 1 else "se-u-big-seed11"
LAB = load_taxonomy("sensitivity")["labels"]
rank = {l: i for i, l in enumerate(LAB)}

TRAIN = ["sensitivity-v2.jsonl", "sensitivity-real.jsonl",
         "sensitivity-enterprise.jsonl", "sensitivity-real-contested.jsonl"]
tr = collections.Counter()
for f in TRAIN:
    for l in open(ROOT / "data/train" / f):
        r = json.loads(l); tr[r.get("tier") or r.get("label")] += 1
rows = (load_jsonl(ROOT/"data/eval/sensitivity-entsec-gold.jsonl")
        + load_jsonl(ROOT/"data/eval/sensitivity-entsec-contested.jsonl"))
ev = collections.Counter(r["tier"] for r in rows)

d = ROOT/"models"/tag
tok = AutoTokenizer.from_pretrained(str(d))
m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
order = [m.config.id2label[i] for i in range(m.config.num_labels)]

NT, NE = sum(tr.values()), sum(ev.values())
shift = np.array([np.log((ev[l]/NE) / (tr[l]/NT)) for l in order])

logits = []
with torch.no_grad():
    for i in range(0, len(rows), 64):
        b = [r["text"] for r in rows[i:i+64]]
        enc = tok(b, truncation=True, max_length=256, padding=True, return_tensors="pt")
        logits.append(m(**enc).logits.numpy())
L = np.concatenate(logits)
y = [r["tier"] for r in rows]
unan = [len(set(r["votes"])) == 1 for r in rows]

def report(tau):
    p = [order[i] for i in (L + tau*shift).argmax(1)]
    acc = np.mean([a == b for a, b in zip(p, y)])
    au = np.mean([a == b for a, b, u in zip(p, y, unan) if u])
    # gate at CONFIDENTIAL+: containment = of truly >=CONF, fraction blocked;
    # over-block = of truly <CONF, fraction blocked
    thr = rank["CONFIDENTIAL"]
    hi = [(rank[a] >= thr) for a in y]
    blk = [(rank[a] >= thr) for a in p]
    cont = np.mean([b for b, h in zip(blk, hi) if h])
    over = np.mean([b for b, h in zip(blk, hi) if not h])
    return acc, au, cont, over

print(f"model {tag}   n={len(rows)}   gate=CONFIDENTIAL+\n")
print(f"{'tau':>5} {'acc':>8} {'acc(unan)':>10} {'containment':>12} {'over-block':>11}")
for tau in (0.0, 0.5, 1.0, 1.5, 2.0):
    a, au, c, o = report(tau)
    star = "  <- theoretical, no free parameter" if tau == 1.0 else ("  <- baseline" if tau == 0 else "")
    print(f"{tau:5.1f} {a:8.2%} {au:10.2%} {c:12.2%} {o:11.2%}{star}")
