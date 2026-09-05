"""Does abstention preferentially drop the rows the JURY could not agree on?

Every gate degrades sharply on jury-split rows -- triage falls to 80.11% against
an 82.95% baseline, i.e. below chance -- and those are roughly 30% of real
traffic. §120 showed abstaining on the least-confident 10% takes the reasoning
gate to 99.41%. The obvious question nobody has asked: are those the SAME rows?

If model confidence tracks jury disagreement, the contested-row problem and the
abstention story are one thing, and "send the uncertain slice to the large model"
is a complete answer to both. If it does not, the models are confidently wrong on
contested rows and abstention leaves the problem untouched.

Measures the contested RATE among dropped rows versus kept rows at each coverage
level. A ratio near 1.0 means confidence is blind to disagreement.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GATES = [("triage",    "tri-at1-seed11", "triage"),
         ("route",     "rt-af1-seed11",  "cx2"),
         ("reasoning", "rs-ae1-seed11",  "reasoning"),
         ("genlen",    "gl-ae2-seed11",  "genlen")]
CANON = {"SMALL":"SIMPLE","LARGE":"REASONING"}

print(f"{'gate':<12}{'contested':>11}{'':>3}" + "".join(f"{f'drop@{c:.0%}':>22}" for c in (0.95,0.90,0.80)))
print(f"{'':12}{'share':>11}{'':>3}" + "".join(f"{'contested% (ratio)':>22}" for _ in range(3)))
for name, tag, pfx in GATES:
    gold = load_jsonl(ROOT/"data/eval"/f"{pfx}-real-gold-v2.jsonl")
    cont = load_jsonl(ROOT/"data/eval"/f"{pfx}-real-contested.jsonl")
    rows = [dict(r, _c=0) for r in gold] + [dict(r, _c=1) for r in cont]
    is_c = np.array([r["_c"] for r in rows])
    y = np.array([r["tier"] for r in rows])
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = np.array([CANON.get(m.config.id2label[i], m.config.id2label[i])
                      for i in range(m.config.num_labels)])
    P = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            P.append(torch.softmax(m(**e).logits, -1).numpy())
    P = np.concatenate(P); conf = P.max(1)
    base = is_c.mean()
    cells = ""
    for cov in (0.95, 0.90, 0.80):
        t = np.quantile(conf, 1-cov)
        dropped = conf < t
        r = is_c[dropped].mean() if dropped.sum() else float("nan")
        cells += f"{r:15.1%} ({r/base:.1f}x)"
    print(f"{name:<12}{base:10.1%}{'':>3}{cells}")
