"""Round AV's actual prediction: does a middle tier cure triage's BLIND confidence?

§121 found the binary triage gate's least-confident slice contains contested rows
at exactly the base rate -- ratio 1.0x, against 1.7-1.9x for every other gate. It
is not uncertain on hard rows, it is CONFIDENTLY WRONG on them, which is why
abstention cannot rescue it.

§119 hit the same shape on the binary egress gate and the fix was structural, not
a threshold: a middle tier gives uncertain rows somewhere to go. Round AV applied
that fix here (TRIVIAL/STANDARD/HARD). Its accuracy is not the test -- a 3-way
task is harder by construction and 90.2% vs 95.9% compares nothing.

THE test is whether triage3cx's abstention curve enriches for contested rows.
Same rows, same jury, same procedure as §121. If the ratio moves off 1.0x, the
middle tier bought what §119 bought. If it stays at 1.0x, the blindness is a
property of the SIGNAL, not of the binary cut, and no taxonomy change fixes it.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# (display name, model tag, eval prefix) -- binary is the §121 control.
ARMS = [("triage (binary)",  "tri-at1-seed11",  "triage"),
        ("triage (binary)",  "tri-at1-seed22",  "triage"),
        ("triage3cx (3-way)","t3cx-av-seed11",  "triage3cx"),
        ("triage3cx (3-way)","t3cx-av-seed22",  "triage3cx")]
COVS = (0.95, 0.90, 0.80)

def run(tag, pfx):
    gold = load_jsonl(ROOT/"data/eval"/f"{pfx}-real-gold-v2.jsonl")
    cont = load_jsonl(ROOT/"data/eval"/f"{pfx}-real-contested.jsonl")
    rows = [dict(r, _c=0) for r in gold] + [dict(r, _c=1) for r in cont]
    is_c = np.array([r["_c"] for r in rows])
    y    = np.array([r["tier"] for r in rows])
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = np.array([m.config.id2label[i] for i in range(m.config.num_labels)])
    P = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True,
                    max_length=256, padding=True, return_tensors="pt")
            P.append(torch.softmax(m(**e).logits, -1).numpy())
    P = np.concatenate(P)
    conf = P.max(1); pred = order[P.argmax(1)]
    return is_c, y, pred, conf

print(f"{'arm':<20}{'tag':<18}{'contested':>10}{'acc@100%':>10}"
      + "".join(f"{f'drop@{c:.0%}':>26}" for c in COVS))
print(f"{'':38}{'share':>10}{'':>10}"
      + "".join(f"{'contested% (ratio) acc':>26}" for _ in COVS))
print("-"*(38+20+26*len(COVS)))
for name, tag, pfx in ARMS:
    is_c, y, pred, conf = run(tag, pfx)
    base = is_c.mean(); acc = (pred == y).mean()
    cells = ""
    for cov in COVS:
        t = np.quantile(conf, 1-cov)
        drop = conf < t; keep = ~drop
        r = is_c[drop].mean() if drop.sum() else float("nan")
        ka = (pred[keep] == y[keep]).mean()
        cells += f"{r:11.1%} ({r/base:.1f}x){ka:9.2%}"
    print(f"{name:<20}{tag:<18}{base:9.1%}{acc:10.2%}{cells}")
