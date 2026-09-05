"""Risk-coverage curves: what does abstention buy each gate?

§119 found that adding a REVIEW tier eliminated the worst failure mode at a cost
of 0.99 accuracy points, because an uncertain row got somewhere safe to go. That
is one instance of a general idea this project has never applied: SELECTIVE
PREDICTION. Let the classifier decline on the rows it is least sure about, send
those to a safe default, and measure accuracy on the rest.

The deployment question it answers is concrete: "if the router falls back to a
default on the 10% of traffic it is least confident about, how good is it on the
other 90%?" Every gate here is currently forced to answer on 100% of rows.

Measured HELD OUT, because §103 and §107 both showed that choosing an operating
point on the rows you then score is worth roughly a point of optimism -- and the
abstention threshold is exactly such a choice. 2-fold, 40 repeats.

Reports the coverage needed to reach 99% accuracy on the covered set, since that
is the number that decides whether a fallback path is worth building.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GATES = [("triage",    "tri-at1-seed11",   "triage-real-gold-v2.jsonl"),
         ("route",     "rt-af1-seed11",    "cx2-real-gold-v2.jsonl"),
         ("reasoning", "rs-ae1-seed11",    "reasoning-real-gold-v2.jsonl"),
         ("genlen",    "gl-ae2-seed11",    "genlen-real-gold-v2.jsonl"),
         ("egress",    "eg-ad1-cw-seed11", "egress-entsec-gold.jsonl"),
         ("egress-triage", "t3-au-seed11", "triage3-entsec-gold.jsonl")]

print(f"{'gate':<16}{'full acc':>10}" + "".join(f"{f'@{c:.0%} cov':>10}" for c in (0.95,0.90,0.80,0.70))
      + f"{'cov for 99%':>13}")
for name, tag, evf in GATES:
    p = ROOT/"data/eval"/evf
    if not p.exists(): continue
    rows = load_jsonl(p); y = np.array([r["tier"] for r in rows])
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = np.array([m.config.id2label[i] for i in range(m.config.num_labels)])
    P = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            P.append(torch.softmax(m(**e).logits, -1).numpy())
    P = np.concatenate(P)
    # route was trained with SMALL/LARGE while its eval file uses the cx2
    # vocabulary SIMPLE/REASONING -- the same fold under two names (§100). Without
    # this map every prediction "misses" and the gate reads 0.00%, which is what
    # the first run of this script reported.
    CANON = {"SMALL":"SIMPLE","LARGE":"REASONING"}
    pred = np.array([CANON.get(x, x) for x in order[P.argmax(1)]])
    conf = P.max(1)                      # max-softmax: the standard baseline
    correct = (pred == y)
    full = correct.mean()

    rng = np.random.default_rng(0)
    cells, cov99 = [], []
    for cov in (0.95, 0.90, 0.80, 0.70):
        held = []
        for _ in range(40):
            idx = rng.permutation(len(y))
            for a, b in ((idx[:len(idx)//2], idx[len(idx)//2:]), (idx[len(idx)//2:], idx[:len(idx)//2])):
                t = np.quantile(conf[a], 1-cov)          # threshold from fold A
                keep = conf[b] >= t                       # applied to fold B
                if keep.sum(): held.append(correct[b][keep].mean())
        cells.append(np.mean(held))
    # smallest coverage reaching 99% on the covered set, held out
    for cov in np.arange(0.95, 0.19, -0.05):
        held = []
        for _ in range(20):
            idx = rng.permutation(len(y))
            for a, b in ((idx[:len(idx)//2], idx[len(idx)//2:]), (idx[len(idx)//2:], idx[:len(idx)//2])):
                t = np.quantile(conf[a], 1-cov); keep = conf[b] >= t
                if keep.sum(): held.append(correct[b][keep].mean())
        if np.mean(held) >= 0.99: cov99 = cov; break
    else: cov99 = None
    c99 = f"{cov99:.0%}" if cov99 else "unreachable"
    print(f"{name:<16}{full:9.2%}" + "".join(f"{c:9.2%}" for c in cells) + f"{c99:>13}")
