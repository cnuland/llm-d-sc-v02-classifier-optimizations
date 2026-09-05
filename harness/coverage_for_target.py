"""What does high-90s accuracy on the DEPLOYED route decision actually cost?

The standing goal is high-90s. Every headline number in this project is on
jury-unanimous rows, where that is already met (triage 95.93%). Pooled with the
31.9% of real traffic the jury SPLIT on, the same gates sit near 89-91%. The
contested rows are not noise -- they are traffic, and three frontier labellers
disagreed about them.

So the honest question is not "what accuracy" but "what accuracy at what
coverage": how much traffic must escalate to the large model to reach a target on
the rest. Inverts the risk-coverage curve -- for each target, the LOWEST
escalation rate that reaches it, held out (thresholds chosen on one half, scored
on the other) because a threshold fitted on the rows it then scores was worth
about a point.

Confidence is read in the deployed 2-way space per §130. Reported alongside is
the ceiling: accuracy on jury-unanimous rows alone, which is what a gate scores
when the hard traffic is simply not counted.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

FOLD = {"SIMPLE": "SIMPLE", "MEDIUM": "SIMPLE",
        "COMPLEX": "REASONING", "REASONING": "REASONING"}
ARMS = [("complexity 4-way -> route", "cx-ab-v2rubric-seed11", True),
        ("complexity 4-way -> route", "cx-ab-v2rubric-seed22", True),
        ("cx2 native binary",         "cx2-am1-mini-seed11",  False),
        ("cx2 native binary",         "cx2-am1-mini-seed22",  False)]
TARGETS = (0.95, 0.97, 0.98, 0.99)

gold = load_jsonl(ROOT/"data/eval/cx2-real-gold-v2.jsonl")
cont = load_jsonl(ROOT/"data/eval/cx2-real-contested.jsonl")
rows = [dict(r, _c=0) for r in gold] + [dict(r, _c=1) for r in cont]
texts = [r["text"] for r in rows]
y = np.array([r["tier"] for r in rows]); is_c = np.array([r["_c"] for r in rows])

def score(tag, fourway):
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = [m.config.id2label[i] for i in range(m.config.num_labels)]
    P = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            e = tok(texts[i:i+64], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            P.append(torch.softmax(m(**e).logits, -1).numpy())
    P = np.concatenate(P)
    if fourway:
        iS = [i for i, l in enumerate(order) if FOLD[l] == "SIMPLE"]
        iR = [i for i, l in enumerate(order) if FOLD[l] == "REASONING"]
        ps, pr = P[:, iS].sum(1), P[:, iR].sum(1)
    else:
        ps, pr = P[:, order.index("SIMPLE")], P[:, order.index("REASONING")]
    return np.where(ps >= pr, "SIMPLE", "REASONING"), np.maximum(ps, pr)

rng = np.random.default_rng(0)
print(f"n={len(y)}  contested share {is_c.mean():.1%}   "
      f"(escalation = share of traffic sent to the large model)\n")
hdr = "".join(f"{f'>={t:.0%}':>16}" for t in TARGETS)
print(f"{'arm':<28}{'tag':<24}{'all':>8}{'unanim':>8}{hdr}")
print(f"{'':60}{'acc':>8}{'only':>8}" + "".join(f"{'escalate':>16}" for _ in TARGETS))
print("-"*(60+16+16*len(TARGETS)))
for name, tag, fw in ARMS:
    pred, conf = score(tag, fw)
    ok = pred == y
    cells = ""
    for tgt in TARGETS:
        need = []
        for _ in range(40):                       # held-out threshold selection
            idx = rng.permutation(len(ok))
            for a, b in ((idx[:len(idx)//2], idx[len(idx)//2:]),
                         (idx[len(idx)//2:], idx[:len(idx)//2])):
                got = None
                for cov in np.arange(1.00, 0.29, -0.01):
                    t = np.quantile(conf[a], 1-cov)
                    keep = conf[b] >= t
                    if keep.sum() >= 20 and ok[b][keep].mean() >= tgt:
                        got = 1-cov; break
                need.append(got if got is not None else np.nan)
        m_ = np.nanmean(need) if np.any(~np.isnan(need)) else float("nan")
        frac = np.mean(~np.isnan(need))
        cells += (f"{m_:14.0%}  " if frac > 0.5 else f"{'unreachable':>16}")
    print(f"{name:<28}{tag:<24}{ok.mean():7.2%}{ok[is_c==0].mean():8.2%}{cells}")
