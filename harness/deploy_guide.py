"""Held-out operating guidance for every published gate.

Each gate is published with an argmax number. A deployer needs something else:
"if I want X% recall on the class that matters, what threshold do I set, and what
will it actually cost me?" — measured on rows the threshold did not see.

§107 found that answer is not what the fitted curve says. On the egress gate, a
threshold tuned for 95% containment delivered 94.04% held out and cost 17.13%
over-block against the fitted 8.87%. **Fitted operating curves are optimistic at
the high end and the gap grows with the target.** This runs the honest version for
all five gates.

Columns: the recall target asked for, what is actually delivered on held-out rows,
and the false-positive rate that comes with it. The shortfall column is the margin
to add when setting the threshold.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GATES = [("triage",    "tri-at1-seed11",   "triage-real-gold-v2.jsonl",    "TRIVIAL", "route to the cheap path"),
         ("route",     "rt-af1-seed11",    "cx2-real-gold-v2.jsonl",       "REASONING","route to the large model"),
         ("reasoning", "rs-ae1-seed11",    "reasoning-real-gold-v2.jsonl", "YES",     "use a reasoning model"),
         ("genlen",    "gl-ae2-seed11",    "genlen-real-gold-v2.jsonl",    "LONG",    "reserve a long budget"),
         ("egress",    "eg-ad1-cw-seed11", "egress-entsec-gold.jsonl",     "BLOCK",   "block egress")]
REPEATS = 40

for name, tag, evf, pos, action in GATES:
    p = ROOT/"data/eval"/evf
    if not p.exists(): continue
    rows = load_jsonl(p); y = np.array([r["tier"] == pos for r in rows])
    d = ROOT/"models"/tag
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
    order = [m.config.id2label[i] for i in range(m.config.num_labels)]
    if pos not in order:                       # route uses SMALL/LARGE naming
        alt = {"REASONING": "LARGE"}.get(pos)
        j = order.index(alt)
    else:
        j = order.index(pos)
    s = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                    padding=True, return_tensors="pt")
            s.append(torch.softmax(m(**e).logits, -1).numpy()[:, j])
    s = np.concatenate(s)
    print(f"\n### {name}  — positive class fires '{action}'  ({y.sum()}/{len(y)} = {y.mean():.1%})")
    print(f"  {'want recall':>12}{'get (held out)':>16}{'shortfall':>11}{'false-fire rate':>17}")
    rng = np.random.default_rng(0)
    for target in (0.90, 0.95, 0.99):
        ach, fp = [], []
        for _ in range(REPEATS):
            idx = rng.permutation(len(y))
            for a, b in ((idx[:len(idx)//2], idx[len(idx)//2:]), (idx[len(idx)//2:], idx[:len(idx)//2])):
                cand = [t for t in np.unique(np.round(s[a], 3)) if (s[a] >= t)[y[a]].mean() >= target]
                if not cand: continue
                t = max(cand); q = s[b] >= t
                ach.append(q[y[b]].mean()); fp.append(q[~y[b]].mean())
        if not ach: print(f"  {target:12.0%}   unreachable"); continue
        print(f"  {target:12.0%}{np.mean(ach):16.2%}{np.mean(ach)-target:+11.2%}{np.mean(fp):17.2%}")
