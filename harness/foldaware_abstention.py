"""What does it COST to abstain using the trained label space instead of the
deployed one?

Established in the two runs above: t3cx's uncertainty is real and
well-calibrated (rescued rows are 71-75% jury-contested vs a 31.9% pool), but
half of it sits on the STANDARD/HARD boundary -- interior to WORK -- where the
binary decision is already correct on 100.0% of rows. Abstaining on those rows
spends budget escalating traffic that was never going to be misrouted.

So there are three ways to run abstention on a 3-way head deployed binary:
  naive-3way  : confidence = max over 3 classes      (what you get by default)
  fold-aware  : confidence = max(P(TRIVIAL), P(STANDARD)+P(HARD))
  binary head : a separately trained 2-class model   (the §121 status quo)
Ranked by kept-accuracy at matched coverage, and by how much of the escalation
budget each one spends on rows the deployed gate already gets right -- WASTE,
the number that decides whether this is worth deploying.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

gold = load_jsonl(ROOT/"data/eval"/"triage-real-gold-v2.jsonl")
cont = load_jsonl(ROOT/"data/eval"/"triage-real-contested.jsonl")
rows = [dict(r, _c=0) for r in gold] + [dict(r, _c=1) for r in cont]
texts = [r["text"] for r in rows]
y  = np.array([r["tier"] for r in rows])

def load(tag):
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
    return np.concatenate(P), order

COVS = (0.98, 0.95, 0.90, 0.85, 0.80)
print(f"{'seed':<6}{'strategy':<14}{'acc@100%':>9}" +
      "".join(f"{f'@{c:.0%}':>17}" for c in COVS))
print(f"{'':20}{'':>9}" + "".join(f"{'acc  (waste)':>17}" for _ in COVS))
print("-"*(20+9+17*len(COVS)))
for seed in (11, 22):
    Pb, ob = load(f"tri-at1-seed{seed}")
    Pt, ot = load(f"t3cx-av-seed{seed}")
    iS, iH, iT = ot.index("STANDARD"), ot.index("HARD"), ot.index("TRIVIAL")
    pw = Pt[:, [iS, iH]].sum(1)
    pred_f = np.where(pw >= Pt[:, iT], "WORK", "TRIVIAL")

    arms = [("binary head", np.array(ob)[Pb.argmax(1)], Pb.max(1)),
            ("naive-3way",  pred_f, Pt.max(1)),
            ("fold-aware",  pred_f, np.maximum(pw, Pt[:, iT]))]
    for name, pred, conf in arms:
        line = f"{seed:<6}{name:<14}{(pred==y).mean():8.2%} "
        for cov in COVS:
            t = np.quantile(conf, 1-cov); drop = conf < t; keep = ~drop
            ka = (pred[keep] == y[keep]).mean()
            # WASTE: of the rows we paid to escalate, how many were already right?
            waste = (pred[drop] == y[drop]).mean() if drop.sum() else float("nan")
            line += f"{ka:10.2%} ({waste:.0%})"
        print(line)
    print()
