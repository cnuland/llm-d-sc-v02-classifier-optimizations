"""WHY the 3-way head's 2.0x enrichment collapses to 1.2x under the fold.

Unfolded, t3cx's least-confident 5% is 2.0x enriched for contested rows. Folded
to the binary decision that ships, the same head gives 1.2x. The fold is a SUM --
P(WORK)=P(STANDARD)+P(HARD) -- so any uncertainty that lives ENTIRELY on the
STANDARD/HARD boundary cancels exactly: a row split 0.45/0.45/0.10 is maximally
uncertain 3-way and a confident 0.90 WORK after folding.

Prediction, if that is the mechanism: the rows dropped 3-way but KEPT after
folding should be overwhelmingly STANDARD-vs-HARD contests, and the binary
decision should already be right on them. That would mean the middle tier
created real, well-calibrated uncertainty about a boundary the SHIPPING decision
does not care about -- valuable only if three tiers are actually deployed.
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
yb = np.array([r["tier"] for r in rows])
is_c = np.array([r["_c"] for r in rows])

for seed in (11, 22):
    d = ROOT/"models"/f"t3cx-av-seed{seed}"
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
    iS, iH, iT = order.index("STANDARD"), order.index("HARD"), order.index("TRIVIAL")
    conf3 = P.max(1)
    p_work = P[:, [iS, iH]].sum(1)
    conf_f = np.maximum(p_work, P[:, iT])
    pred_f = np.where(p_work >= P[:, iT], "WORK", "TRIVIAL")

    t3 = np.quantile(conf3, 0.05); tf = np.quantile(conf_f, 0.05)
    drop3, dropf = conf3 < t3, conf_f < tf
    rescued = drop3 & ~dropf          # uncertain 3-way, confident after folding

    # Which boundary is each rescued row contesting? Compare the two runner-up
    # gaps: a STANDARD/HARD contest has both of those high and TRIVIAL low.
    sh = P[:, [iS, iH]].min(1)        # mass on the WEAKER of STANDARD/HARD
    print(f"\n===== seed {seed} =====")
    print(f"  dropped 3-way @95% cov : {drop3.sum():3d}")
    print(f"  of those, RESCUED by the fold (confident once summed): {rescued.sum():3d}"
          f"  ({rescued.sum()/max(1,drop3.sum()):.0%})")
    if rescued.sum():
        print(f"  rescued rows -- mean P(TRIVIAL) {P[rescued, iT].mean():.3f}, "
              f"mean min(P(STANDARD),P(HARD)) {sh[rescued].mean():.3f}")
        print(f"  rescued rows -- binary decision already correct: "
              f"{(pred_f[rescued]==yb[rescued]).mean():.1%}")
        print(f"  rescued rows -- contested share {is_c[rescued].mean():.1%} "
              f"(pool {is_c.mean():.1%})")
    stay = drop3 & dropf
    if stay.sum():
        print(f"  still dropped after folding: {stay.sum():3d} -- "
              f"mean P(TRIVIAL) {P[stay, iT].mean():.3f}, "
              f"binary correct {(pred_f[stay]==yb[stay]).mean():.1%}, "
              f"contested {is_c[stay].mean():.1%}")
