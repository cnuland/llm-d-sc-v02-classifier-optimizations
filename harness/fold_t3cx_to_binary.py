"""Is the 3-way head a better instrument for the SAME binary decision?

triage and triage3cx are folds of the SAME complexity labels over the SAME rows
in the SAME order:
    triage    : SIMPLE->TRIVIAL,  MEDIUM/COMPLEX/REASONING->WORK
    triage3cx : SIMPLE->TRIVIAL,  MEDIUM->STANDARD,  COMPLEX/REASONING->HARD
so collapsing {STANDARD,HARD}->WORK reproduces the binary label EXACTLY. That
makes this a matched comparison in the §111-113 sense: identical rows, identical
labels, only the training taxonomy differs. Paired, so McNemar applies -- the
unpaired Wilson intervals used elsewhere would be needlessly wide here.

Two questions, and they are separate:
  1. ACCURACY on the deployed binary decision. Does splitting WORK into
     STANDARD/HARD cost anything at the cut that ships?
  2. CONFIDENCE. §121's finding was that binary triage is confidently wrong on
     jury-contested rows (1.0x enrichment). Folded confidence is
     P(STANDARD)+P(HARD) vs P(TRIVIAL) -- the 3-way head's own uncertainty,
     projected onto the binary decision. If enrichment survives the fold, the
     middle tier bought a usable abstention signal for the SHIPPING gate, not
     merely for a taxonomy nobody deploys.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import binomtest

FOLD = {"TRIVIAL": "TRIVIAL", "STANDARD": "WORK", "HARD": "WORK"}
COVS = (0.95, 0.90, 0.80)

def probs(tag, texts):
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

gold = load_jsonl(ROOT/"data/eval"/"triage-real-gold-v2.jsonl")
cont = load_jsonl(ROOT/"data/eval"/"triage-real-contested.jsonl")
rows = [dict(r, _c=0) for r in gold] + [dict(r, _c=1) for r in cont]
texts = [r["text"] for r in rows]
y = np.array([r["tier"] for r in rows])          # already TRIVIAL/WORK
is_c = np.array([r["_c"] for r in rows])
base = is_c.mean()

for seed in (11, 22):
    Pb, ob = probs(f"tri-at1-seed{seed}", texts)
    Pt, ot = probs(f"t3cx-av-seed{seed}", texts)
    # binary head: confidence is max prob, prediction is argmax
    pred_b = np.array(ob)[Pb.argmax(1)]
    conf_b = Pb.max(1)
    # 3-way head folded: P(WORK) = P(STANDARD)+P(HARD)
    iw = [ot.index(l) for l in ("STANDARD", "HARD")]
    it = ot.index("TRIVIAL")
    p_work = Pt[:, iw].sum(1); p_triv = Pt[:, it]
    pred_t = np.where(p_work >= p_triv, "WORK", "TRIVIAL")
    conf_t = np.maximum(p_work, p_triv)

    ok_b, ok_t = pred_b == y, pred_t == y
    b_only = int((ok_b & ~ok_t).sum()); t_only = int((ok_t & ~ok_b).sum())
    p = binomtest(t_only, b_only + t_only, 0.5).pvalue if b_only + t_only else 1.0

    print(f"\n===== seed {seed} =====  n={len(y)}  contested share {base:.1%}")
    print(f"  binary head          acc {ok_b.mean():7.2%}")
    print(f"  3-way head, folded   acc {ok_t.mean():7.2%}   "
          f"delta {ok_t.mean()-ok_b.mean():+.2%}")
    print(f"  McNemar: binary-only-right {b_only}, 3way-only-right {t_only}, p={p:.4f}")
    print(f"  {'':22}{'contested% (ratio)':>22}{'kept acc':>10}{'  |':>3}"
          f"{'contested% (ratio)':>22}{'kept acc':>10}")
    print(f"  {'coverage':22}{'BINARY':>32}{'  |':>3}{'3-WAY FOLDED':>32}")
    for cov in COVS:
        line = f"  {cov:<22.0%}"
        for conf, pred in ((conf_b, pred_b), (conf_t, pred_t)):
            t = np.quantile(conf, 1-cov); drop = conf < t; keep = ~drop
            r = is_c[drop].mean() if drop.sum() else float("nan")
            line += f"{r:15.1%} ({r/base:.1f}x){(pred[keep]==y[keep]).mean():10.2%}"
            line += "  |" if conf is conf_b else ""
        print(line)
