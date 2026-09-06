"""Do the published gates have usable confidence orderings?

§132 found a natively-binary head whose risk-coverage curve PEAKS at 60% coverage
and falls to 91.57% at 30% -- its most confident rows are not its most correct
ones -- while its accuracy was indistinguishable from a folded 4-way alternative
that reached 99%. The proposed mechanism is saturation: a 2-class softmax on a
hard boundary has one logit difference to express both which side and how sure.

If that mechanism is right it is not one bad model, it is a property of every
natively-binary gate this project trained and published. Every published gate is
binary. This audits them all, on their own real-gold + contested pools, and
reports the shape of each curve plus the spread left in the confident half.

Prediction to falsify: binary-native gates saturate and go non-monotone; the
fold-derived arms do not.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CXFOLD = {"SIMPLE": "SIMPLE", "MEDIUM": "SIMPLE",
          "COMPLEX": "REASONING", "REASONING": "REASONING"}
T3FOLD = {"TRIVIAL": "TRIVIAL", "STANDARD": "WORK", "HARD": "WORK"}

# (display, tag, eval prefix, fold-or-None, gold prefix for labels)
ARMS = [
 ("triage      (binary native)", "tri-at1-seed11", "triage", None, "triage"),
 ("triage      (binary native)", "tri-at1-seed22", "triage", None, "triage"),
 ("reasoning   (binary native)", "rs-ae1-seed11",  "reasoning", None, "reasoning"),
 ("reasoning   (binary native)", "rs-ae1-seed22",  "reasoning", None, "reasoning"),
 ("genlen      (binary native)", "gl-ae2-seed11",  "genlen", None, "genlen"),
 ("genlen      (binary native)", "gl-ae2-seed22",  "genlen", None, "genlen"),
 ("route       (binary native)", "rt-af1-seed11",  "route", None, "route"),
 ("route       (binary native)", "rt-af1-seed22",  "route", None, "route"),
 ("cx2         (binary native)", "cx2-am1-mini-seed11", "cx2", None, "cx2"),
 ("cx2         (binary native)", "cx2-am1-mini-seed22", "cx2", None, "cx2"),
 ("complexity4 (folded->route)", "cx-ab-v2rubric-seed11", "cx2", CXFOLD, "cx2"),
 ("complexity4 (folded->route)", "cx-ab-v2rubric-seed22", "cx2", CXFOLD, "cx2"),
 ("triage3cx   (folded->work)",  "t3cx-av-seed11", "triage", T3FOLD, "triage"),
 ("triage3cx   (folded->work)",  "t3cx-av-seed22", "triage", T3FOLD, "triage"),
]
COV = [1.0, .9, .8, .7, .6, .5, .4, .3]

def curve(tag, pfx, fold, gpfx):
    rows = load_jsonl(ROOT/"data/eval"/f"{gpfx}-real-gold-v2.jsonl") + \
           load_jsonl(ROOT/"data/eval"/f"{gpfx}-real-contested.jsonl")
    texts = [r["text"] for r in rows]; y = np.array([r["tier"] for r in rows])
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
    if fold:
        outs = sorted(set(fold.values()))
        F = np.stack([P[:, [i for i, l in enumerate(order) if fold[l] == o]].sum(1)
                      for o in outs], 1)
        pred = np.array(outs)[F.argmax(1)]; conf = F.max(1)
    else:
        pred = np.array(order)[P.argmax(1)]; conf = P.max(1)
    ok = pred == y
    accs = [ok[conf >= np.quantile(conf, 1-c)].mean() for c in COV]
    top = conf[conf >= np.median(conf)]
    return accs, float(top.max() - top.min())

print(f"{'arm':<30}{'tag':<24}" + "".join(f"{c:>8.0%}" for c in COV)
      + f"{'spread':>9}{'verdict':>16}")
print("-"*(30+24+8*len(COV)+25))
for name, tag, pfx, fold, gpfx in ARMS:
    try:
        accs, spread = curve(tag, pfx, fold, gpfx)
    except Exception as e:
        print(f"{name:<30}{tag:<24}  !! {type(e).__name__}: {str(e)[:50]}"); continue
    drops = [a0-a1 for a0, a1 in zip(accs, accs[1:]) if a0-a1 > 0.005]
    v = (f"FAIL -{max(drops):.1%}" if drops else
         ("WARN saturated" if spread < 0.02 else "pass"))
    print(f"{name:<30}{tag:<24}" + "".join(f"{a:8.1%}" for a in accs)
          + f"{spread:9.3f}{v:>16}")
