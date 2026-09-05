"""The deployed router does argmax-then-fold. That is not fold-then-argmax.

llm-d-sc serves the 4-way complexity taxonomy and Praxis routes on a label->cluster
table (SIMPLE/MEDIUM->small, COMPLEX/REASONING->large). So the deployed pipeline
takes the argmax FIRST and folds the winning label SECOND.

Summing inside each route class first gives a different answer, and the gap is not
a rounding detail -- it is every row whose mass is SPLIT across one route class
while a single label in the other route class wins on its own:

    SIMPLE .35  MEDIUM .05 | COMPLEX .30  REASONING .30
    argmax-then-fold -> SIMPLE  -> small      (top label is SIMPLE, .35)
    fold-then-argmax -> .40 vs .60 -> large   (the route class has more mass)

Neither is obviously right a priori. argmax-then-fold answers "what is this
prompt?" and then routes the answer; fold-then-argmax answers "which cluster
should serve this?" directly. The second is the question actually being asked, and
it is the one the deployed system does NOT ask.

§130 established that folding is a sum and that the sum matters for confidence.
This is the same arithmetic reaching the ROUTING DECISION itself, on the signal
that is actually in production (§129). Matched rows, so McNemar.
"""
import sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import binomtest

FOLD = {"SIMPLE": "SIMPLE", "MEDIUM": "SIMPLE",
        "COMPLEX": "REASONING", "REASONING": "REASONING"}
FOURWAY = ["cx-ab-v2rubric-seed11", "cx-ab-v2rubric-seed22", "cx-ab-v2rubric-seed33",
           "cx-t-big-seed33", "cx-ak1-allrows-seed11"]
BINARY  = ["cx2-am1-mini-seed11", "cx2-am1-mini-seed22"]

gold = load_jsonl(ROOT/"data/eval/cx2-real-gold-v2.jsonl")
cont = load_jsonl(ROOT/"data/eval/cx2-real-contested.jsonl")
rows = gold + cont
texts = [r["text"] for r in rows]
y = np.array([r["tier"] for r in rows])            # already the ROUTE label

def probs(tag):
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

print(f"n={len(y)}  (gold {len(gold)} + contested {len(cont)})\n")
print(f"{'model':<24}{'argmax->fold':>14}{'fold->argmax':>14}{'delta':>8}"
      f"{'disagree':>10}{'McNemar':>22}")
print("-"*92)
for tag in FOURWAY:
    P, order = probs(tag)
    lab = np.array(order)
    a = np.array([FOLD[l] for l in lab[P.argmax(1)]])          # deployed path
    iS = [i for i, l in enumerate(order) if FOLD[l] == "SIMPLE"]
    iR = [i for i, l in enumerate(order) if FOLD[l] == "REASONING"]
    b = np.where(P[:, iS].sum(1) >= P[:, iR].sum(1), "SIMPLE", "REASONING")
    ok_a, ok_b = a == y, b == y
    ao, bo = int((ok_a & ~ok_b).sum()), int((ok_b & ~ok_a).sum())
    p = binomtest(bo, ao+bo, 0.5).pvalue if ao+bo else 1.0
    print(f"{tag:<24}{ok_a.mean():13.2%}{ok_b.mean():14.2%}"
          f"{ok_b.mean()-ok_a.mean():+8.2%}{(a!=b).mean():10.1%}"
          f"{f'{ao} vs {bo}, p={p:.4f}':>22}")
print()
for tag in BINARY:
    P, order = probs(tag)
    pred = np.array(order)[P.argmax(1)]
    print(f"{tag:<24}{(pred==y).mean():13.2%}   (native binary head, reference)")
