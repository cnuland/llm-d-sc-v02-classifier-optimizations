"""EXPERIMENT 04 -- encoder x decision-rule, all frozen, no fine-tuning.

exp01 was confounded: it probed embeddings from a model already fine-tuned on
the very data used to fit the probe, so the encoder had absorbed the signal and
the head had little left to add. Here every encoder is NEUTRAL (never seen this
task), which isolates two questions:

  which base encoder carries the most task signal out of the box, and
  how much does a learned head add over cosine-to-fixed-anchors?

Fine-tuning goes on top of whatever wins; picking that base first avoids
fine-tuning five models to find out.
"""
import sys, time; sys.path.insert(0,"/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *
from sklearn.linear_model import LogisticRegression
import numpy as np

ENCODERS = [
    "sentence-transformers/all-MiniLM-L6-v2",      # what ships today, 23M
    "sentence-transformers/all-mpnet-base-v2",     # 110M classic strong ST
    "BAAI/bge-base-en-v1.5",                       # 109M
    "intfloat/e5-base-v2",                         # 109M
    "Alibaba-NLP/gte-base-en-v1.5",                # 137M
    "mixedbread-ai/mxbai-embed-large-v1",          # 335M
]
TRAIN = {"sensitivity": GENESIS/"training/data/sensitivity-train.jsonl",
         "cost":        GENESIS/"training/data/cost-train.jsonl"}

for sig in ("sensitivity","cost"):
    tax=load_taxonomy(sig); labels=tax["labels"]; ho=heldout(sig); tr=load_jsonl(TRAIN[sig])
    a_txt=[t for l in labels for t in tax["anchors"][l]]
    a_lab=[l for l in labels for _ in tax["anchors"][l]]
    y_tr=[r["tier"] for r in tr]; y_ho=[r["tier"] for r in ho]
    hard=[r.get("hard") for r in ho]
    print(f"\n########## {sig}  (heldout n={len(ho)}, train n={len(tr)}) ##########")
    print(f"  {'encoder':<40}{'anchor':>9}{'head':>9}{'delta':>9}{'dim':>6}{'enc_ms':>8}")
    for mid in ENCODERS:
        try:
            t0=time.time()
            hv=embed(mid,[r["text"] for r in ho]); tv=embed(mid,[r["text"] for r in tr])
            av=embed(mid,a_txt); el=(time.time()-t0)*1000/max(1,len(ho)+len(tr)+len(a_txt))
            pa,_=anchor_topk_predict(hv,av,a_lab,labels,tax.get("top_k",3))
            aacc=metrics(y_ho,pa,labels)["accuracy"]
            clf=LogisticRegression(max_iter=5000,C=10.0,class_weight="balanced").fit(
                unit(tv.astype(np.float64)),y_tr)
            hacc=metrics(y_ho,list(clf.predict(unit(hv.astype(np.float64)))),labels)["accuracy"]
            print(f"  {mid.split('/')[-1]:<40}{aacc:>9.4f}{hacc:>9.4f}{hacc-aacc:>+9.4f}"
                  f"{hv.shape[1]:>6}{el:>8.2f}")
        except Exception as e:
            print(f"  {mid.split('/')[-1]:<40}  FAILED {type(e).__name__}: {str(e)[:60]}")
