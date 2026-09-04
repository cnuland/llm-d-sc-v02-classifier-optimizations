"""EXPERIMENT 01 -- how much does anchor-topk-mean cost us?

Holds the embedding model FIXED and swaps only the decision rule:
  A. anchor-topk-mean  (what llm-d-sc ships; anchors are text, top-3 mean cosine)
  B. logistic-regression head on the SAME frozen embeddings
  C. centroid prototypes learned from training data (nearest class mean)

If B >> A, the ceiling is the decision rule, not the encoder, and the highest-
leverage change is a softmax head in the Rust runtime. If B ~= A, the encoder is
the bottleneck and we should be fine-tuning / changing base models instead.
"""
import sys; sys.path.insert(0,"/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *
from sklearn.linear_model import LogisticRegression
import numpy as np, json

TRAIN = {"cost": GENESIS/"training/data/cost-train.jsonl",
         "sensitivity": GENESIS/"training/data/sensitivity-train.jsonl"}
MODEL = {"cost": ("cnuland/llm-d-sc-cost", None),
         "sensitivity": ("cnuland/llm-d-sc-sensitivity", None)}

for sig in ("sensitivity","cost"):
    tax=load_taxonomy(sig); labels=tax["labels"]; ho=heldout(sig); tr=load_jsonl(TRAIN[sig])
    mid,rev=MODEL[sig]
    # leakage guard: exact-text overlap between train and held-out
    overlap = set(r["text"].strip() for r in tr) & set(r["text"].strip() for r in ho)
    print(f"\n=== {sig} === train={len(tr)} heldout={len(ho)} leakage={len(overlap)} model={mid}")
    if overlap: print("   !! LEAKED:", list(overlap)[:3])

    a_txt=[t for lab in labels for t in tax["anchors"][lab]]
    a_lab=[lab for lab in labels for _ in tax["anchors"][lab]]
    ho_v=embed(mid,[r["text"] for r in ho],revision=rev)
    tr_v=embed(mid,[r["text"] for r in tr],revision=rev)
    an_v=embed(mid,a_txt,revision=rev)
    y_tr=[r["tier"] for r in tr]; y_ho=[r["tier"] for r in ho]
    hard=[r.get("hard") for r in ho]

    # A. shipped rule
    pa,_=anchor_topk_predict(ho_v,an_v,a_lab,labels,tax.get("top_k",3))
    print(fmt(sig,"A anchor-topk-mean (shipped)",metrics(y_ho,pa,labels,hard)))

    # B. logistic head on frozen embeddings
    Xtr=unit(tr_v.astype(np.float64)); Xho=unit(ho_v.astype(np.float64))
    clf=LogisticRegression(max_iter=4000,C=10.0,class_weight="balanced").fit(Xtr,y_tr)
    print(fmt(sig,"B logistic head (frozen emb)",metrics(y_ho,list(clf.predict(Xho)),labels,hard)))

    # C. learned centroids, scored the same top-k way (k=1 == nearest centroid)
    cent=np.stack([unit(Xtr[[i for i,y in enumerate(y_tr) if y==lab]].mean(0,keepdims=True))[0] for lab in labels])
    pc=[labels[i] for i in (Xho@cent.T).argmax(1)]
    print(fmt(sig,"C learned centroids",metrics(y_ho,pc,labels,hard)))
