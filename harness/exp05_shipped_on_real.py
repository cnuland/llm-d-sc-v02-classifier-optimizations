"""EXPERIMENT 05 -- what do the SHIPPED classifiers score on real traffic?

Everything else is measured against this. If the shipped models already do well
on WildChat, the accuracy problem is smaller than it looks; if they collapse,
the hand-authored held-out numbers have been flattering them.
"""
import sys, json; sys.path.insert(0,"/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *
import collections

ARMS = {"complexity":[("shipped","cnuland/llm-d-sc-complexity",None),
                      ("baseline-minilm","sentence-transformers/all-MiniLM-L6-v2",None)],
        "cost":[("shipped cost.json","sentence-transformers/all-MiniLM-L6-v2",None),
                ("retrained","cnuland/llm-d-sc-cost",None)],
        "sensitivity":[("shipped","cnuland/llm-d-sc-sensitivity",None),
                       ("baseline-minilm","sentence-transformers/all-MiniLM-L6-v2",None)]}

for sig in SIGNALS:
    tax=load_taxonomy(sig); labels=tax["labels"]
    a_txt=[t for l in labels for t in tax["anchors"][l]]
    a_lab=[l for l in labels for _ in tax["anchors"][l]]
    sets=[("heldout-v1",heldout(sig))]
    for nm,f in [("real-gold",f"{sig}-real-gold"),("real-contested",f"{sig}-real-contested")]:
        p=ROOT/f"data/eval/{f}.jsonl"
        if p.exists(): sets.append((nm,load_jsonl(p)))
    print(f"\n########## {sig} ##########")
    for nm,rows in sets:
        print(f"  [{nm}] n={len(rows)}  prior={dict(collections.Counter(r['tier'] for r in rows))}")
    for arm,mid,rev in ARMS[sig]:
        av=embed(mid,a_txt,revision=rev)
        line=f"    {arm:<20}"
        for nm,rows in sets:
            qv=embed(mid,[r["text"] for r in rows],revision=rev)
            pred,_=anchor_topk_predict(qv,av,a_lab,labels,tax.get("top_k",3))
            m=metrics([r["tier"] for r in rows],pred,labels)
            line+=f"  {nm}={m['accuracy']:.3f}/F1={m['macro_f1']:.3f}"
        print(line)
