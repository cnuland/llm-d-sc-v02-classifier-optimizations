import sys; sys.path.insert(0,"/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *
tax=load_taxonomy("cost"); rows=heldout("cost"); labels=tax["labels"]
a_txt=[t for lab in labels for t in tax["anchors"][lab]]
a_lab=[lab for lab in labels for _ in tax["anchors"][lab]]
for name,mid,rev in [("shipped cost.json (baseline MiniLM)",tax["model_repo"],tax.get("model_revision")),
                     ("retrained cnuland/llm-d-sc-cost","cnuland/llm-d-sc-cost",None)]:
    qv=embed(mid,[r["text"] for r in rows],revision=rev); av=embed(mid,a_txt,revision=rev)
    pred,_=anchor_topk_predict(qv,av,a_lab,labels,tax.get("top_k",3))
    m=metrics([r["tier"] for r in rows],pred,labels,[r.get("hard") for r in rows])
    print(fmt("cost",name,m))
