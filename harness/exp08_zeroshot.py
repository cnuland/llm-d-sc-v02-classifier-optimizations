"""EXPERIMENT 08 -- zero-shot NLI as a no-training reference point.

Every trained arm so far inherits the register of whatever wrote its training
data, which is precisely what costs 25-44 points on real traffic. A zero-shot
NLI model has never seen this task or this generator, so it has no register to
inherit. If it beats the fine-tuned models on real traffic, the gap is dominated
by training-data distribution rather than by model capacity.

Scoring is standard NLI entailment: premise = the prompt, hypothesis = a natural
sentence stating the tier, prediction = argmax entailment probability.
"""
import sys, torch; sys.path.insert(0,"/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *

MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
HYP = {
 "complexity": {
   "SIMPLE":"This request asks for a single fact or a one-step mechanical answer.",
   "MEDIUM":"This request asks for one routine deliverable that has a conventional right answer.",
   "COMPLEX":"This request asks for open-ended system design weighing competing tradeoffs.",
   "REASONING":"This request requires a step-by-step derivation, proof, or logical deduction."},
 "cost": {
   "MINIMAL":"Answering this needs only a phrase or a single sentence.",
   "LOW":"Answering this needs a short paragraph or a small snippet.",
   "MODERATE":"Answering this needs one substantial document or a complete module.",
   "HIGH":"Answering this needs processing a large body of material or exhaustive coverage."},
 "sensitivity": {
   "PUBLIC":"This contains only general public knowledge and nothing organisation specific.",
   "INTERNAL":"This contains internal company operational detail that is not commercially sensitive.",
   "CONFIDENTIAL":"This contains commercially sensitive business information such as finances, strategy, or proprietary code.",
   "REGULATED":"This contains personal, medical, or financial data about identifiable people that is governed by law.",
   "NEVER_EGRESS":"This contains live credentials, secret keys, or legally privileged material."},
}
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
dv="mps" if torch.backends.mps.is_available() else "cpu"; m.to(dv)
ENT=[i for i,l in m.config.id2label.items() if l.lower().startswith("entail")][0]

def predict(texts, labels, sig, bs=16):
    hyps=[HYP[sig][l] for l in labels]; out=[]
    with torch.no_grad():
        for i in range(0,len(texts),bs):
            chunk=texts[i:i+bs]
            prem=[t for t in chunk for _ in hyps]; hy=[h for _ in chunk for h in hyps]
            b=tok(prem,hy,truncation=True,max_length=256,padding=True,return_tensors="pt").to(dv)
            lg=m(**b).logits.softmax(-1)[:,ENT].view(len(chunk),len(hyps))
            out+= [labels[j] for j in lg.argmax(1).tolist()]
    return out

for sig in SIGNALS:
    labels=load_taxonomy(sig)["labels"]
    sets=[("heldout-v1",heldout(sig))]
    for nm,f in [("real-gold",f"{sig}-real-gold")]:
        p=ROOT/f"data/eval/{f}.jsonl"
        if p.exists(): sets.append((nm,load_jsonl(p)))
    for nm,rows in sets:
        mm=metrics([r["tier"] for r in rows],
                   predict([r["text"] for r in rows],labels,sig),labels)
        print(fmt(sig,f"zeroshot-nli:{nm}",mm))
