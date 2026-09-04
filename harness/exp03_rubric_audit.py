"""EXPERIMENT 03 -- do the rubrics agree with the existing held-out gold labels?

Blind-label every held-out row from text + rubric alone (the judge never sees the
gold label). Three outcomes matter:
  high agreement          -> rubric and eval encode the same task; trust both
  systematic disagreement -> one is wrong; the DIRECTION says which
  scattered disagreement  -> the boundary is genuinely ill-defined

Runs before any training, because it decides whether the target is real.
"""
import sys, json; sys.path.insert(0,"/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *
import llmkit as L

JUDGE, BATCH = "claude-opus-5", 20
HDR = "truth\\LLM"

def label_batch(a):
    sig, labels, chunk = a
    listing = "\n".join(f"{i+1}. {r['text']}" for i, r in enumerate(chunk))
    p = (f"{L.rubric(sig)}\n\n---\n\nClassify each request below into exactly one of: "
         f"{', '.join(labels)}.\nApply the rubric strictly, including its boundary "
         f"rules. Give a confidence in [0,1] and a reason under 12 words.\n\n{listing}")
    return L.ask_json(p, L.label_schema(labels), model=JUDGE,
                      effort="medium", max_tokens=8000)["items"]

for sig in SIGNALS:
    labels = load_taxonomy(sig)["labels"]; rows = heldout(sig)
    chunks = [rows[i:i+BATCH] for i in range(0, len(rows), BATCH)]
    res = L.amap(label_batch, [(sig, labels, c) for c in chunks], workers=6)
    pred = [o["label"] for part in res for o in part]
    conf = [o["confidence"] for part in res for o in part]
    why  = [o.get("why","") for part in res for o in part]
    gold = [r["tier"] for r in rows]
    assert len(pred) == len(gold), f"{sig}: got {len(pred)} labels for {len(gold)} rows"
    m = metrics(gold, pred, labels, [r.get("hard") for r in rows])
    print(f"\n=== {sig}: rubric-vs-gold agreement {m['accuracy']:.4f}  n={m['n']} ===")
    print(f"{HDR:>13}" + "".join(f"{l[:9]:>11}" for l in labels))
    for t in labels:
        print(f"{t:>13}" + "".join(f"{m['confusion'][t][p]:>11}" for p in labels))
    dis = [(g,p,c,w,r['text']) for g,p,c,w,r in zip(gold,pred,conf,why,rows) if g!=p]
    mc = sum(d[2] for d in dis)/len(dis) if dis else 0
    print(f"  {len(dis)} disagreements, mean judge confidence on them {mc:.2f}")
    for g,p,c,w,t in dis[:10]:
        print(f"    gold={g:<13} llm={p:<13} {c:.2f}  {t[:62]}\n        why: {w[:70]}")
    json.dump([{"text":r["text"],"gold":g,"llm":p,"conf":c,"why":w}
               for r,g,p,c,w in zip(rows,gold,pred,conf,why)],
              open(f"data/eval/rubric_audit_{sig}.json","w"), indent=1)
