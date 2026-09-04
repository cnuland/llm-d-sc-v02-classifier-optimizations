"""Estimate gold-label error rate over the WHOLE eval, not just the errors.

Adjudicating only the model's mistakes and then fixing the gold labels it
disputes would be fitting the eval to the model -- pure circularity, and it can
only ever move the number up.

The honest version samples BOTH strata:

  * rows the model got WRONG   -> some gold labels are wrong, inflating the
                                  apparent error count (measured: 16.7%)
  * rows the model got RIGHT   -> if gold is wrong here too, the model AGREED
                                  with a wrong label and accuracy is OVERSTATED

Only with both can the eval's true ceiling be estimated rather than assumed, and
only the second stratum can move the number DOWN -- which is why it has to be
checked. Same blind paired protocol: the judge sees two candidate labels in
random order with no provenance, and may answer NEITHER.

For the correct stratum the alternative label is the model's second choice, so
the comparison is still a real one rather than a strawman.
"""
import sys, json, random, collections, torch
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy
import llmkit as L
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sig = sys.argv[1] if len(sys.argv) > 1 else "complexity"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "cx-b4-soft-512-mbert"
N_CORRECT = int(sys.argv[3]) if len(sys.argv) > 3 else 120
JUDGE = "claude-opus-5"
labels = load_taxonomy(sig)["labels"]
rows = load_jsonl(ROOT / f"data/eval/{sig}-real-gold.jsonl")

tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / MODEL))
m = AutoModelForSequenceClassification.from_pretrained(str(ROOT / "models" / MODEL)).eval()
top1, top2 = [], []
with torch.no_grad():
    for i in range(0, len(rows), 32):
        b = tok([r["text"] for r in rows[i:i+32]], truncation=True, max_length=256,
                padding=True, return_tensors="pt")
        lg = m(**b).logits
        o = lg.argsort(-1, descending=True)
        top1 += [labels[j] for j in o[:, 0].tolist()]
        top2 += [labels[j] for j in o[:, 1].tolist()]

correct = [(r, t2) for r, t1, t2 in zip(rows, top1, top2) if r["tier"] == t1]
rnd = random.Random(20260903); rnd.shuffle(correct)
sample = correct[:N_CORRECT]
print(f"{sig}: auditing {len(sample)} rows the model got RIGHT "
      f"(of {len(correct)}); alternative = the model's 2nd choice")

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["n", "choice", "why"],
              "properties": {"n": {"type": "integer"},
                             "choice": {"type": "string", "enum": ["A", "B", "NEITHER"]},
                             "why": {"type": "string"}}}}}}

def judge(chunk):
    lines = [f"{i}. REQUEST: {r['text'][:1200]}\n   A = {A}\n   B = {B}"
             for i, (r, _alt, A, B) in enumerate(chunk, 1)]
    p = (f"{L.rubric(sig)}\n\n---\n\nFor each request, two candidate tier labels are "
         f"given. Pick the better one under the rubric, applying its boundary rules "
         f"strictly. Answer A, B, or NEITHER if both are wrong. Reason under 15 "
         f"words.\n\n" + "\n\n".join(lines))
    return L.ask_json(p, SCHEMA, model=JUDGE, effort="high", max_tokens=16000)["items"]

cases = []
for r, alt in sample:
    flip = rnd.random() < 0.5
    cases.append((r, alt, alt if flip else r["tier"], r["tier"] if flip else alt))
chunks = [cases[i:i+10] for i in range(0, len(cases), 10)]
out = [x for part in L.amap(judge, chunks, workers=6, desc="audit") for x in part]

tally = collections.Counter()
for (r, alt, A, B), o in zip(cases, out):
    picked = A if o["choice"] == "A" else (B if o["choice"] == "B" else None)
    tally["neither" if picked is None else
          ("gold" if picked == r["tier"] else "alternative")] += 1
n = sum(tally.values())
print(f"\non rows the model got RIGHT:")
for k in ("gold", "alternative", "neither"):
    print(f"  judge prefers {k:<12} {tally[k]:>3}  ({tally[k]/n:.1%})")
bad = (tally["alternative"] + tally["neither"]) / n
print(f"\n  estimated gold error rate on CORRECT rows: {bad:.1%}")
print(f"  -> accuracy is OVERSTATED by roughly {bad*len(correct)/len(rows):.1%} "
      f"of the eval ({bad*len(correct):.0f} of {len(rows)} rows)")
json.dump({"stratum": "correct", "n": n, "tally": dict(tally)},
          open(ROOT / f"reports/gold-audit-{sig}.json", "w"), indent=1)
