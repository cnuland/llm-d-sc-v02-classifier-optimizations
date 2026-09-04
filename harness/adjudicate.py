"""How many of the residual errors are actually the MODEL's fault?

The gold labels are unanimous three-model jury verdicts, but they were produced
by a fast pass over batches of 20. The residual errors are, by construction, the
hardest prompts in the set — exactly where a batched judgement is most likely to
be sloppy. Several of them look defensible on inspection.

So re-adjudicate, blind and paired: a strong judge sees the prompt and TWO
candidate labels in randomised order, with no indication of which came from the
jury and which from the model, and picks the better one under the rubric. It may
also answer NEITHER.

Blind pairing matters. Asking "is this gold label right?" invites ratification;
asking "which of these two is better?" with the provenance hidden does not.

The output is a ceiling estimate: if the judge prefers the model on a share of
these, the eval's effective maximum is below 1.0 and the remaining headroom is
smaller than the raw accuracy gap suggests.
"""
import sys, json, random, collections, torch
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy
import llmkit as L
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sig = sys.argv[1] if len(sys.argv) > 1 else "complexity"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "cx-b4-soft-512-mbert"
JUDGE = "claude-opus-5"
labels = load_taxonomy(sig)["labels"]
rows = load_jsonl(ROOT / f"data/eval/{sig}-real-gold.jsonl")

tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / MODEL))
m = AutoModelForSequenceClassification.from_pretrained(str(ROOT / "models" / MODEL)).eval()
preds = []
with torch.no_grad():
    for i in range(0, len(rows), 32):
        b = tok([r["text"] for r in rows[i:i+32]], truncation=True, max_length=256,
                padding=True, return_tensors="pt")
        preds += [labels[j] for j in m(**b).logits.argmax(-1).tolist()]
err = [(r, p) for r, p in zip(rows, preds) if r["tier"] != p]
print(f"{sig}: {len(err)} residual errors out of {len(rows)}")

rnd = random.Random(20260903)
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["n", "choice", "why"],
              "properties": {"n": {"type": "integer"},
                             "choice": {"type": "string", "enum": ["A", "B", "NEITHER"]},
                             "why": {"type": "string"}}}}}}

def judge(chunk):
    lines = []
    for i, (r, p, A, B) in enumerate(chunk, 1):
        lines.append(f"{i}. REQUEST: {r['text'][:1200]}\n   A = {A}\n   B = {B}")
    prompt = (f"{L.rubric(sig)}\n\n---\n\nFor each request below, two candidate tier "
              f"labels are given. Decide which is the better label under the rubric, "
              f"applying its boundary rules strictly. Answer A, B, or NEITHER if both "
              f"are wrong. Give a reason under 15 words.\n\n" + "\n\n".join(lines))
    return L.ask_json(prompt, SCHEMA, model=JUDGE, effort="high", max_tokens=16000)["items"]

cases = []
for r, p in err:
    flip = rnd.random() < 0.5          # hide provenance
    cases.append((r, p, p if flip else r["tier"], r["tier"] if flip else p))
chunks = [cases[i:i+10] for i in range(0, len(cases), 10)]
out = [x for part in L.amap(judge, chunks, workers=6, desc="adjudicate") for x in part]

tally = collections.Counter()
detail = []
for (r, p, A, B), o in zip(cases, out):
    picked = A if o["choice"] == "A" else (B if o["choice"] == "B" else None)
    if picked is None:
        verdict = "neither"
    elif picked == r["tier"]:
        verdict = "gold"
    else:
        verdict = "model"
    tally[verdict] += 1
    detail.append({"text": r["text"][:400], "gold": r["tier"], "model": p,
                   "verdict": verdict, "why": o.get("why", "")})

n = sum(tally.values())
print(f"\nblind adjudication of the {n} residual errors:")
for k in ("gold", "model", "neither"):
    print(f"  judge prefers {k:<8} {tally[k]:>3}  ({tally[k]/n:.1%})")
ceiling = 1 - (tally["gold"] / len(rows))
print(f"\n  effective eval ceiling for this set: {ceiling:.4f}")
print(f"  (errors the judge blames on the model: {tally['gold']}/{len(rows)} of the eval)")
print("\n  examples where the judge preferred the MODEL over gold:")
for d in [d for d in detail if d["verdict"] == "model"][:6]:
    print(f"    gold={d['gold']:<10} model={d['model']:<10} {d['why'][:60]}")
    print(f"        {d['text'][:110]}")
json.dump(detail, open(ROOT / f"reports/adjudication-{sig}.json", "w"), indent=1)
