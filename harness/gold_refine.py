"""Re-adjudicate an ENTIRE eval set at high effort, model-blind.

The current gold was produced by a fast pass: batches of 20 prompts, medium
reasoning effort, three jurors, keep unanimous. Two-stratum blind adjudication
later measured 4.9-6.5% of those labels to be wrong, and inspection showed the
failure mode -- the jury sometimes did not apply the rubric it was given
(`"List of 60 prompts..."` labelled MODERATE when the rubric's own bulk rule
makes it HIGH).

That noise is the binding constraint on every accuracy figure in this project:
a perfect classifier caps at 0.9514 on complexity and 0.9350 on cost. So refine
the labels themselves.

Three rules keep this honest rather than circular:

  * MODEL-BLIND. No model prediction is shown, and the whole set is
    re-adjudicated -- not just rows some model got wrong. Re-labelling only
    disputed rows would fit the eval to the model and could only move accuracy
    up.
  * HIGHER EFFORT THAN THE ORIGINAL, not merely a repeat: batches of 5 instead
    of 20, `effort="high"`, and each juror must cite the rubric clause it relied
    on, which makes unreasoned agreement harder.
  * PROVENANCE KEPT. The original label is retained alongside the new one so
    every change is auditable and the refinement can be reverted or re-checked.

Rows where the refined jury is not unanimous are moved OUT of gold, because a
prompt three careful jurors cannot agree on is not a fair test.
"""
import sys, json, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy
import llmkit as L

sig = sys.argv[1]
JURY = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"]
labels = load_taxonomy(sig)["labels"]
src = ROOT / f"data/eval/{sig}-real-gold.jsonl"
rows = load_jsonl(src)
print(f"{sig}: re-adjudicating {len(rows)} gold rows at high effort, model-blind")

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["n", "label", "clause", "confidence"],
              "properties": {
                  "n": {"type": "integer"},
                  "label": {"type": "string", "enum": labels},
                  "clause": {"type": "string",
                             "description": "the rubric rule this decision rests on"},
                  "confidence": {"type": "number"}}}}}}

def label_chunk(args):
    model, chunk = args
    listing = "\n\n".join(f"{i+1}. {t[:1500]}" for i, t in enumerate(chunk))
    p = (f"{L.rubric(sig)}\n\n---\n\nClassify each request into exactly one of: "
         f"{', '.join(labels)}.\n\nWork carefully. For each one, name the specific "
         f"rubric rule or boundary clause your decision rests on — if you cannot "
         f"point to a clause, reconsider. Then give a confidence in [0,1].\n\n"
         f"{listing}")
    try:
        items = L.ask_json(p, SCHEMA, model=model, effort="high", max_tokens=16000)["items"]
        if len(items) == len(chunk):
            return items
    except Exception:
        pass
    # Bisect: real traffic contains prompts a model will refuse, and one of them
    # must not take down a 400-call adjudication. Unlabelable items come back
    # None and their rows are dropped from gold rather than guessed at.
    if len(chunk) == 1:
        return [{"label": None, "clause": "", "confidence": 0.0}]
    h = len(chunk) // 2
    return label_chunk((model, chunk[:h])) + label_chunk((model, chunk[h:]))

texts = [r["text"] for r in rows]
chunks = [texts[i:i+5] for i in range(0, len(texts), 5)]
votes = {}
for mdl in JURY:
    out = L.amap(label_chunk, [(mdl, c) for c in chunks], workers=8, desc=f"refine {mdl[7:18]}")
    v = [x for part in out for x in part]
    votes[mdl] = v if len(v) == len(texts) else None
    print(f"  {mdl}: {'ok' if votes[mdl] else 'LENGTH MISMATCH — dropped'}")
JURY = [m for m in JURY if votes[m]]

kept, dropped, changed = [], 0, 0
for i, r in enumerate(rows):
    vs = [votes[m][i]["label"] for m in JURY]
    if any(v is None for v in vs):
        dropped += 1
        continue
    cnt = collections.Counter(vs)
    top, n = cnt.most_common(1)[0]
    if n < len(JURY):
        dropped += 1
        continue
    rec = dict(r)
    rec["tier_v1"] = r["tier"]
    rec["tier"] = top
    rec["refined"] = True
    rec["clause"] = votes[JURY[0]][i].get("clause", "")[:180]
    if top != r["tier"]:
        changed += 1
    kept.append(rec)

out = ROOT / f"data/eval/{sig}-real-gold-v2.jsonl"
with open(out, "w") as fh:
    for r in kept:
        fh.write(json.dumps(r) + "\n")
print(f"\n  unanimous after refinement : {len(kept)}")
print(f"  dropped (no longer unanimous): {dropped}")
print(f"  LABEL CHANGED vs v1          : {changed}  ({changed/max(1,len(kept)):.1%} of kept)")
pairs = collections.Counter(f"{r['tier_v1']}->{r['tier']}"
                            for r in kept if r["tier"] != r["tier_v1"])
print(f"  change pairs: {pairs.most_common(6)}")
print(f"  tier mix: {dict(collections.Counter(r['tier'] for r in kept))}")
print(f"  -> {out}")
