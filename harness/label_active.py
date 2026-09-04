"""Jury-label the actively selected prompts and fold them into training.

Same two-labeller protocol as the random slices, so the new rows are directly
comparable to the existing corpus: blind (no labeller sees a proposed label),
agreements become hard rows, disagreements are kept with both votes so they can
carry a soft target.

The disagreement rate here is itself informative. These prompts were chosen
BECAUSE the model could not separate them; if the human-proxy labellers also
disagree far more than usual, the boundary is genuinely ill-posed rather than
merely unlearned, and no amount of extra labelling will fix it.
"""
import json, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy
from sdg import blind_relabel

LABELLERS = ["claude-opus-5", "claude-sonnet-5"]
sig = sys.argv[1]
KIND = sys.argv[2] if len(sys.argv) > 2 else "active"   # "active" | "disagree"
labels = load_taxonomy(sig)["labels"]
rows = load_jsonl(ROOT / f"data/train/{sig}-{KIND}-pool.jsonl")
print(f"{sig}: labelling {len(rows)} {KIND}-selected prompts")

votes = {}
for m in LABELLERS:
    votes[m] = [x["label"] for x in
                blind_relabel(sig, labels, [r["text"] for r in rows],
                              model=m, batch=25, effort="low")]
    print(f"  {m}: done", flush=True)

hard, soft, unanswered = [], [], 0
for i, r in enumerate(rows):
    a, b = votes[LABELLERS[0]][i], votes[LABELLERS[1]][i]
    if a is None or b is None:
        unanswered += 1
        continue
    rec = {"text": r["text"], "tier": a, "votes": [a, b], "agree": a == b,
           "margin": r.get("margin"), "model_pred": r.get("model_pred"),
           "screen_pred": r.get("screen_pred"),
           "source": f"{KIND}-selected"}
    (hard if a == b else soft).append(rec)

n = len(hard) + len(soft)
agree = len(hard) / max(1, n)
print(f"  agreed {len(hard)} ({agree:.3f})  disagreed {len(soft)}  unanswered {unanswered}")
print(f"  tier mix (agreed): {dict(collections.Counter(r['tier'] for r in hard))}")

# How often was the model simply wrong on the prompts it was unsure about?
# How often was the model actually wrong on the prompts it was CONFIDENT about?
# This is the number that decides whether the acquisition function works: the
# cheap screen only flagged candidates, the strong jury adjudicates them.
allr = hard + soft
wrong = sum(1 for r in allr if r["model_pred"] != r["tier"])
print(f"  jury contradicts the model on {wrong}/{len(allr)} = "
      f"{wrong/max(1,len(allr)):.1%} of selected rows")
if any(r.get("screen_pred") for r in allr):
    sc = [r for r in allr if r.get("screen_pred")]
    sok = sum(1 for r in sc if r["screen_pred"] == r["tier"])
    print(f"  cheap screen agreed with the jury on {sok}/{len(sc)} = "
          f"{sok/max(1,len(sc)):.1%} (how trustworthy the screen's flags were)")

for name, rs in [(KIND, hard), (f"{KIND}-contested", soft)]:
    out = ROOT / f"data/train/{sig}-{name}.jsonl"
    with open(out, "w") as fh:
        for r in rs:
            fh.write(json.dumps(r) + "\n")
    print(f"  -> {out.name}  ({len(rs)} rows)")
