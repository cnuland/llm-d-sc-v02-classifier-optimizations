"""Label a large slice of real WildChat traffic for TRAINING.

Experiment 05 showed the shipped models drop 25-44 points from hand-authored
held-out prompts to real ones. Two causes: register (real prompts are messy
pastes and fragments) and prior (training is balanced, traffic is not).
Synthetic data written in a real "voice" narrows the register gap; training on
actual traffic closes it.

Disjointness is enforced by construction: the eval sample was drawn from the
FIRST `EVAL_SCREEN` prompts of the seed-20260903 shuffle, so training draws from
strictly after that point, and every row is re-checked against the eval keys
before it is written.

Two labellers, not three. A 2/2 agreement is a weaker guarantee than the 3/3
used for the eval, which is the right trade: training tolerates some label noise,
evaluation does not, and this buys roughly twice the volume.
"""
import json, pathlib, random, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, SIGNALS, load_taxonomy, load_jsonl
import llmkit as L
from sdg import blind_relabel, key

LABELLERS = ["claude-opus-5", "claude-sonnet-5"]
EVAL_SCREEN = 6000                       # must match build_eval_real.py
N = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
ONLY = sys.argv[2] if len(sys.argv) > 2 else None

pool = load_jsonl(ROOT/"data/eval/wildchat_pool.jsonl")
random.Random(20260903).shuffle(pool)
train_pool = pool[EVAL_SCREEN:EVAL_SCREEN + N]
print(f"training slice: {len(train_pool)} prompts, drawn after the eval screen")

for sig in ([ONLY] if ONLY else SIGNALS):
    labels = load_taxonomy(sig)["labels"]
    evalkeys = set()
    for f in [f"{sig}-real-gold", f"{sig}-real-contested"]:
        p = ROOT/f"data/eval/{f}.jsonl"
        if p.exists():
            evalkeys |= {key(r["text"]) for r in load_jsonl(p)}
    rows = [r for r in train_pool if key(r["text"]) not in evalkeys]
    print(f"\n########## {sig}: labelling {len(rows)} real prompts ##########")

    contested = []
    votes = {}
    for m in LABELLERS:
        votes[m] = [x["label"] for x in
                    blind_relabel(sig, labels, [r["text"] for r in rows],
                                  model=m, batch=25, effort="low")]
        print(f"  {m}: done", flush=True)

    # Keep EVERY row and record both votes, not just the agreements.
    #
    # Dropping disagreements throws away exactly the boundary cases where all the
    # errors live -- 2,359 rows for complexity, and the confusion matrices put
    # almost all their mass on those same boundaries. With both votes retained a
    # row can carry a SOFT target (a distribution over the labellers' votes)
    # instead of being discarded, and mixing hard and soft targets is better than
    # either alone (arXiv 2605.26246).
    kept, disagreed, unanswered = [], 0, 0
    for i, r in enumerate(rows):
        a, b = votes[LABELLERS[0]][i], votes[LABELLERS[1]][i]
        if a is None or b is None:
            unanswered += 1
            continue
        rec = {"text": r["text"], "tier": a, "votes": [a, b],
               "agree": a == b,
               "source": "wildchat-labelled" if a == b else "wildchat-contested"}
        if a == b:
            kept.append(rec)
        else:
            disagreed += 1
            rec["tier"] = a          # nominal; consumers should use `votes`
            contested.append(rec)
    agree = len(kept) / max(1, len(rows) - unanswered)
    print(f"  agreed {len(kept)} ({agree:.3f})  disagreed {disagreed}  unanswered {unanswered}")
    print(f"  tier mix: {dict(collections.Counter(r['tier'] for r in kept))}")
    out = ROOT/f"data/train/{sig}-real.jsonl"
    with open(out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    outc = ROOT/f"data/train/{sig}-real-contested.jsonl"
    with open(outc, "w") as fh:
        for r in contested:
            fh.write(json.dumps(r) + "\n")
    print(f"  -> {out}  (+{len(contested)} contested -> {outc.name})")
