"""Training labels from SELF-CONSISTENCY voting, not panel consensus.

§83 is the justification and §85 is the caveat.

§83: a single juror re-sampled agrees with itself 80.9% of the time while the
three-model panel agrees only 53.2% of the time on the same rows. The ceiling
every conclusion in this report leans on is a property of THIS PANEL, not of the
task. Intra-juror sampling noise is currently baked into the gold -- 10% of rows
the panel called unanimous are unanimous only by luck of the draw.

§85: swapping the EVAL to a self-consistency gold lowers measured accuracy by 2.7
points, because the model reproduces the panel it was trained on. So the eval
swap alone is not the experiment. The untested cell is training AND evaluating on
the denoised target, which is what this builds.

Majority-of-3 from one juror removes the sampling component of label noise while
leaving each rater's systematic view intact. If §83's reading is right, a model
trained against that coherent target should score materially higher on the
matching gold than the panel-trained model does on the panel gold -- and if it
does not, the noise was never the binding constraint.

Reported honestly either way: accuracy against a single coherent rater answers
"does the model reproduce this rater", which is a different question from "does
it reproduce the consensus". Both golds are scored in the round that follows.
"""
import sys, json, collections, random, pathlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy, load_jsonl
from sdg import blind_relabel, key

SIG   = sys.argv[1] if len(sys.argv) > 1 else "complexity"
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
MODEL = sys.argv[3] if len(sys.argv) > 3 else "claude-sonnet-5"
K     = 3
labels = load_taxonomy(SIG)["labels"]

src = ROOT/"data/train"/f"{SIG}-real-v2rubric.jsonl"
if not src.exists(): src = ROOT/"data/train"/f"{SIG}-real.jsonl"
pool = [json.loads(l)["text"] for l in open(src)]
# exclude anything in an eval set, by content hash
evk = set()
for f in (f"{SIG}-real-gold.jsonl", f"{SIG}-real-contested.jsonl",
          f"{SIG}-real-gold-v2.jsonl", f"{SIG}-selfconsistency-gold.jsonl"):
    p = ROOT/"data/eval"/f
    if p.exists(): evk |= {key(r["text"]) for r in load_jsonl(p)}
pool = list(dict.fromkeys(t for t in pool if key(t) not in evk))
random.Random(13).shuffle(pool)
pool = pool[:N]
print(f"{SIG}: {len(pool)} prompts from {src.name}, disjoint from every eval")

passes = []
for k in range(K):
    passes.append([x["label"] for x in blind_relabel(SIG, labels, pool, model=MODEL,
                   batch=20, effort="low", pass_tag=f"sctrain-{k}")])
    print(f"  pass {k+1}/{K} done", flush=True)

rows, nomaj, unstable = [], 0, 0
for i, t in enumerate(pool):
    v = [p[i] for p in passes if p[i]]
    if not v: continue
    c = collections.Counter(v).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        nomaj += 1; continue                      # genuinely unresolved: drop
    if len(set(v)) > 1: unstable += 1
    rows.append({"text": t, "tier": c[0][0], "votes": v, "source": "selfconsistency"})

print(f"  kept {len(rows)}; {nomaj} had no majority (dropped); "
      f"{unstable} were unstable but resolvable ({unstable/max(1,len(rows)):.1%})")
print(f"  self-agreement (all {K} passes identical): "
      f"{1 - unstable/max(1,len(rows)):.1%}")
print(f"  tier mix: {dict(collections.Counter(r['tier'] for r in rows))}")
out = ROOT/"data/train"/f"{SIG}-selfconsistency.jsonl"
with open(out, "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n")
print(f"  -> {out}")
