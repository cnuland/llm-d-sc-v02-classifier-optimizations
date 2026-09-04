"""Is the ceiling INTRINSIC ambiguity, or disagreement BETWEEN models?

§69's ceilings and §74's tracking result both rest on inter-juror agreement, and
all three jurors are Claude models. That leaves a question the project has never
asked: when the SAME juror labels the SAME prompt twice, does it agree with
itself?

  self-agreement ~= inter-agreement   the prompt is genuinely ambiguous. The
        ceiling is a property of the task and no better labelling process moves
        it.
  self-agreement >> inter-agreement   each model is internally consistent but
        they systematically differ. That is not irreducible ambiguity, it is
        model-specific bias -- and it is attackable by rubric work, by juror
        selection, or by treating jurors as annotators to be modelled (per-juror
        heads) rather than averaged.

This distinguishes the two readings for the first time, and it decides whether
"high 90s is unreachable" is a fact about the task or a fact about the panel.

Cheap: same prompts, one model, three independent passes with distinct cache tags.
"""
import sys, json, collections, itertools
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy
import llmkit as L
from sdg import blind_relabel

SIG = sys.argv[1] if len(sys.argv) > 1 else "complexity"
N   = int(sys.argv[2]) if len(sys.argv) > 2 else 400
MODEL = sys.argv[3] if len(sys.argv) > 3 else "claude-sonnet-5"
labels = load_taxonomy(SIG)["labels"]

evs = [f"{SIG}-real-gold.jsonl", f"{SIG}-real-contested.jsonl"]
# BUG FIXED: taking the first N rows of gold+contested took only GOLD rows,
# which are unanimous BY CONSTRUCTION -- the first run reported inter-juror
# agreement of exactly 100.0%, which is a definition, not a measurement.
# Sample each stratum separately and report them separately.
import random as _r
_pool = {f: [r for r in load_jsonl(ROOT/"data/eval"/f) if r.get("votes")]
         for f in evs if (ROOT/"data/eval"/f).exists()}
for _f, _v in _pool.items(): _r.Random(5).shuffle(_v)
rows, strata = [], []
for _f, _v in _pool.items():
    take = _v[:N//len(_pool)]
    rows += take; strata += ["unanimous" if "contested" not in _f else "contested"]*len(take)
texts = [r["text"] for r in rows]
print(f"{SIG}: {len(rows)} prompts (gold + contested, so the hard cases are included)")

# the panel's own agreement on exactly these rows, for a like-for-like baseline
inter = sum(1 for r in rows if len(set(r["votes"])) == 1) / len(rows)
import collections as _c
print(f"  strata: {dict(_c.Counter(strata))}")
print(f"  inter-juror agreement over the sample: {inter:.1%}")

passes = []
for k in range(3):
    out = blind_relabel(SIG, labels, texts, model=MODEL, batch=20, effort="low",
                        pass_tag=f"selfconsistency-{k}")
    passes.append([x["label"] for x in out])
    print(f"  pass {k+1} done", flush=True)

ok = [i for i in range(len(rows)) if all(p[i] for p in passes)]
self_agree = sum(1 for i in ok if len({p[i] for p in passes}) == 1) / max(1, len(ok))
print(f"\n  SELF-agreement of {MODEL} across 3 passes: {self_agree:.1%}  (n={len(ok)})")
print(f"  inter-juror agreement on the same rows      : {inter:.1%}")
print(f"  DELTA: {self_agree-inter:+.1%}")

for st in ("unanimous", "contested"):
    ix = [i for i in ok if strata[i] == st]
    if not ix: continue
    a = sum(1 for i in ix if len({p[i] for p in passes}) == 1) / len(ix)
    print(f"  self-agreement on {st:<10} rows (n={len(ix):3d}): {a:.1%}")
d = collections.Counter()
for i in ok:
    ss = sorted({p[i] for p in passes})
    if len(ss) > 1: d["/".join(ss)] += 1
print(f"  self-disagreements: {d.most_common(5)}")
