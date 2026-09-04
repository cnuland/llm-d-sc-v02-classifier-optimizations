"""Resolve contested TRAINING rows with a third juror.

§51/§52 established that more data pays only below ~80% juror agreement.
Complexity sits at 87.5%, so its corpus scaling bought nothing measurable
(+0.26 against a 1.06 floor). The rule's own prescription for a high-agreement
signal is to spend on LABEL QUALITY instead of volume.

Complexity has 7,435 rows where its two labellers disagreed. They currently
train as soft targets — a 50/50 split between two tiers — which is honest but
uninformative. A third independent juror breaks most of those ties: where it
sides with one of the two, the row becomes a 2-of-3 majority and can train as a
hard label; where it picks a third tier entirely, the row is genuinely ill-posed
and is dropped rather than guessed at.

This converts noise into signal on exactly the rows that carry the most
information, and it is the one lever the label-noise rule endorses for a
high-agreement signal.
"""
import json, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy
from sdg import blind_relabel

sig = sys.argv[1] if len(sys.argv) > 1 else "complexity"
THIRD = "claude-fable-5-1"          # not one of the original two labellers
labels = load_taxonomy(sig)["labels"]
rows = load_jsonl(ROOT / f"data/train/{sig}-real-contested.jsonl")
print(f"{sig}: breaking ties on {len(rows)} contested rows with an independent third juror")

third = [x["label"] for x in blind_relabel(sig, labels, [r["text"] for r in rows],
                                           model=THIRD, batch=25, effort="low")]
resolved, unresolved, unanswered = [], 0, 0
for r, t in zip(rows, third):
    votes = r.get("votes") or []
    if t is None:
        unanswered += 1
        continue
    if t in votes:                       # third juror sides with one of the two
        resolved.append({"text": r["text"], "tier": t, "votes": votes + [t],
                         "source": "tie-resolved"})
    else:                                # a third tier entirely: genuinely ill-posed
        unresolved += 1

out = ROOT / f"data/train/{sig}-resolved.jsonl"
with open(out, "w") as fh:
    for r in resolved:
        fh.write(json.dumps(r) + "\n")
n = len(rows) - unanswered
print(f"  resolved to a 2-of-3 majority : {len(resolved)}  ({len(resolved)/max(1,n):.1%})")
print(f"  third juror picked a third tier: {unresolved}  ({unresolved/max(1,n):.1%}) — dropped as ill-posed")
print(f"  unanswered                     : {unanswered}")
print(f"  tier mix: {dict(collections.Counter(r['tier'] for r in resolved))}")
print(f"  -> {out}")
