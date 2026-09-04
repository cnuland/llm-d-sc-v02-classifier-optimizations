"""Does the sharpened MEDIUM/COMPLEX rule raise LABELLER agreement?

§53 showed complexity is at its taxonomy ceiling: neither more data nor cleaner
labels moves it, and MEDIUM/COMPLEX draws 2.2x more juror disagreement than its
frequency predicts. The v1 wording asked labellers to imagine how other experts
would answer -- a counterfactual about people rather than an inspection of the
request.

v2 replaces it with a countable 2-of-3 test on features visibly present in the
text. If that works, two independent labellers should agree MORE often on the
same prompts. That is the thing to measure, and it is measurable BEFORE any
retraining: if agreement does not move, the rewrite is cosmetic and no amount of
relabelling will help.

Runs both rubrics over the SAME prompts with the SAME two labellers, so the only
variable is the wording.
"""
import sys, json, collections, pathlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl, load_taxonomy
import llmkit as L

LABELLERS = ["claude-opus-5", "claude-sonnet-5"]
SIG = sys.argv[2] if len(sys.argv) > 2 else "complexity"
labels = load_taxonomy(SIG)["labels"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600

# the prompts that actually exercise this boundary: rows where the two labellers
# split MEDIUM vs COMPLEX under v1
# §65's lesson: measuring only on contested rows gives the HARD-SUBSET effect and
# says nothing about the corpus. Sample both, and report them separately.
MODE = sys.argv[3] if len(sys.argv) > 3 else "contested"
import random
if MODE == "random":
    pool = load_jsonl(ROOT / f"data/train/{SIG}-real.jsonl") + \
           load_jsonl(ROOT / f"data/train/{SIG}-real-contested.jsonl")
    random.Random(7).shuffle(pool)
    rows = pool[:N]
else:
    rows = load_jsonl(ROOT / f"data/train/{SIG}-real-contested.jsonl")[:N]
print(f"[{SIG}/{MODE}] testing on {len(rows)} prompts")

def agreement(rubric_text, tag):
    votes = {}
    for m in LABELLERS:
        out = []
        chunks = [rows[i:i+20] for i in range(0, len(rows), 20)]
        def one(chunk):
            """Bisect around refusals and truncations, as blind_relabel does.

            Real traffic contains prompts a model will not engage with; one of
            them must not take down a 60-call A/B. Unlabelable items return None
            and are excluded from the agreement denominator rather than counted
            as a disagreement, which would bias the comparison.
            """
            listing = "\n".join(f"{i+1}. {r['text'][:900]}" for i, r in enumerate(chunk))
            p = (f"{rubric_text}\n\n---\n\nClassify each request into exactly one of: "
                 f"{', '.join(labels)}. Apply the rubric strictly, including its "
                 f"boundary rules.\n\n{listing}")
            try:
                items = L.ask_json(p, L.label_schema(labels, with_reason=False),
                                   model=m, effort="low", max_tokens=8000,
                                   seed_tag=tag)["items"]
                if len(items) == len(chunk):
                    return items
            except Exception:
                pass
            if len(chunk) == 1:
                return [{"label": None, "confidence": 0.0}]
            h = len(chunk) // 2
            return one(chunk[:h]) + one(chunk[h:])
        for part in L.amap(one, chunks, workers=12, desc=f"{tag}:{m[7:16]}"):
            out.extend(part)
        votes[m] = [x["label"] for x in out]
    a, b = votes[LABELLERS[0]], votes[LABELLERS[1]]
    ok = [i for i in range(len(rows)) if a[i] and b[i]]
    agree = sum(a[i] == b[i] for i in ok) / max(1, len(ok))
    return agree, len(ok), collections.Counter(f"{a[i]}/{b[i]}" for i in ok if a[i] != b[i])

v1 = (ROOT / f"rubrics/{SIG}-v1.md").read_text()
v2 = (ROOT / f"rubrics/{SIG}.md").read_text()
a1, n1, d1 = agreement(v1, "rubricv1")
a2, n2, d2 = agreement(v2, "rubricv2")
print(f"\n  v1 wording ('experts would differ'): agreement {a1:.1%}  (n={n1})")
print(f"  v2 rubric: agreement {a2:.1%}  (n={n2})")
print(f"  DELTA: {a2-a1:+.1%}")
print(f"\n  v1 residual disagreements: {d1.most_common(4)}")
print(f"  v2 residual disagreements: {d2.most_common(4)}")
