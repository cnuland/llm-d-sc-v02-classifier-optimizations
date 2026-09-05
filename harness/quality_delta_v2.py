"""Quality delta, length-controlled. The §123 redesign.

§122 measured a quality gap between a small and a large model by tier and §123
retracted it: the judge picked the longer answer in 70.2% of decided pairs, and
the large model writes 1.20-1.35x more. Every naive pairwise comparison in this
setting favours the expensive model for a reason that has nothing to do with
quality.

Three changes, each aimed at that specific failure:

  1. BOTH MODELS GET THE SAME LENGTH BUDGET, via an identical system prompt. This
     removes the degree of freedom the judge was exploiting. It is imposed on both
     equally so it cannot advantage either, and the resulting lengths are MEASURED
     and reported -- if they do not converge, the control did not work and the
     result is not usable.

  2. THE JUDGE IS ASKED A CORRECTNESS QUESTION, NOT A PREFERENCE QUESTION.
     "Does either answer contain an error, omission, or misunderstanding the other
     does not?" A preference prompt invites the judge to reward polish; an error
     prompt asks about content. TIE remains first-class.

  3. LENGTH-WIN-RATE IS REPORTED AS A BUILT-IN CONTROL. If the longer answer still
     wins far above 50%, the control failed and the numbers are void. §123 had to
     run that check separately and after the fact; here it ships with the result.

The judge is a third model, neither contestant. Model identities are configurable
and deliberately not reported: this measures whether ROUTING pays, not which
vendor model is better, and naming them would date the finding to specific
versions and read as a benchmark it is not.
"""
import sys, json, random, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
import llmkit as L
from transformers import AutoTokenizer, AutoModelForSequenceClassification

N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 100
import os
SMALL = os.environ.get("QD_SMALL", "claude-haiku-4-5")
LARGE = os.environ.get("QD_LARGE", "claude-sonnet-5")
JUDGE = os.environ.get("QD_JUDGE", "claude-fable-5-1")
BUDGET = ("Answer the user's request directly and completely. Keep your answer "
          "under 200 words unless the request explicitly asks for a longer "
          "artifact, in which case produce what was asked for and nothing more. "
          "Do not add preamble, restatement of the question, or closing offers of "
          "further help.")

rows = (load_jsonl(ROOT/"data/eval/triage-real-gold.jsonl")
        + load_jsonl(ROOT/"data/eval/triage-real-contested.jsonl"))
d = ROOT/"models/tri-at1-seed11"
tok = AutoTokenizer.from_pretrained(str(d))
m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
order = [m.config.id2label[i] for i in range(m.config.num_labels)]
pred = []
with torch.no_grad():
    for i in range(0, len(rows), 64):
        e = tok([r["text"] for r in rows[i:i+64]], truncation=True, max_length=256,
                padding=True, return_tensors="pt")
        pred += [order[j] for j in m(**e).logits.argmax(-1).tolist()]
by = collections.defaultdict(list)
for r, p in zip(rows, pred):
    t = r["text"].strip()
    if 20 <= len(t) <= 6000: by[p].append(t)
rng = random.Random(17)
sample = {k: rng.sample(v, min(N_PER, len(v))) for k, v in by.items()}
print("routed by the model: " + "  ".join(f"{k}={len(v)}" for k, v in sample.items()), flush=True)

def gen(a):
    t, mdl = a
    try: return L.ask_text(t, model=mdl, max_tokens=900, system=BUDGET, seed_tag="lenctl")
    except Exception: return None

S = {"type":"object","additionalProperties":False,"required":["verdict"],
     "properties":{"verdict":{"type":"string","enum":["A","B","TIE"]}}}
def judge(a):
    prompt, x, y, tag = a
    p = (f"Two assistants answered the same user request.\n\n"
         f"Does either answer contain an ERROR, OMISSION, or MISUNDERSTANDING that "
         f"the other does not? Answer A if A is more correct or complete for what "
         f"was actually asked, B if B is, and TIE if neither has a substantive "
         f"advantage.\n\n"
         f"Do NOT reward polish, tone, formatting or thoroughness beyond what was "
         f"asked. Two answers that would both leave the user equally well served "
         f"are a TIE regardless of which reads better.\n\n"
         f"=== REQUEST ===\n{prompt[:4000]}\n\n=== ANSWER A ===\n{x[:4000]}\n\n"
         f"=== ANSWER B ===\n{y[:4000]}\n")
    try: return L.ask_json(p, S, model=JUDGE, max_tokens=2000, seed_tag=tag)["verdict"]
    except Exception: return None

jobs = [(t, mdl) for tier in sample for t in sample[tier] for mdl in (SMALL, LARGE)]
print(f"generating {len(jobs)} length-budgeted answers...", flush=True)
outs = L.amap(gen, jobs, workers=14, desc="gen")
A = {k: v for k, v in zip(jobs, outs)}

jj, meta = [], []
for tier, texts in sample.items():
    for i, t in enumerate(texts):
        s, l = A.get((t, SMALL)), A.get((t, LARGE))
        if not s or not l: continue
        flip = rng.random() < 0.5
        jj.append((t, l if flip else s, s if flip else l, f"qdv2:{tier}:{i}"))
        meta.append((tier, flip, len(s.split()), len(l.split())))
print(f"judging {len(jj)} pairs...", flush=True)
V = L.amap(judge, jj, workers=10, desc="judge")

res = collections.defaultdict(collections.Counter); lens = collections.defaultdict(list)
longer_won = collections.Counter()
for (tier, flip, ns, nl), v in zip(meta, V):
    lens[tier].append((ns, nl))
    if v is None: continue
    if v == "TIE": res[tier]["tie"] += 1; continue
    large_won = (v == "A") == flip
    res[tier]["large" if large_won else "small"] += 1
    longer_won["yes" if (nl if large_won else ns) > (ns if large_won else nl) else "no"] += 1

print("\n  CONTROL 1 — did the length budget work?")
print(f"  {'tier':<10}{'small words':>13}{'large words':>13}{'ratio':>9}   (§122 was 1.35x / 1.20x)")
for tier in ("TRIVIAL","WORK"):
    if not lens[tier]: continue
    a=np.median([x[0] for x in lens[tier]]); b=np.median([x[1] for x in lens[tier]])
    print(f"  {tier:<10}{a:13.0f}{b:13.0f}{b/max(1,a):8.2f}x")
tot=sum(longer_won.values())
print(f"\n  CONTROL 2 — longer answer won {longer_won['yes']}/{tot} = "
      f"{longer_won['yes']/max(1,tot):.1%}  (§122 judge: 70.2%; 50% = length-blind)")
print(f"\n  {'routed as':<12}{'n':>5}{'large':>9}{'tie':>9}{'small':>9}{'large net':>11}")
for tier in ("TRIVIAL","WORK"):
    c=res[tier]; n=sum(c.values())
    if n: print(f"  {tier:<12}{n:5d}{c['large']/n:8.1%}{c['tie']/n:9.1%}{c['small']/n:9.1%}"
                f"{(c['large']-c['small'])/n:+10.1%}")
print(f"\n  uncontrolled runs for comparison: opus +37.4/+15.4, fable +33.0/+33.0")
