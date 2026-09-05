"""Does routing to a small model actually cost quality? The unmeasured term.

121 findings measure whether the classifier picks the right tier. NOTHING
measures whether picking the right tier is worth anything. A router pays off when
(cost saved routing easy->small) exceeds (quality lost routing hard->small), and
only the first term has ever been touched here.

Design choices that matter:

  ROUTED BY THE MODEL, NOT BY GOLD. Prompts are split by what the deployed triage
  classifier PREDICTS, not by their jury label. That is what the router would
  actually do, so misroutes are included in the measurement rather than excluded
  from it.

  BLIND PAIRWISE, POSITION RANDOMISED. The judge sees two unlabelled answers in
  random order and never learns which model produced either. Position is the
  best-documented bias in LLM judging and randomising it is the cheapest control.

  JUDGE IS NEITHER GENERATOR. Small = haiku-4-5, large = sonnet-5, judge = opus-5.
  A judge that is also a contestant prefers itself.

  TIES ARE A FIRST-CLASS OUTCOME. On trivial prompts the honest answer is usually
  "these are the same", and a forced choice would manufacture a quality gap that
  does not exist. That tie rate IS the result for the TRIVIAL tier.

The number that decides the product: if TRIVIAL prompts tie or the small model
wins, routing them down is free. If WORK prompts show a large gap, routing them up
is necessary. Both must hold for the router to be worth building.
"""
import sys, json, random, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl
import llmkit as L
from transformers import AutoTokenizer, AutoModelForSequenceClassification

N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 100
# Routing ROLES, not a model comparison. Override via env to run this against
# any small/large pair; the finding is about whether routing pays, and naming
# specific models would make it read as a vendor benchmark and date it.
import os
SMALL = os.environ.get("QD_SMALL", "claude-haiku-4-5")
LARGE = os.environ.get("QD_LARGE", "claude-sonnet-5")
JUDGE = os.environ.get("QD_JUDGE", "claude-opus-5")
MAXTOK = 1200

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
print(f"routed by the model: " + "  ".join(f"{k}={len(v)}" for k, v in sample.items()), flush=True)

def gen(a):
    txt, model = a
    try:
        return L.ask_text(txt, model=model, max_tokens=MAXTOK)
    except Exception as e:
        return None

JSCHEMA = {"type":"object","additionalProperties":False,"required":["verdict"],
  "properties":{"verdict":{"type":"string","enum":["A","B","TIE"]}}}

def judge(a):
    prompt, ans_a, ans_b, tag = a
    p = (f"Two assistants answered the same user request. Decide which answer is "
         f"better for the user, or whether they are equivalent.\n\n"
         f"Judge on: correctness, completeness for what was asked, and usefulness. "
         f"Ignore length, formatting and style unless they change usefulness. If "
         f"both are adequate and the difference would not matter to the user, "
         f"answer TIE -- TIE is the correct answer for most routine requests and "
         f"you should use it freely.\n\n"
         f"=== REQUEST ===\n{prompt[:4000]}\n\n"
         f"=== ANSWER A ===\n{ans_a[:4000]}\n\n=== ANSWER B ===\n{ans_b[:4000]}\n")
    try:
        return L.ask_json(p, JSCHEMA, model=JUDGE, max_tokens=2000, seed_tag=tag)["verdict"]
    except Exception:
        return None

jobs = [(t, mdl) for tier in sample for t in sample[tier] for mdl in (SMALL, LARGE)]
print(f"generating {len(jobs)} answers...", flush=True)
answers = L.amap(gen, jobs, workers=14, desc="gen")
A = {}
for (t, mdl), out in zip(jobs, answers): A[(t, mdl)] = out

results = collections.defaultdict(collections.Counter)
jjobs, meta = [], []
for tier, texts in sample.items():
    for i, t in enumerate(texts):
        s, l = A.get((t, SMALL)), A.get((t, LARGE))
        if not s or not l: continue
        flip = rng.random() < 0.5                       # position randomisation
        jjobs.append((t, l if flip else s, s if flip else l, f"qd:{tier}:{i}"))
        meta.append((tier, flip))
print(f"judging {len(jjobs)} pairs...", flush=True)
verdicts = L.amap(judge, jjobs, workers=10, desc="judge")
for (tier, flip), v in zip(meta, verdicts):
    if v is None: continue
    if v == "TIE": results[tier]["tie"] += 1
    else:
        large_won = (v == "A") == flip
        results[tier]["large" if large_won else "small"] += 1

print(f"\n  roles: SMALL vs LARGE, judged by a third model that is neither contestant.")
print(f"  (model identities intentionally not reported — this measures routing economics,\n   not a vendor comparison; set QD_SMALL/QD_LARGE/QD_JUDGE to use your own)")
print(f"\n  {'routed as':<12}{'n':>5}{'large wins':>12}{'tie':>9}{'small wins':>12}{'large net':>11}")
for tier in ("TRIVIAL", "WORK"):
    c = results[tier]; n = sum(c.values())
    if not n: continue
    print(f"  {tier:<12}{n:5d}{c['large']/n:11.1%}{c['tie']/n:9.1%}{c['small']/n:11.1%}"
          f"{(c['large']-c['small'])/n:+10.1%}")
json.dump({k: dict(v) for k, v in results.items()}, open(ROOT/"reports/quality_delta.json","w"), indent=1)
