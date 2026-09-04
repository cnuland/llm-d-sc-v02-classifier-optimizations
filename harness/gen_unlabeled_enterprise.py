"""Generate a large UNLABELLED enterprise pool for distillation.

Distillation gave complexity its best single model, but it cannot help
sensitivity off the WildChat pool: that pool is 85% PUBLIC, so 30,000 distilled
rows would teach almost nothing about the tiers that gate egress while flooding
training with majority-class examples.

What sensitivity needs is a large pool in the ENTERPRISE distribution. Jury
labelling one costs thousands of API calls; the ensemble can label it for free.
So generate the prompts only — no jury, no verification — and let the teacher
supply the targets.

Because nothing here is used as ground truth, generation can be cheap and broad:
many roles x settings x moments, unconditioned as always (the model is never
shown the taxonomy), deduplicated, and checked against every eval by content
hash so no distilled row can leak into a measurement.
"""
import json, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl
import llmkit as L
from sdg import key
from build_train_enterprise import ROLES, SETTINGS, MOMENTS, SCHEMA, GEN

N = int(sys.argv[1]) if len(sys.argv) > 1 else 900

def scene(a):
    role, setting, moment, n, tag = a
    p = (f"You are helping collect realistic examples of how people at work "
         f"actually use an AI assistant.\n\n"
         f"Person: a {role}\nOrganisation: {setting}\nMoment: {moment}\n\n"
         f"Write {n} DIFFERENT things this person might genuinely type into an "
         f"internal AI assistant at that moment.\n\n"
         f"- Write what they would actually type, including whatever detail they "
         f"would naturally reference or quote.\n"
         f"- Everything is fictional; use obvious placeholders for identifiers.\n"
         f"- Vary from one line to a long paste; some messages are mundane.\n"
         f"- Do not classify, categorise, or comment. Just the messages.\n")
    try:
        return [x["text"] for x in L.ask_json(p, SCHEMA, model=GEN,
                max_tokens=12000, seed_tag=tag)["items"]]
    except Exception:
        return []

jobs = [(ROLES[i % len(ROLES)], SETTINGS[(i * 11) % len(SETTINGS)],
         MOMENTS[(i * 13) % len(MOMENTS)], 10, f"unlab-{i}") for i in range(N)]
print(f"generating {len(jobs)} unlabelled enterprise scenes")
texts = [t for part in L.amap(scene, jobs, workers=12, desc="scenes") for t in part]

# exclude anything that appears in any eval or labelled training file
seen_keys = set()
import glob
for p in glob.glob(str(ROOT / "data/eval/sensitivity-*.jsonl")) + \
         glob.glob(str(ROOT / "data/train/sensitivity-*.jsonl")):
    try:
        seen_keys |= {key(r["text"]) for r in load_jsonl(p)}
    except Exception:
        pass
pool, seen = [], set()
for t in texts:
    t = (t or "").strip()
    if not (15 <= len(t) <= 4000):
        continue
    k = key(t)
    if k in seen or k in seen_keys:
        continue
    seen.add(k); pool.append(t)

out = ROOT / "data/eval/enterprise_unlabeled_pool.jsonl"
with open(out, "w") as fh:
    for t in pool:
        fh.write(json.dumps({"text": t, "src": "enterprise-unlabeled"}) + "\n")
print(f"  {len(texts)} generated -> {len(pool)} unique, disjoint from all "
      f"sensitivity evals and labelled training data")
print(f"  -> {out}")
