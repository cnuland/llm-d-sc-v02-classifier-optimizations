"""An enterprise eval for sensitivity, built WITHOUT telling the generator a tier.

WildChat is consumer traffic: it screens 93% PUBLIC and yields only 6
CONFIDENTIAL / 7 REGULATED / 2 NEVER_EGRESS rows. It cannot measure the tiers
that actually matter for an egress decision.

The obvious fix -- "generate 200 CONFIDENTIAL prompts" -- reintroduces exactly
the circularity this project is trying to escape: a tier-conditioned generator
writes tier-shaped text, and the classifier learns the shape rather than the
substance. So generation here is UNCONDITIONED. A model is given a role, a
company setting and a moment in the working day, and asked what that person
would actually type. It is never told the taxonomy exists. The tiers arrive
afterwards, from the same blind three-model jury used on real traffic.

Whatever mix comes out is the mix; imbalance is a property of the world, and is
reported rather than engineered away.
"""
import json, pathlib, random, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy, load_jsonl
import llmkit as L
from sdg import blind_relabel, key

GEN = "claude-opus-5"                 # not the training generator (sonnet-5)
JURY = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"]
SIG = "sensitivity"
N_SCENE = int(sys.argv[1]) if len(sys.argv) > 1 else 130

ROLES = [
 "staff backend engineer","SRE on call","security analyst","IT helpdesk technician",
 "data engineer","ML engineer","engineering manager","product manager",
 "HR business partner","recruiter","payroll administrator","finance analyst",
 "controller","corporate counsel","paralegal","compliance officer",
 "clinical research coordinator","hospital billing specialist","nurse informaticist",
 "insurance claims adjuster","bank branch operations lead","credit risk analyst",
 "sales account executive","customer success manager","support escalation engineer",
 "marketing operations lead","procurement specialist","facilities manager",
 "manufacturing plant supervisor","supply chain planner","field service technician",
 "school district administrator","university registrar","municipal clerk",
 "non-profit grants manager","executive assistant to the CFO","internal auditor",
 "database administrator","network engineer","QA lead","technical writer",
 "M&A associate","real estate portfolio manager","pharmacovigilance associate",
]
SETTINGS = [
 "a 40-person startup with no formal data policy",
 "a regional hospital network",
 "a publicly traded manufacturer mid-quarter",
 "a retail bank under regulatory examination",
 "a government agency contractor",
 "a university IT department",
 "a Series C SaaS company preparing a funding round",
 "an insurance carrier during renewal season",
 "a logistics firm integrating an acquisition",
 "a pharmaceutical company running a trial",
]
MOMENTS = [
 "in the middle of an outage","the day before a board meeting",
 "while onboarding a new hire","during a customer escalation",
 "preparing a routine weekly report","after finding something odd in a log",
 "while cleaning up an old spreadsheet","responding to an auditor's request",
 "debugging a failing job at 2am","drafting something for a colleague",
 "reviewing a vendor contract","closing the books for the month",
]

SCHEMA = {"type":"object","additionalProperties":False,"required":["items"],
 "properties":{"items":{"type":"array","items":{
   "type":"object","additionalProperties":False,"required":["text"],
   "properties":{"text":{"type":"string"}}}}}}


def scene(a):
    role, setting, moment, n, tag = a
    p = (f"You are helping collect realistic examples of how people at work "
         f"actually use an AI assistant.\n\n"
         f"Person: a {role}\nOrganisation: {setting}\nMoment: {moment}\n\n"
         f"Write {n} DIFFERENT things this person might genuinely type into an "
         f"internal AI assistant at that moment.\n\n"
         f"Make them realistic, not illustrative:\n"
         f"- Some are mundane; some involve whatever material this person "
         f"actually handles day to day, pasted in as they would paste it.\n"
         f"- Include the concrete detail a real person includes -- names, ids, "
         f"figures, config, log lines, snippets of documents. Invent all of it; "
         f"it must be plausible but entirely fictional.\n"
         f"- Vary length from one line to a long paste.\n"
         f"- Do not classify, categorise, or comment. Just the messages.\n")
    try:
        return [x["text"] for x in L.ask_json(p, SCHEMA, model=GEN,
                max_tokens=12000, seed_tag=tag)["items"]]
    except Exception as e:
        print(f"    ! scene failed: {type(e).__name__}", flush=True)
        return []


rnd = random.Random(20260903)
jobs = []
for i in range(N_SCENE):
    jobs.append((ROLES[i % len(ROLES)], SETTINGS[(i*3) % len(SETTINGS)],
                 MOMENTS[(i*5) % len(MOMENTS)], 8, f"ent{i}"))
print(f"generating {len(jobs)} unconditioned workplace scenes")
texts = [t for part in L.amap(scene, jobs, workers=10, desc="scenes") for t in part]

seen, pool = set(), []
train_keys = set()
tp = ROOT/f"data/train/{SIG}-v2.jsonl"
if tp.exists():
    train_keys = {key(r["text"]) for r in load_jsonl(tp)}
for t in texts:
    t = (t or "").strip()
    if not (15 <= len(t) <= 4000):
        continue
    k = key(t)
    if k in seen or k in train_keys:
        continue
    seen.add(k); pool.append(t)
print(f"  {len(texts)} generated -> {len(pool)} unique, disjoint from training")

labels = load_taxonomy(SIG)["labels"]
votes = {}
for m in JURY:
    votes[m] = [x["label"] for x in blind_relabel(SIG, labels, pool, model=m,
                                                  batch=15, effort="medium")]
    print(f"  {m}: done")

gold, contested, unresolved, unanswered = [], [], 0, 0
for i, t in enumerate(pool):
    vs = [votes[m][i] for m in JURY]
    if any(v is None for v in vs):
        unanswered += 1; continue
    c = collections.Counter(vs); top, n = c.most_common(1)[0]
    rec = {"text": t, "tier": top, "votes": vs, "src": "enterprise-synthetic"}
    if n == len(JURY): gold.append(rec)
    elif n >= 2: contested.append(rec)
    else: unresolved += 1

for name, rows in [("gold", gold), ("contested", contested)]:
    with open(ROOT/f"data/eval/{SIG}-enterprise-{name}.jsonl", "w") as fh:
        for r in rows: fh.write(json.dumps(r)+"\n")

ok = [i for i in range(len(pool)) if all(votes[m][i] is not None for m in JURY)]
pair = lambda a,b: sum(votes[a][i]==votes[b][i] for i in ok)/max(1,len(ok))
print(f"\nunanimous {len(gold)}  contested {len(contested)}  unresolved {unresolved}"
      f"  unanswered {unanswered}")
print("pairwise: " + "  ".join(f"{JURY[a][7:18]}~{JURY[b][7:18]}={pair(JURY[a],JURY[b]):.3f}"
                               for a,b in [(0,1),(0,2),(1,2)]))
print("gold mix:", dict(collections.Counter(r["tier"] for r in gold)))
print("contested mix:", dict(collections.Counter(r["tier"] for r in contested)))
