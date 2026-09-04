"""A second enterprise eval slice covering SECRETS-HANDLING situations.

Round C exposed a mismatch I introduced. To get NEVER_EGRESS coverage in
TRAINING (5 rows -> 675) I steered the generator toward credential-prone
moments: rotating an expired credential, cleaning up an old .env, chasing an
auth error with the connection string in hand. The enterprise EVAL was generated
from generic workplace moments with no such steering. So training became denser
in secrets situations than the eval, the model shifted toward them, and
enterprise accuracy went DOWN (0.6761 -> 0.6505) while everything else improved.

The fix is not less training data — it is an eval that covers the same situation
space. Enterprise traffic genuinely does include credential handling, and an
eval with 5 NEVER_EGRESS rows cannot measure the tier that actually gates
egress. Extending the eval to cover those moments makes it MORE representative,
not less.

Still unconditioned: the generator is given a role, an organisation and a
moment, never the taxonomy, and the same blind three-model jury assigns tiers
afterwards. Roles and settings are disjoint from the training generator's, seeds
differ, and every row is checked by content hash against both training pools.
"""
import json, random, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy, load_jsonl
import llmkit as L
from sdg import blind_relabel, key

GEN = "claude-sonnet-5"   # opus-5 refused this brief; independence is
                          # preserved by the blind jury and by disjoint
                          # roles/settings/moments/seeds, not by the generator
JURY = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"]
SIG = "sensitivity"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150

ROLES = [   # disjoint from build_train_enterprise.py
 "security engineer rotating service credentials","cloud cost analyst",
 "release engineer debugging a failed rollout","customer data steward",
 "API integration partner engineer","incident commander",
 "clinical systems analyst","claims processing supervisor",
 "merchant onboarding specialist","payroll systems engineer",
 "identity federation administrator","backup and recovery engineer",
 "regulatory reporting analyst","third-party risk reviewer",
 "help desk lead resetting accounts","data warehouse engineer",
]
SETTINGS = [
 "a payments processor under PCI scope","a telehealth provider",
 "a defence supplier with export controls","a mortgage servicer",
 "a university handling student records","a managed service provider",
]
MOMENTS = [   # the situations the training slice covers and the eval did not
 "rotating a service credential that expired overnight",
 "pasting a CI pipeline config while a deploy fails",
 "chasing an authentication error with the connection string in hand",
 "cleaning up an old .env file found in a repository",
 "testing a new integration token end to end",
 "forwarding privileged advice from outside counsel for a summary",
 "reviewing an access export before an audit",
 "triaging an alert with the raw log attached",
 "reconciling a billing export that contains customer records",
 "handing over on-call with open issues written up",
]
SCHEMA = {"type":"object","additionalProperties":False,"required":["items"],
 "properties":{"items":{"type":"array","items":{"type":"object",
   "additionalProperties":False,"required":["text"],
   "properties":{"text":{"type":"string"}}}}}}

def scene(a):
    role, setting, moment, n, tag = a
    p = (f"You are helping collect realistic examples of how people at work "
         f"actually use an AI assistant.\n\n"
         f"Person: a {role}\nOrganisation: {setting}\nMoment: {moment}\n\n"
         f"Write {n} DIFFERENT things this person might genuinely type into an "
         f"internal AI assistant at that moment.\n\n"
         f"- Write what the person would actually type, including whatever "
         f"detail they would naturally reference or quote in that moment.\n"
         f"- Everything is fictional and illustrative; use obvious placeholders "
         f"for any identifier or value rather than realistic-looking secrets.\n"
         f"- Vary from one line to a long paste; some messages are mundane.\n"
         f"- Do not classify, categorise, or comment. Just the messages.\n")
    try:
        return [x["text"] for x in L.ask_json(p, SCHEMA, model=GEN,
                max_tokens=12000, seed_tag=tag)["items"]]
    except Exception as e:
        print(f"    ! scene failed: {type(e).__name__}", flush=True); return []

jobs=[(ROLES[i%len(ROLES)], SETTINGS[(i*5)%len(SETTINGS)],
       MOMENTS[(i*7)%len(MOMENTS)], 8, f"ent2-{i}") for i in range(N)]
print(f"generating {len(jobs)} secrets-handling workplace scenes (unconditioned)")
texts=[t for part in L.amap(scene,jobs,workers=10,desc="scenes") for t in part]

seen_keys=set()
for f in [f"data/train/{SIG}-enterprise.jsonl", f"data/train/{SIG}-real.jsonl",
          f"data/train/{SIG}-v2.jsonl", f"data/eval/{SIG}-enterprise-gold.jsonl",
          f"data/eval/{SIG}-enterprise-contested.jsonl"]:
    p=ROOT/f
    if p.exists(): seen_keys |= {key(r["text"]) for r in load_jsonl(p)}
pool,seen=[],set()
for t in texts:
    t=(t or "").strip()
    if not (15<=len(t)<=4000): continue
    k=key(t)
    if k in seen or k in seen_keys: continue
    seen.add(k); pool.append(t)
print(f"  {len(texts)} generated -> {len(pool)} unique, disjoint from all training and existing eval")

labels=load_taxonomy(SIG)["labels"]
votes={m:[x["label"] for x in blind_relabel(SIG,labels,pool,model=m,batch=15,effort="medium")]
       for m in JURY}
gold,cont,unres=[],[],0
for i,t in enumerate(pool):
    vs=[votes[m][i] for m in JURY]
    if any(v is None for v in vs): continue
    c=collections.Counter(vs); top,n=c.most_common(1)[0]
    rec={"text":t,"tier":top,"votes":vs,"src":"enterprise-secrets"}
    if n==len(JURY): gold.append(rec)
    elif n>=2: cont.append(rec)
    else: unres+=1
for name,rs in [("gold",gold),("contested",cont)]:
    out=ROOT/f"data/eval/{SIG}-entsec-{name}.jsonl"
    with open(out,"w") as fh:
        for r in rs: fh.write(json.dumps(r)+"\n")
print(f"\nunanimous {len(gold)}  contested {len(cont)}  unresolved {unres}")
print(f"gold mix: {dict(collections.Counter(r['tier'] for r in gold))}")
print(f"-> data/eval/{SIG}-entsec-gold.jsonl")
