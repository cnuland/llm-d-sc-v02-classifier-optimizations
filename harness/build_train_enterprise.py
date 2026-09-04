"""Unconditioned enterprise scenes for TRAINING sensitivity, plus targeted
coverage of the tiers that never appear by chance.

Two problems to solve at once.

1. The shipped sensitivity model scores 0.2903 on enterprise text -- below the
   un-finetuned base at 0.4113 -- because it was trained on a fictional register
   (§10). Training on unconditioned enterprise scenes fixes the register.

2. NEVER_EGRESS has 5 unanimous enterprise rows and 2 real ones. Nobody writes
   an API key into a chat because a taxonomy asked them to, so the tier is
   invisible to unconditioned sampling. The fix is NOT to prompt for the tier --
   that reintroduces the tier-shaped fingerprint this whole eval design exists
   to avoid. Instead we steer the SITUATION, not the label: someone debugging a
   failing deploy naturally pastes a CI config, someone rotating keys naturally
   quotes the old one. The tier is still assigned afterwards by the blind jury,
   and whatever mix results is the mix.

Roles, settings and moments are disjoint from the eval generator's, and every
row is checked against the eval keys by content hash.
"""
import json, random, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy, load_jsonl
import llmkit as L
from sdg import blind_relabel, key

GEN = "claude-sonnet-5"                 # eval used opus-5
LABELLERS = ["claude-opus-5", "claude-sonnet-5"]
SIG = "sensitivity"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 260

ROLES = [
 "platform engineer","release manager","identity and access admin","SOC analyst",
 "database reliability engineer","integration developer","mobile developer",
 "solutions architect","technical account manager","business analyst",
 "treasury analyst","tax accountant","benefits administrator","talent operations lead",
 "litigation support specialist","contracts manager","privacy officer",
 "clinical data manager","medical coder","utilization review nurse",
 "underwriter","actuary","branch compliance officer","AML investigator",
 "channel partner manager","pricing analyst","field sales engineer",
 "warehouse operations manager","quality engineer","EHS coordinator",
 "curriculum designer","admissions officer","city planning analyst",
 "donor relations manager","board secretary","internal communications lead",
 "site reliability manager","data governance lead","vendor risk analyst",
]
SETTINGS = [
 "a mid-size credit union","a specialty pharmacy chain","a defense subcontractor",
 "a fast-growing fintech after a security incident","a K-12 school district",
 "a global logistics operator","a biotech startup pre-IPO","a property insurer",
 "a regional utility","a media company in licensing negotiations",
]
# Situations, not labels. Credentials and privilege appear because the WORK
# involves them, which is how they appear in production traffic.
MOMENTS = [
 "while a deployment is failing and they are pasting the pipeline config",
 "rotating a credential that expired overnight",
 "cleaning up an old .env file they found in a repo",
 "chasing an authentication error with the connection string in hand",
 "forwarding a note from outside counsel for a plain-English summary",
 "triaging an alert with the raw log attached",
 "reconciling a spreadsheet exported from the billing system",
 "preparing an access review for the auditors",
 "handing over on-call and writing up open issues",
 "writing a postmortem for last week's incident",
 "answering a customer complaint that quotes their account details",
 "setting up a new integration and testing the token",
 "summarising an offer package for a candidate",
 "reviewing an invoice dispute with the vendor contract open",
 "checking why a patient record failed to sync",
 "drafting a message to the team about a reorg that is not announced yet",
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
         f"- Include the concrete material the moment implies, pasted the way a "
         f"hurried person pastes it: config fragments, log lines, table rows, "
         f"quoted paragraphs, identifiers, figures.\n"
         f"- Invent every value. Plausible but entirely fictional.\n"
         f"- Vary from one line to a long paste. Some messages are mundane.\n"
         f"- Do not classify, categorise, or comment. Just the messages.\n")
    try:
        return [x["text"] for x in L.ask_json(p, SCHEMA, model=GEN,
                max_tokens=12000, seed_tag=tag)["items"]]
    except Exception as e:
        print(f"    ! scene failed: {type(e).__name__}", flush=True); return []

jobs=[(ROLES[i%len(ROLES)], SETTINGS[(i*7)%len(SETTINGS)],
       MOMENTS[(i*3)%len(MOMENTS)], 8, f"entr{i}") for i in range(N)]
print(f"generating {len(jobs)} enterprise training scenes")
texts=[t for part in L.amap(scene,jobs,workers=12,desc="scenes") for t in part]

evalkeys=set()
for f in [f"{SIG}-enterprise-gold", f"{SIG}-enterprise-contested",
          f"{SIG}-real-gold", f"{SIG}-real-contested"]:
    p=ROOT/f"data/eval/{f}.jsonl"
    if p.exists(): evalkeys |= {key(r["text"]) for r in load_jsonl(p)}
seen,pool=set(),[]
for t in texts:
    t=(t or "").strip()
    if not (15<=len(t)<=4000): continue
    k=key(t)
    if k in seen or k in evalkeys: continue
    seen.add(k); pool.append(t)
print(f"  {len(texts)} generated -> {len(pool)} unique, disjoint from eval")

labels=load_taxonomy(SIG)["labels"]
votes={m:[x["label"] for x in blind_relabel(SIG,labels,pool,model=m,batch=15,effort="low")]
       for m in LABELLERS}
kept=[{"text":t,"tier":votes[LABELLERS[0]][i],"source":"enterprise-labelled"}
      for i,t in enumerate(pool)
      if votes[LABELLERS[0]][i] is not None
      and votes[LABELLERS[0]][i]==votes[LABELLERS[1]][i]]
print(f"  agreed {len(kept)}/{len(pool)} = {len(kept)/max(1,len(pool)):.3f}")
print(f"  tier mix: {dict(collections.Counter(r['tier'] for r in kept))}")
out=ROOT/f"data/train/{SIG}-enterprise.jsonl"
with open(out,"w") as fh:
    for r in kept: fh.write(json.dumps(r)+"\n")
print(f"  -> {out}")
