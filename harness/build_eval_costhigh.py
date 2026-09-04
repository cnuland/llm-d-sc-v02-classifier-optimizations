"""A cost eval slice with enough HIGH-tier examples to be measurable.

Cost's weakest tier is HIGH (recall 0.62) and it has **13 rows** in the refined
gold — a 95% CI of roughly 0.36-0.83. It is the rarest tier in real traffic
(227 of 6,000 screened, 3.8%), so random sampling will never give it enough
support. Exactly the problem sensitivity's egress tiers had before the
enterprise-secrets slice, and the same remedy applies.

Unconditioned, as always: the generator is given a WORK SITUATION that naturally
involves large volumes of material — a migration audit, a quarter of tickets, a
document review — and is never told the taxonomy or that HIGH exists. Some of
what it writes will be MODERATE or LOW, and that is correct: the point is to
enrich the region around the MODERATE/HIGH boundary, not to manufacture a
particular label. The blind three-model jury assigns tiers afterwards.
"""
import json, random, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy, load_jsonl
import llmkit as L
from sdg import blind_relabel, key

GEN = "claude-sonnet-5"
JURY = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"]
SIG = "cost"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120

ROLES = [
 "technical writer","research assistant","business analyst","paralegal",
 "data migration engineer","QA automation lead","content operations manager",
 "financial auditor","market researcher","curriculum developer",
 "localisation manager","support operations analyst","systems archivist",
 "grant writer","competitive intelligence analyst","compliance documentation lead",
]
SETTINGS = [
 "a software company mid-migration","a law firm in discovery",
 "a publisher relaunching a catalogue","a bank closing the quarter",
 "a university revising a programme","a marketplace auditing listings",
]
# Situations that naturally involve VOLUME, without ever naming a tier.
MOMENTS = [
 "working through a quarter of support tickets",
 "reviewing every page of a vendor contract set",
 "consolidating notes from a dozen interviews",
 "auditing an entire repository before a handover",
 "summarising a stack of research papers for a literature section",
 "translating a full product manual",
 "rewriting a documentation site section by section",
 "reconciling a year of transaction exports",
 "drafting one section of a report",          # deliberately bounded, for contrast
 "answering a single question from a colleague",   # deliberately small
 "writing a short update for the team",            # deliberately small
 "preparing a detailed design document",           # deliberately MODERATE
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
         f"AI assistant at that moment.\n\n"
         f"- Write what they would actually type, including how much material "
         f"they would reference and how much output they would ask for.\n"
         f"- Vary the ask: some are quick questions, some are large jobs.\n"
         f"- Do not classify, categorise, or comment. Just the messages.\n")
    try:
        return [x["text"] for x in L.ask_json(p, SCHEMA, model=GEN,
                max_tokens=12000, seed_tag=tag)["items"]]
    except Exception as e:
        print(f"    ! scene failed: {type(e).__name__}", flush=True); return []

jobs=[(ROLES[i%len(ROLES)], SETTINGS[(i*5)%len(SETTINGS)],
       MOMENTS[(i*7)%len(MOMENTS)], 8, f"costhigh-{i}") for i in range(N)]
print(f"generating {len(jobs)} volume-varied work scenes (unconditioned)")
texts=[t for part in L.amap(scene,jobs,workers=10,desc="scenes") for t in part]

seen_keys=set()
for f in [f"data/train/{SIG}-v2.jsonl", f"data/train/{SIG}-real.jsonl",
          f"data/train/{SIG}-real-contested.jsonl",
          f"data/eval/{SIG}-real-gold.jsonl", f"data/eval/{SIG}-real-gold-v2.jsonl"]:
    p=ROOT/f
    if p.exists(): seen_keys |= {key(r["text"]) for r in load_jsonl(p)}
pool,seen=[],set()
for t in texts:
    t=(t or "").strip()
    if not (15<=len(t)<=4000): continue
    k=key(t)
    if k in seen or k in seen_keys: continue
    seen.add(k); pool.append(t)
print(f"  {len(texts)} generated -> {len(pool)} unique, disjoint from training and eval")

labels=load_taxonomy(SIG)["labels"]
votes={m:[x["label"] for x in blind_relabel(SIG,labels,pool,model=m,batch=15,effort="medium")]
       for m in JURY}
gold,cont,unres=[],[],0
for i,t in enumerate(pool):
    vs=[votes[m][i] for m in JURY]
    if any(v is None for v in vs): continue
    c=collections.Counter(vs); top,n=c.most_common(1)[0]
    rec={"text":t,"tier":top,"votes":vs,"src":"cost-volume"}
    if n==len(JURY): gold.append(rec)
    elif n>=2: cont.append(rec)
    else: unres+=1
for name,rs in [("gold",gold),("contested",cont)]:
    out=ROOT/f"data/eval/{SIG}-volume-{name}.jsonl"
    with open(out,"w") as fh:
        for r in rs: fh.write(json.dumps(r)+"\n")
print(f"\nunanimous {len(gold)}  contested {len(cont)}  unresolved {unres}")
print(f"gold mix: {dict(collections.Counter(r['tier'] for r in gold))}")
print(f"-> data/eval/{SIG}-volume-gold.jsonl")
