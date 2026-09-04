"""Does GENERATION POOL BREADTH drive separability from the eval?

§76 falsified the agreement-band explanation and left separability as the
operative variable, with one untested hypothesis for what controls it: the
enterprise corpus that works (78.1% separable) accumulated 31,168 rows over
39 roles x 10 settings x 16 moments across many runs, while the grey-zone corpus
that failed (94.8%) drew 2,075 rows from 22 x 14 x 18 in a single pass.

Narrow pool -> repeated combinations -> homogeneous text -> easy to tell from
real traffic. Plausible, and exactly the kind of plausible story §63 was.

So test it directly instead of asserting it. Two arms, SAME model, SAME prompt
template, SAME number of scenes and items, same single pass. The only thing that
differs is how many distinct (role, setting, moment) combinations the scenes are
drawn from:

  NARROW   6 x 3 x 4  =    72 combinations, so each recurs ~2.2x over 160 scenes
  WIDE   120 x 40 x 60 = 288,000 combinations, so no combination recurs

Separability is measured against the same eval both times. If breadth is the
mechanism, WIDE lands materially closer to 50%. If the two come out level, the
mechanism is something else -- the generating MODEL's register, most likely --
and no amount of pool engineering will fix synthetic data for this eval.
"""
import json, sys, random, collections, itertools, pathlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_jsonl
import llmkit as L
from sdg import key

GEN = "claude-sonnet-5"
SCENES = int(sys.argv[1]) if len(sys.argv) > 1 else 160

FUNCS = ["engineering","finance","legal","sales","marketing","operations","HR",
         "security","data","support","procurement","product","compliance","facilities",
         "research","clinical","logistics","quality","training","communications"]
SENIOR = ["junior","senior","lead","principal","director of","interim head of",
          "contract","graduate","staff","deputy"]
NOUNS  = ["analyst","engineer","manager","coordinator","specialist","administrator"]
INDUST = ["credit union","pharmacy chain","defense subcontractor","fintech","school district",
          "logistics operator","biotech","property insurer","utility","media company",
          "robotics vendor","grocery co-op","hotel group","payments processor","law firm",
          "hospital network","university","charity","car dealership","games studio"]
SIZE   = ["a 40-person","a mid-size","a global","a family-owned","a newly merged",
          "a private-equity-owned","a state-run","a fast-shrinking"]
ACTS   = ["chasing a deadline","cleaning up after a handover","preparing for an audit",
          "onboarding someone new","closing out a quarter","reacting to a complaint",
          "writing something for their manager","checking a supplier's claim",
          "reconciling two systems that disagree","drafting a policy nobody has read",
          "answering a question from another team","triaging a backlog"]
WHEN   = ["late on a Friday","first thing Monday","during an incident","before a review",
          "while travelling"]

SCHEMA = {"type":"object","additionalProperties":False,"required":["items"],
 "properties":{"items":{"type":"array","items":{"type":"object",
   "additionalProperties":False,"required":["text"],
   "properties":{"text":{"type":"string"}}}}}}

def scene(a):
    role, org, moment, tag = a
    p = (f"You are helping collect realistic examples of how people at work "
         f"actually use an AI assistant.\n\n"
         f"Person: a {role}\nOrganisation: {org}\nMoment: {moment}\n\n"
         f"Write 8 DIFFERENT things this person might genuinely type into an "
         f"internal AI assistant at that moment.\n\n"
         f"- Include the concrete material the moment implies, pasted the way a "
         f"hurried person pastes it.\n"
         f"- Invent every value. Plausible but entirely fictional.\n"
         f"- Vary from one line to a long paste. Some messages are mundane.\n"
         f"- Do not classify, categorise, or comment. Just the messages.\n")
    try:
        return [x["text"] for x in L.ask_json(p, SCHEMA, model=GEN, max_tokens=12000,
                                              seed_tag=tag)["items"]]
    except Exception as e:
        print(f"    ! {type(e).__name__}", flush=True); return []

def build(name, roles, orgs, moments):
    rng = random.Random(11)
    jobs = [(rng.choice(roles), rng.choice(orgs), rng.choice(moments), f"{name}{i}")
            for i in range(SCENES)]
    combos = len(set((a, b, c) for a, b, c, _ in jobs))
    print(f"\n[{name}] pool {len(roles)}x{len(orgs)}x{len(moments)} = "
          f"{len(roles)*len(orgs)*len(moments):,} combos; "
          f"{combos} distinct across {SCENES} scenes")
    texts = [t for part in L.amap(scene, jobs, workers=12, desc=name) for t in part]
    seen, out = set(), []
    for t in texts:
        t = (t or "").strip()
        if not (15 <= len(t) <= 4000): continue
        k = key(t)
        if k in seen: continue
        seen.add(k); out.append({"text": t, "tier": None, "source": name})
    p = ROOT/f"data/train/breadth-{name}.jsonl"
    p.write_text("".join(json.dumps(r)+"\n" for r in out))
    print(f"[{name}] {len(texts)} generated -> {len(out)} unique -> {p.name}")

# NARROW: a deliberately small pool, drawn the way the grey-zone run was
build("narrow", [f"{s} {n}" for s, n in itertools.islice(itertools.product(SENIOR, NOUNS), 6)],
      [f"{s} {i}" for s, i in itertools.islice(itertools.product(SIZE, INDUST), 3)],
      ACTS[:4])
# WIDE: same template, same count, combinatorially unique scenes
build("wide", [f"{s} {f} {n}" for s in SENIOR for f in FUNCS for n in NOUNS][:1200],
      [f"{s} {i}" for s in SIZE for i in INDUST],
      [f"{a} {w}" for a in ACTS for w in WHEN])
