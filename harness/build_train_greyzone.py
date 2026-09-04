"""Enterprise scenes aimed at the INTERNAL/CONFIDENTIAL grey zone.

§63 is the design constraint. Minimal pairs for this boundary scored 98.4%
blind-check agreement and were 8 points per row WORSE than real data: agreement
that high means the items are unambiguous, and the failures are the ambiguous
ones. The lesson was that **a synthetic corpus whose agreement sits far above the
real distribution's is off-distribution by construction** — so agreement is a
target BAND, not a maximand. Real entsec agreement on this pair is roughly 75-85%.

The way to land in that band is to steer the SITUATION and let the jury decide,
never to name the tier. So this reuses `build_train_enterprise.py`'s
unconditioned-scene design — which produced the least off-distribution corpus in
§64's table, 78.3% separability against v2's 92.8% and real's 98.2% — and only
changes WHICH situations are sampled.

Every moment below is one where INTERNAL vs CONFIDENTIAL genuinely turns on
detail: ordinary running of the business (INTERNAL) shades into money, market
position or IP (CONFIDENTIAL) without any change of register. That is the region
the model cannot separate (§57), and it is invisible to the original MOMENTS
list, which is weighted toward credentials and regulated data.

Writes its OWN file. `build_train_enterprise.py` opens its output "w" and would
silently replace 31,168 existing rows — the same in-place overwrite that lost the
v1 complexity labels (§65).
"""
import json, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy, load_jsonl
import llmkit as L
from sdg import blind_relabel, key

GEN = "claude-sonnet-5"
LABELLERS = ["claude-opus-5", "claude-sonnet-5"]
SIG = "sensitivity"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

ROLES = [
 "pricing analyst","FP&A analyst","corporate development associate","product manager",
 "partner manager","sales operations lead","competitive intelligence analyst",
 "procurement lead","engineering manager","principal architect","chief of staff",
 "investor relations associate","category manager","revenue operations analyst",
 "portfolio manager","brand strategist","supply chain planner","licensing manager",
 "M&A integration lead","customer success director","field CTO","staff data scientist",
]
SETTINGS = [
 "a mid-size credit union","a specialty pharmacy chain","a defense subcontractor",
 "a fast-growing fintech","a K-12 curriculum publisher","a global logistics operator",
 "a biotech startup pre-IPO","a property insurer","a regional utility",
 "a media company in licensing negotiations","an industrial robotics vendor",
 "a grocery co-operative","a hotel group","a payments processor",
]
# Situations, never labels. Each one can land either side of INTERNAL /
# CONFIDENTIAL depending on details the writer chooses without being told to.
MOMENTS = [
 "pulling together the numbers for next quarter's forecast",
 "comparing two vendors' quotes ahead of a renewal",
 "writing up why a large deal slipped this quarter",
 "preparing talking points for a pricing conversation with a major account",
 "sketching the roadmap slide for an all-hands",
 "drafting the competitive positioning section of a deck",
 "reviewing the churn list before a quarterly business review",
 "documenting an architecture decision that touches the auth path",
 "summarising a partner term sheet for their manager",
 "reworking headcount plans after a budget cut",
 "explaining a margin variance to finance",
 "putting together a build-versus-buy recommendation",
 "annotating a customer list before a territory reshuffle",
 "writing the internal FAQ for a product sunset that is not public yet",
 "reviewing a draft press release with the numbers still in it",
 "assessing whether a competitor's launch changes their plan",
 "walking through a discount approval that went outside policy",
 "capturing notes from a diligence call",
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
         f"hurried person pastes it: table rows, figures, quoted paragraphs, "
         f"slide bullets, email fragments, identifiers.\n"
         f"- Invent every value. Plausible but entirely fictional.\n"
         f"- Vary the stakes. Some of these are routine housekeeping; some carry "
         f"real commercial weight. Do not signal which is which.\n"
         f"- Vary from one line to a long paste. Some messages are mundane.\n"
         f"- Do not classify, categorise, or comment. Just the messages.\n")
    try:
        return [x["text"] for x in L.ask_json(p, SCHEMA, model=GEN,
                max_tokens=12000, seed_tag=tag)["items"]]
    except Exception as e:
        print(f"    ! scene failed: {type(e).__name__}", flush=True); return []

jobs = [(ROLES[i % len(ROLES)], SETTINGS[(i*5) % len(SETTINGS)],
         MOMENTS[(i*3) % len(MOMENTS)], 8, f"grey{i}") for i in range(N)]
print(f"generating {len(jobs)} grey-zone scenes")
texts = [t for part in L.amap(scene, jobs, workers=12, desc="scenes") for t in part]

evalkeys = set()
for f in (f"{SIG}-enterprise-gold", f"{SIG}-enterprise-contested", f"{SIG}-entsec-gold",
          f"{SIG}-entsec-contested", f"{SIG}-real-gold", f"{SIG}-real-contested"):
    p = ROOT/f"data/eval/{f}.jsonl"
    if p.exists(): evalkeys |= {key(r["text"]) for r in load_jsonl(p)}
trainkeys = set()
for f in ("sensitivity-enterprise.jsonl",):
    p = ROOT/"data/train"/f
    if p.exists(): trainkeys |= {key(json.loads(l)["text"]) for l in open(p)}

seen, pool = set(), []
for t in texts:
    t = (t or "").strip()
    if not (15 <= len(t) <= 4000): continue
    k = key(t)
    if k in seen or k in evalkeys or k in trainkeys: continue
    seen.add(k); pool.append(t)
print(f"  {len(texts)} generated -> {len(pool)} unique, disjoint from eval and existing train")

labels = load_taxonomy(SIG)["labels"]
votes = {m: [x["label"] for x in blind_relabel(SIG, labels, pool, model=m, batch=15, effort="low")]
         for m in LABELLERS}
agree_n = sum(1 for i in range(len(pool))
              if votes[LABELLERS[0]][i] and votes[LABELLERS[0]][i] == votes[LABELLERS[1]][i])
rate = agree_n / max(1, len(pool))
kept = [{"text": t, "tier": votes[LABELLERS[0]][i], "source": "greyzone"}
        for i, t in enumerate(pool)
        if votes[LABELLERS[0]][i] is not None
        and votes[LABELLERS[0]][i] == votes[LABELLERS[1]][i]]
print(f"  blind agreement {rate:.1%}  (§63 target band 75-85%; "
      f"{'ON TARGET' if 0.70 <= rate <= 0.88 else 'OFF TARGET — too unambiguous to be useful' if rate > 0.88 else 'very noisy'})")
print(f"  tier mix: {dict(collections.Counter(r['tier'] for r in kept))}")
out = ROOT/"data/train/sensitivity-greyzone.jsonl"
with open(out, "w") as fh:
    for r in kept: fh.write(json.dumps(r) + "\n")
print(f"  -> {out}  ({len(kept)} rows)")
