"""Resample a corpus toward a TARGET class prior.

Round X measured uniform balancing and it lost: bge on the raw corpus scored
0.8119 on entsec, and the same model on the uniformly balanced corpus scored
0.7808 -- **-3.11 points**. That is the opposite of §57's prediction, and §56
explains it. Uniform balancing downsampled INTERNAL from 16,450 to 7,694, but
INTERNAL is 51.2% of the entsec eval. Balancing moved the training prior AWAY
from the evaluation prior, and §56 had already measured that mismatch as worth
about a point via logit adjustment alone.

So the question is not "balanced or not", it is WHICH prior. Two candidates:

  eval    match the evaluation prior exactly -- maximally matched, but it starves
          CONFIDENTIAL down to 5.8% of training, and CONFIDENTIAL is the class
          §57 showed the model cannot learn
  sqrt    geometric interpolation between the corpus and eval priors -- keeps the
          direction of the correction while leaving rare classes enough examples

Both are hypotheses with a real tension between them, which is why both get run
rather than one getting argued for.
"""
import sys, json, collections, random, pathlib
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from evalkit import ROOT, load_jsonl

mode = sys.argv[1]                      # eval | sqrt
out  = sys.argv[2]
files = sys.argv[3].split(",")
evalf = sys.argv[4].split(",")
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rng = random.Random(seed)

by = collections.defaultdict(list)
for f in files:
    for l in open(f):
        r = json.loads(l)
        by[r.get("tier") or r.get("label")].append(l.rstrip("\n"))
ev = collections.Counter(r["tier"] for f in evalf for r in load_jsonl(ROOT/"data/eval"/f))

N = sum(len(v) for v in by.values()); E = sum(ev.values())
p_tr = {k: len(v)/N for k, v in by.items()}
p_ev = {k: ev.get(k, 0)/E for k in by}
if mode == "eval":
    tgt = p_ev
else:
    g = {k: (p_tr[k]*p_ev[k])**0.5 for k in by}
    s = sum(g.values()); tgt = {k: v/s for k, v in g.items()}

# largest class that needs no upsampling fixes the total, so nothing is invented
total = min(len(by[k])/tgt[k] for k in by if tgt[k] > 0)
rows = []
print(f"{'tier':<15}{'have':>8}{'train%':>9}{'eval%':>8}{'target%':>9}{'take':>8}")
for k in sorted(by):
    want = int(total*tgt[k])
    v = by[k]
    take = rng.sample(v, want) if want <= len(v) else v + [rng.choice(v) for _ in range(want-len(v))]
    rows += take
    print(f"{k:<15}{len(v):8d}{p_tr[k]:8.1%}{p_ev[k]:7.1%}{tgt[k]:8.1%}{len(take):8d}")
rng.shuffle(rows)
pathlib.Path(out).write_text("\n".join(rows) + "\n")
print(f"  total={len(rows)} -> {out}")
