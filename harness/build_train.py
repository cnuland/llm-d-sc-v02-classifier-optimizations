"""Generate and verify the v2 training corpora.

Composition is deliberate, not uniform. The confusion matrices from every prior
run put almost all their mass on ONE boundary per signal (MEDIUM/COMPLEX for
complexity, INTERNAL/CONFIDENTIAL for sensitivity, MODERATE/HIGH for cost), so
roughly half the budget goes to boundary-shaped data:

  grid      broad coverage over (tier x domain x register)
  pairs     minimal pairs straddling each confusable boundary
  anticue   items that read like the neighbouring tier but are not
  real      rewrites in the voice of actual WildChat traffic

Then every row is BLIND re-labelled by a second model. Rows where the verifier
disagrees with the generator are not silently dropped -- they are written to
<signal>-contested.jsonl, because a generator/verifier disagreement is either a
bad row or a genuinely hard one, and the second kind is valuable.
"""
import json, pathlib, random, sys, collections, itertools
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import SIGNALS, load_taxonomy, load_jsonl
import llmkit as L
import sdg
from sdg import DOMAINS, REGISTERS, key, norm

ROOT = pathlib.Path("/Users/cnuland/llm-d-sc-accuracy")
GEN = "claude-sonnet-5"
VERIFY = "claude-opus-5"
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
ONLY = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "all" else None

# Which boundaries actually cost us accuracy. Ordered pairs: (tier, looks_like).
CONFUSABLE = {
    "complexity": [("MEDIUM","COMPLEX"),("COMPLEX","MEDIUM"),("SIMPLE","MEDIUM"),
                   ("MEDIUM","SIMPLE"),("COMPLEX","REASONING"),("REASONING","COMPLEX"),
                   ("MEDIUM","REASONING"),("REASONING","MEDIUM")],
    "cost": [("MODERATE","HIGH"),("HIGH","MODERATE"),("MINIMAL","LOW"),("LOW","MINIMAL"),
             ("LOW","MODERATE"),("MODERATE","LOW"),("MINIMAL","MODERATE"),("HIGH","LOW")],
    "sensitivity": [("INTERNAL","CONFIDENTIAL"),("CONFIDENTIAL","INTERNAL"),
                    ("CONFIDENTIAL","REGULATED"),("REGULATED","CONFIDENTIAL"),
                    ("REGULATED","NEVER_EGRESS"),("NEVER_EGRESS","REGULATED"),
                    ("PUBLIC","INTERNAL"),("INTERNAL","PUBLIC")],
}
BOUNDARIES = {s: sorted({tuple(sorted(p)) for p in v}) for s, v in CONFUSABLE.items()}

real_pool = [r["text"] for r in load_jsonl(ROOT/"data/eval/wildchat_pool.jsonl")]


def plan(signal, labels):
    """Build the full call list up front so progress is legible and resumable."""
    rnd = random.Random(f"plan-{signal}")
    jobs = []
    n_grid = max(1, int(26 * SCALE))
    for tier in labels:
        for i in range(n_grid):
            d = DOMAINS[(i * 7 + labels.index(tier)) % len(DOMAINS)]
            r = REGISTERS[(i * 3 + labels.index(tier)) % len(REGISTERS)]
            jobs.append(("grid", (signal, labels, d, r, tier, 10, GEN,
                                  f"g{signal}{tier}{i}")))
    n_pair = max(1, int(9 * SCALE))
    for (a, b) in BOUNDARIES[signal]:
        for i in range(n_pair):
            d = DOMAINS[(i * 11 + hash(a+b)) % len(DOMAINS)]
            jobs.append(("pairs", (signal, labels, a, b, d, 5, GEN,
                                   f"p{signal}{a}{b}{i}")))
    n_anti = max(1, int(7 * SCALE))
    for (tier, decoy) in CONFUSABLE[signal]:
        for i in range(n_anti):
            d = DOMAINS[(i * 13 + hash(tier+decoy)) % len(DOMAINS)]
            jobs.append(("anticue", (signal, labels, tier, decoy, d, 8, GEN,
                                     f"a{signal}{tier}{decoy}{i}")))
    n_real = max(1, int(14 * SCALE))
    for tier in labels:
        for i in range(n_real):
            ex = rnd.sample(real_pool, 6)
            jobs.append(("real", (signal, labels, ex, tier, 10, GEN,
                                  f"r{signal}{tier}{i}")))
    rnd.shuffle(jobs)
    return jobs


FN = {"grid": sdg.gen_grid, "pairs": sdg.gen_pairs,
      "anticue": sdg.gen_anticue, "real": sdg.gen_from_real}


def run(signal):
    labels = load_taxonomy(signal)["labels"]
    jobs = plan(signal, labels)
    print(f"\n########## {signal}: {len(jobs)} generation calls ##########", flush=True)

    def one(j):
        kind, args = j
        try:
            return [dict(x, source=kind) for x in FN[kind](*args)]
        except Exception as e:
            print(f"    ! {kind} failed: {type(e).__name__}: {str(e)[:110]}", flush=True)
            return []

    got = L.amap(one, jobs, workers=14, desc=f"gen {signal}")
    rows = [x for part in got for x in part]

    # de-dup, and drop anything colliding with a held-out prompt
    ho = set()
    for p in [ROOT/f"data/eval/{signal}-real-gold.jsonl",
              ROOT/f"data/eval/{signal}-real-contested.jsonl"]:
        if p.exists():
            ho |= {key(r["text"]) for r in load_jsonl(p)}
    from evalkit import heldout
    ho |= {key(r["text"]) for r in heldout(signal)}

    seen, clean = set(), []
    for r in rows:
        t = (r.get("text") or "").strip()
        if not (10 <= len(t) <= 4000):
            continue
        k = key(t)
        if k in seen or k in ho:
            continue
        seen.add(k)
        clean.append({"text": t, "tier": r["tier"], "source": r["source"]})
    print(f"  generated {len(rows)} -> {len(clean)} unique, leak-free")
    print(f"  by source: {dict(collections.Counter(r['source'] for r in clean))}")
    print(f"  by tier  : {dict(collections.Counter(r['tier'] for r in clean))}")

    # blind verification
    v = sdg.blind_relabel(signal, labels, [r["text"] for r in clean],
                          model=VERIFY, batch=20, effort="low")
    kept, contested = [], []
    for r, vv in zip(clean, v):
        (kept if vv["label"] == r["tier"] else contested).append(
            dict(r, verifier=vv["label"], vconf=vv["confidence"]))
    rate = len(kept) / max(1, len(clean))
    print(f"  verifier agreed on {len(kept)}/{len(clean)} = {rate:.3f}")
    dis = collections.Counter(f"{r['tier']}->{r['verifier']}" for r in contested)
    print(f"  top disagreements: {dis.most_common(6)}")

    with open(ROOT/f"data/train/{signal}-v2.jsonl", "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    with open(ROOT/f"data/train/{signal}-v2-contested.jsonl", "w") as fh:
        for r in contested:
            fh.write(json.dumps(r) + "\n")
    print(f"  kept tier mix: {dict(collections.Counter(r['tier'] for r in kept))}")
    return len(kept)


for s in ([ONLY] if ONLY else SIGNALS):
    run(s)
