"""Build the primary honest eval: REAL user prompts, jury-labelled.

Every number produced so far rests on prompts that were either hand-authored for
the eval or written by an LLM asked to hit a tier. Both are in-distribution with
our own idea of the task, which is exactly how the v1 complexity model came to
report 98.53% and then measure 97.5% on 80 hand-written rows with a +/-8pt CI.
Real traffic is the only set that can falsify a "high 90s" claim.

Two stages, because a careful jury over 24k prompts is wasteful:

  Stage 1  cheap screen (haiku, low effort) over the whole pool -> approximate
           tier, used ONLY to stratify. Never used as a label.
  Stage 2  three independent blind labellers over a balanced sample. Unanimous
           rows become the gold eval; 2-1 rows become a separate 'contested'
           set; 1-1-1 rows are discarded as genuinely ill-posed.

Reporting the contested and discarded counts is part of the result: they measure
how much of real traffic the taxonomy simply does not resolve.
"""
import json, pathlib, random, sys, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import SIGNALS, load_taxonomy, load_jsonl
import llmkit as L
from sdg import blind_relabel, key

ROOT = pathlib.Path("/Users/cnuland/llm-d-sc-accuracy")
POOL = ROOT / "data/eval/wildchat_pool.jsonl"
SCREEN_MODEL = "claude-haiku-4-5-20251001"
JURY = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"]
PER_TIER = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SCREEN_N = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

pool = load_jsonl(POOL)
random.Random(20260903).shuffle(pool)
pool = pool[:SCREEN_N]
print(f"pool={len(pool)} prompts")

for sig in SIGNALS:
    labels = load_taxonomy(sig)["labels"]
    print(f"\n########## {sig} ##########")

    # ---- stage 1: cheap stratification screen
    scr = blind_relabel(sig, labels, [r["text"] for r in pool],
                        model=SCREEN_MODEL, batch=25, effort="low")
    by = collections.defaultdict(list)
    for r, s in zip(pool, scr):
        by[s["label"]].append(r)
    print("  screen distribution:", {k: len(v) for k, v in by.items()})

    # ---- stage 2: balanced sample -> independent jury
    rnd = random.Random(hash(sig) & 0xffff)
    sample = []
    for lab in labels:
        c = by.get(lab, [])
        rnd.shuffle(c)
        sample += c[:PER_TIER]
    rnd.shuffle(sample)
    texts = [r["text"] for r in sample]
    print(f"  jury sample n={len(texts)} over {len(JURY)} labellers")

    votes = {}
    for m in JURY:
        v = blind_relabel(sig, labels, texts, model=m, batch=20, effort="medium")
        votes[m] = [x["label"] for x in v]
        print(f"    {m}: done")

    gold, contested, dropped, unanswered = [], [], 0, 0
    for i, t in enumerate(texts):
        vs = [votes[m][i] for m in JURY]
        if any(v is None for v in vs):
            unanswered += 1          # a juror would not label it; not evidence
            continue
        cnt = collections.Counter(vs)
        top, n = cnt.most_common(1)[0]
        rec = {"text": t, "tier": top, "votes": vs, "src": "wildchat"}
        if n == len(JURY):
            gold.append(rec)
        elif n >= 2:
            contested.append(rec)
        else:
            dropped += 1

    out = ROOT / f"data/eval/{sig}-real-gold.jsonl"
    with open(out, "w") as fh:
        for r in gold:
            fh.write(json.dumps(r) + "\n")
    with open(ROOT / f"data/eval/{sig}-real-contested.jsonl", "w") as fh:
        for r in contested:
            fh.write(json.dumps(r) + "\n")

    ok = [i for i in range(len(texts)) if all(votes[m][i] is not None for m in JURY)]
    pair = lambda a, b: (sum(votes[a][i] == votes[b][i] for i in ok) / max(1, len(ok)))
    print(f"  unanimous {len(gold)}  contested {len(contested)}  unresolved {dropped}"
          f"  unanswered {unanswered}")
    print(f"  pairwise agreement: " + "  ".join(
        f"{JURY[a][7:18]}~{JURY[b][7:18]}={pair(JURY[a],JURY[b]):.3f}"
        for a, b in [(0,1),(0,2),(1,2)]))
    print(f"  gold tier mix: {dict(collections.Counter(r['tier'] for r in gold))}")
    print(f"  -> {out}")
