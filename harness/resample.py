"""Class-balanced resampling of a training corpus.

§57 showed a stronger encoder carries the INTERNAL/CONFIDENTIAL distinction that
MiniLM misses -- but only when the classes were BALANCED. §50 concluded "bigger
is worse" from fine-tuning on the raw skewed corpus, where the extra capacity
went to the majority class. Those two facts are only reconcilable by testing the
big encoder on a balanced corpus, which needs the corpus to exist.

Class WEIGHTS and class BALANCE are not the same intervention. Weights rescale
the gradient of a rare example; balance changes how often the model SEES one,
which is what governs representation learning in the encoder layers rather than
just the head.

Balances to the median class count -- downsample above, duplicate below --
so total size stays comparable and neither extreme dominates the schedule.
"""
import sys, json, collections, random, pathlib

out = sys.argv[1]
files = sys.argv[2].split(",")
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = random.Random(seed)

by = collections.defaultdict(list)
for f in files:
    for l in open(f):
        r = json.loads(l)
        by[r.get("tier") or r.get("label")].append(l.rstrip("\n"))

counts = sorted(len(v) for v in by.values())
target = counts[len(counts)//2]
rows = []
for t, v in sorted(by.items()):
    if len(v) >= target:
        pick = rng.sample(v, target)
    else:
        pick = list(v) + [rng.choice(v) for _ in range(target - len(v))]
    rows += pick
    print(f"  {t:<15} {len(v):6d} -> {len(pick):6d}")
rng.shuffle(rows)
pathlib.Path(out).write_text("\n".join(rows) + "\n")
print(f"  target/class={target}  total={len(rows)} -> {out}")
