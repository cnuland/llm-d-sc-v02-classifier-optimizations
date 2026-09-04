"""Build a SELF-CONSISTENCY gold: one juror, three samples, majority vote.

§83 measured a 27.7-point gap between how consistent a juror is with itself
(80.9%) and how consistent the three jurors are with each other (53.2%). Two
things follow, and this script tests both.

1. Voting over repeated samples from ONE juror removes intra-juror sampling
   noise, which §83 showed is currently baked into the panel gold: 10% of rows
   the panel called unanimous are unanimous only by luck of the draw.

2. A single coherent rater is a different target from a three-model consensus.
   Accuracy against it is NOT the same claim -- it measures "does the model
   reproduce this rater" rather than "does the model reproduce the consensus" --
   so both are reported side by side and neither is presented as the number.

The comparison that matters is where the two golds DISAGREE: those rows are
where consensus and coherence pull apart, and they are the ones any headline
accuracy figure is silently taking a position on.
"""
import sys, json, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from sdg import blind_relabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SIG   = sys.argv[1] if len(sys.argv) > 1 else "complexity"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-5"
TAG   = sys.argv[3] if len(sys.argv) > 3 else "cx-v-resolved-seed11"
K     = 3
labels = load_taxonomy(SIG)["labels"]

rows = []
for f in (f"{SIG}-real-gold.jsonl", f"{SIG}-real-contested.jsonl"):
    p = ROOT/"data/eval"/f
    if p.exists():
        rows += [dict(r, _src=("gold" if "contested" not in f else "contested"))
                 for r in load_jsonl(p) if r.get("votes")]
texts = [r["text"] for r in rows]
print(f"{SIG}: {len(rows)} eval rows  ({collections.Counter(r['_src'] for r in rows)})")

passes = []
for k in range(K):
    passes.append([x["label"] for x in blind_relabel(SIG, labels, texts, model=MODEL,
                   batch=20, effort="low", pass_tag=f"scgold-{k}")])
    print(f"  pass {k+1}/{K}", flush=True)

sc, unstable = [], 0
for i in range(len(rows)):
    v = [p[i] for p in passes if p[i]]
    if not v: sc.append(None); continue
    c = collections.Counter(v).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]: unstable += 1      # no majority
    sc.append(c[0][0])
keep = [i for i in range(len(rows)) if sc[i]]
print(f"  majority formed for {len(keep)}/{len(rows)}; {unstable} rows had no clear majority")

panel = [r["tier"] for r in rows]
same = sum(1 for i in keep if sc[i] == panel[i])
print(f"  self-consistency gold agrees with PANEL gold on {same}/{len(keep)} = {same/len(keep):.1%}")
diff = collections.Counter(f"{panel[i]}->{sc[i]}" for i in keep if sc[i] != panel[i])
print(f"  where they differ: {diff.most_common(6)}")

d = ROOT/"models"/TAG
tok = AutoTokenizer.from_pretrained(str(d))
m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
order = [m.config.id2label[i] for i in range(m.config.num_labels)]
pred = []
with torch.no_grad():
    for i in range(0, len(texts), 64):
        e = tok(texts[i:i+64], truncation=True, max_length=256, padding=True, return_tensors="pt")
        pred += [order[j] for j in m(**e).logits.argmax(-1).tolist()]

print(f"\n  model {TAG}")
for nm, ref in (("vs PANEL gold (3 models, 1 sample each)", panel),
                ("vs SELF-CONSISTENCY gold (1 model, 3 samples)", sc)):
    ix = [i for i in keep if ref[i]]
    print(f"    {nm:<48} n={len(ix):4d}  acc={np.mean([pred[i]==ref[i] for i in ix]):.4f}")
agree_ix = [i for i in keep if sc[i] == panel[i]]
print(f"    {'vs rows where BOTH golds agree':<48} n={len(agree_ix):4d}  "
      f"acc={np.mean([pred[i]==panel[i] for i in agree_ix]):.4f}")
out = ROOT/"data/eval"/f"{SIG}-selfconsistency-gold.jsonl"
with open(out, "w") as fh:
    for i in keep:
        fh.write(json.dumps({"text": rows[i]["text"], "tier": sc[i],
                             "panel_tier": panel[i], "votes": [p[i] for p in passes],
                             "src": "selfconsistency"}) + "\n")
print(f"  -> {out}")
