"""Does extractive compression beat head-truncation at the same token budget?

The comparison is deliberately narrow: only prompts that ACTUALLY exceed the
budget can be affected, so scoring the whole set would dilute a real effect
into the noise of prompts the change never touched. Reports the long subset
separately and the full set for context.

Both arms see 256 tokens. The only difference is WHICH 256.
"""
import sys, json, collections
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from compress import compress

tag = sys.argv[1]
sig = sys.argv[2]
files = sys.argv[3].split(",")
rows = [r for f in files for r in load_jsonl(ROOT/"data/eval"/f)]

d = ROOT/"models"/tag
tok = AutoTokenizer.from_pretrained(str(d))
m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval()
order = [m.config.id2label[i] for i in range(m.config.num_labels)]

ntok = [len(tok(r["text"], truncation=False)["input_ids"]) for r in rows]
long = [i for i, n in enumerate(ntok) if n > 256]
print(f"{sig} / {tag}: n={len(rows)}  over 256 tokens: {len(long)} ({len(long)/len(rows):.1%})")
if not long:
    sys.exit("  nothing to compress")

def predict(texts):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i+32], truncation=True, max_length=256,
                      padding=True, return_tensors="pt")
            out += [order[j] for j in m(**enc).logits.argmax(-1).tolist()]
    return out

raw = [r["text"] for r in rows]
cnt = lambda s: len(tok(s, truncation=False)["input_ids"])
L = set(long)
comp = [compress(t, 250, count=cnt) if i in L else t for i, t in enumerate(raw)]
print(f"  compression changed {sum(1 for i in L if comp[i] != raw[i])}/{len(L)} long prompts")
p_raw, p_cmp = predict(raw), predict(comp)
y = [r["tier"] for r in rows]

def acc(pred, idx):
    return np.mean([pred[i] == y[i] for i in idx])
allidx = list(range(len(rows)))
print(f"  {'subset':<22}{'truncate':>10}{'compress':>10}{'delta':>9}")
for name, idx in (("long prompts only", long), ("full eval set", allidx)):
    a, b = acc(p_raw, idx), acc(p_cmp, idx)
    print(f"  {name+f' (n={len(idx)})':<22}{a:9.2%}{b:10.2%}{b-a:+9.2%}")
