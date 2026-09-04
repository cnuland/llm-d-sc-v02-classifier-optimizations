"""Span-max inference: the tier of a request is the MAX over its parts.

Grounded in the actual residual errors, not in theory. Of the 25 NEVER_EGRESS
prompts the classifier fails to contain, only 2 contain a credential pattern.
The rest are attorney-client privileged emails QUOTED inside a casual user
framing:

    "just summarise this thanks  \"Anil - quick one, off the back of your
     voicemail. You do not need to answer the customer's...\""

The tier is a property of the quoted span. The framing is INTERNAL at most, it
is first in the sequence, and it is what the pooled representation is dominated
by -- so the model reads the wrapper and misses the payload. That is a
composition problem, not a capacity problem, and it does not need retraining:
score the parts and take the highest tier.

Segmentation is deliberately dumb (quoted blocks, then paragraphs) because a
learned segmenter would be another thing to validate. Whole-prompt score is
always included as one candidate, so span-max can only add evidence.
"""
import re, sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import torch
from evalkit import load_taxonomy

_QUOTE = re.compile(r'"([^"]{80,})"|“([^”]{80,})”', re.S)

def segments(text, min_chars=80, max_segs=6):
    segs = []
    for m in _QUOTE.finditer(text):
        segs.append((m.group(1) or m.group(2)).strip())
    if not segs:
        segs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= min_chars]
    segs = [s for s in segs if len(s) >= min_chars][:max_segs]
    return [text] + segs          # whole prompt always competes

def predict_spanmax(model, tok, order, texts, rank, maxlen=256, bs=64):
    flat, owner = [], []
    for i, t in enumerate(texts):
        for s in segments(t):
            flat.append(s); owner.append(i)
    preds = []
    with torch.no_grad():
        for i in range(0, len(flat), bs):
            enc = tok(flat[i:i+bs], truncation=True, max_length=maxlen,
                      padding=True, return_tensors="pt")
            preds += [order[j] for j in model(**enc).logits.argmax(-1).tolist()]
    best = [None]*len(texts)
    for p, o in zip(preds, owner):
        if best[o] is None or rank[p] > rank[best[o]]:
            best[o] = p
    return best, len(flat)
