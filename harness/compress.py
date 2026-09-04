"""Extractive prompt compression as a DENOISING filter for classification.

From "98x Faster LLM Routing Without a Dedicated GPU" (arXiv 2603.12646), which
reports compression RAISING classifier accuracy, not merely speed: domain
classification 53.1% -> 61.2%, PII detection 78.5% -> 92.4%. The mechanism they
credit is "lost in the middle" -- dropping irrelevant sentences concentrates the
signal inside the encoder's effective attention span.

That matters here for a reason specific to this project: the shipped models
truncate at 256 tokens, so a long prompt is currently compressed by the crudest
possible rule -- keep the first 256 tokens, discard everything after. A prompt
whose only sensitive span sits in paragraph six is classified on text that never
contained it. Extractive selection is a strictly better use of the same budget.

Four classical signals, weights from the paper (alpha = .20/.40/.35/.05), no
neural inference so it stays cheap on the CPU serving path:
  TextRank   PageRank over TF-IDF sentence similarity
  position   U-shaped -- openings and closings carry framing
  TF-IDF     information density
  novelty    inverse centrality, to keep outlier sentences a summary would drop

Selected sentences are re-emitted IN ORIGINAL ORDER, so the text stays readable
to an encoder trained on prose.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")

def split_sentences(text):
    parts = [s.strip() for s in _SENT.split(text) if s.strip()]
    return parts or [text.strip()]

def _textrank(S, d=0.85, iters=30):
    n = S.shape[0]
    W = S.copy()
    np.fill_diagonal(W, 0.0)
    rs = W.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    P = W / rs
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = (1 - d) / n + d * (P.T @ r)
    return r

def compress(text, budget_tokens=256, alpha=(0.20, 0.40, 0.35, 0.05), count=None):
    """Keep the highest-scoring sentences that fit in budget_tokens.

    `count` must be the SAME tokenizer the classifier uses. Budgeting in
    whitespace words instead was a real bug: these prompts run 3.34 tokens per
    word (they are dense with keys, IDs and code), so a word budget of 256
    triggered on only 20 of 148 over-length prompts and the experiment silently
    measured almost nothing.
    """
    count = count or (lambda s: len(s.split()))
    sents = split_sentences(text)
    if len(sents) < 3 or count(text) <= budget_tokens:
        return text
    try:
        X = TfidfVectorizer(stop_words="english", max_features=20000).fit_transform(sents)
    except ValueError:
        return text
    if X.shape[1] == 0:
        return text
    Xn = X.toarray()
    norm = np.linalg.norm(Xn, axis=1, keepdims=True); norm[norm == 0] = 1.0
    Xn = Xn / norm
    S = Xn @ Xn.T

    n = len(sents)
    tr = _textrank(S)
    idx = np.arange(n) / max(1, n - 1)
    pos = 1.0 - 4.0 * (idx - 0.5) ** 2          # U-shape: ends weighted above middle
    pos = 0.5 + 0.5 * (1.0 - pos)               # invert so ends score high
    dens = Xn.sum(1)
    cent = S.mean(1)
    nov = 1.0 - (cent - cent.min()) / (np.ptp(cent) or 1.0)

    z = lambda v: (v - v.mean()) / (v.std() or 1.0)
    score = alpha[0]*z(tr) + alpha[1]*z(pos) + alpha[2]*z(dens) + alpha[3]*z(nov)

    order = np.argsort(-score)
    keep, used = [], 0
    for i in order:
        w = count(sents[i])
        if used + w > budget_tokens and keep:
            continue
        keep.append(i); used += w
        if used >= budget_tokens:
            break
    return " ".join(sents[i] for i in sorted(keep))
