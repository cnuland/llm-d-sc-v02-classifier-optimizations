"""Shared evaluation kit for llm-d-sc classifier accuracy work.

The runtime we must match is llm-d-sc's `anchor-topk-mean`:
  embed(text) -> masked mean-pool -> cosine against every anchor vector
  -> per-label mean of that label's top_k best anchors -> argmax.

Anchors in llm-d-sc are TEXT (taxonomy.rs: `anchors: BTreeMap<String, Vec<String>>`),
embedded at load time, so any accuracy gain that requires precomputed prototype
vectors or a softmax head is a RUNTIME CHANGE, not a config change. We measure
both anyway, because the size of that gap is the argument for making the change.
"""
from __future__ import annotations
import json, math, os, pathlib, hashlib
import numpy as np

GENESIS = pathlib.Path("/Users/cnuland/llm-d-sc-genesis")
ROOT = pathlib.Path("/Users/cnuland/llm-d-sc-accuracy")
SIGNALS = ("complexity", "cost", "sensitivity")


def load_taxonomy(signal: str) -> dict:
    return json.loads((GENESIS / "classifiers" / f"{signal}.json").read_text())


def load_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def heldout(signal: str) -> list[dict]:
    return load_jsonl(GENESIS / "evals" / "datasets" / f"{signal}-heldout.jsonl")


_ENCODERS: dict[str, object] = {}


def encoder(model_id: str, revision: str | None = None):
    """Cache SentenceTransformer instances; they are expensive to build."""
    key = f"{model_id}@{revision}"
    if key not in _ENCODERS:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _ENCODERS[key] = SentenceTransformer(model_id, revision=revision, device=dev)
    return _ENCODERS[key]


CACHE = ROOT / "data" / ".embcache"


def embed(model_id, texts, revision=None, batch_size=64, normalize=False, cache=True):
    """Embed `texts`, memoised on disk.

    Experiments re-embed the same corpora dozens of times while only the
    decision rule changes; the cache key covers the model, revision and the
    exact text list so a stale hit is not possible.
    """
    texts = list(texts)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        h = hashlib.blake2b(
            json.dumps([model_id, revision, normalize, texts]).encode(), digest_size=16
        ).hexdigest()
        f = CACHE / f"{h}.npy"
        if f.exists():
            return np.load(f)
    m = encoder(model_id, revision)
    v = m.encode(texts, batch_size=batch_size, show_progress_bar=False,
                 convert_to_numpy=True, normalize_embeddings=normalize)
    if cache:
        np.save(f, v)
    return v


def unit(a: np.ndarray) -> np.ndarray:
    """L2-normalise rows; cosine then reduces to a dot product."""
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n == 0, 1, n)


_unit = unit


def anchor_topk_predict(qvecs, anchor_vecs, anchor_labels, labels, top_k=3):
    """Exactly llm-d-sc's scoring rule. Returns (preds, score_matrix)."""
    q, a = _unit(np.asarray(qvecs, np.float64)), _unit(np.asarray(anchor_vecs, np.float64))
    sims = q @ a.T                                     # (n_query, n_anchor)
    scores = np.empty((len(q), len(labels)))
    for j, lab in enumerate(labels):
        cols = sims[:, [i for i, L in enumerate(anchor_labels) if L == lab]]
        k = min(top_k, cols.shape[1])
        # top_k highest sims for this label, averaged
        scores[:, j] = np.sort(cols, axis=1)[:, -k:].mean(axis=1)
    return [labels[i] for i in scores.argmax(1)], scores


def metrics(truth, pred, labels, hard_flags=None) -> dict:
    truth, pred = list(truth), list(pred)
    n = len(truth)
    acc = sum(t == p for t, p in zip(truth, pred)) / n
    f1s = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(truth, pred))
        fp = sum(t != lab and p == lab for t, p in zip(truth, pred))
        fn = sum(t == lab and p != lab for t, p in zip(truth, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"n": n, "accuracy": acc, "macro_f1": sum(f1s.values()) / len(labels),
           "per_label_f1": f1s, "wilson95": wilson(acc, n)}
    if hard_flags is not None:
        hi = [i for i, h in enumerate(hard_flags) if h]
        if hi:
            out["hard_accuracy"] = sum(truth[i] == pred[i] for i in hi) / len(hi)
            out["hard_n"] = len(hi)
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truth, pred):
        cm[t][p] += 1
    out["confusion"] = cm
    return out


def wilson(p, n, z=1.96):
    """95% CI for a proportion. At n=75 this is +/-7pt, which is the whole
    reason the current held-out sets cannot certify a 'high 90s' claim."""
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def fmt(sig, arm, m):
    lo, hi = m["wilson95"]
    hard = f"  hard={m['hard_accuracy']:.3f}" if "hard_accuracy" in m else ""
    return (f"  {sig:<12} {arm:<28} n={m['n']:<4} acc={m['accuracy']:.4f} "
            f"[{lo:.3f},{hi:.3f}]  macroF1={m['macro_f1']:.4f}{hard}")
