"""Orchestration: model + domain spec -> full metric contract, controls, MLflow.

Nothing here knows what complexity, sensitivity or cost mean. A model adapter is
anything with `predict_proba(list[str]) -> np.ndarray` whose columns are ordered
to match `spec.labels`, so a team can evaluate a scikit-learn pipeline, an API
endpoint or a rules engine on the same footing as a fine-tuned encoder.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from ..metrics import classification as M
from .spec import DomainSpec


class HFAdapter:
    """Sequence-classification adapter. Reorders columns to the spec's label order.

    That reordering is not incidental. Two models in this project were trained on
    the same fold under different label vocabularies and were briefly treated as
    separate results; aligning to the spec makes such a duplicate detectable.
    """
    def __init__(self, path: str, spec: DomainSpec, maxlen: int = 256):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path).eval()
        native = [self.m.config.id2label[i] for i in range(self.m.config.num_labels)]
        missing = set(spec.labels) - set(native)
        if missing:
            raise ValueError(f"model emits {native}, spec expects {spec.labels}; "
                             f"missing {sorted(missing)}")
        self.idx = [native.index(l) for l in spec.labels]
        self.maxlen = maxlen

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 64):
                e = self.tok(texts[i:i+64], truncation=True, max_length=self.maxlen,
                             padding=True, return_tensors="pt")
                out.append(self.torch.softmax(self.m(**e).logits, -1).numpy()[:, self.idx])
        return np.concatenate(out)


def load_rows(spec: DomainSpec, root: pathlib.Path, roles=("eval", "contested")):
    rows = []
    for ds in spec.datasets:
        if ds.role not in roles: continue
        p = root / ds.path
        if not p.exists(): continue
        for line in open(p):
            r = json.loads(line)
            if r.get(spec.label_field) in spec.labels:
                rows.append({**r, "_dataset": ds.id, "_role": ds.role})
    return rows


def evaluate(adapter, spec: DomainSpec, rows: list[dict],
             seed_scores: list[float] | None = None,
             fold: str | None = None) -> tuple[dict, dict]:
    """Full contract + controls for one model on one domain (optionally folded)."""
    labels = spec.labels
    if fold:
        table = spec.folds[fold]
        labels = sorted(set(table.values()))
    texts = [r[spec.text_field] for r in rows]
    P = adapter.predict_proba(texts)

    if fold:
        table = spec.folds[fold]
        Pf = np.zeros((len(rows), len(labels)))
        for j, l in enumerate(spec.labels):
            Pf[:, labels.index(table[l])] += P[:, j]
        P = Pf
        y = [table[r[spec.label_field]] for r in rows]
        vrows = [{**r, spec.label_field: table[r[spec.label_field]],
                  spec.votes_field: [table[v] for v in r.get(spec.votes_field, [])
                                     if v in table]} for r in rows]
        fspec = DomainSpec(name=f"{spec.name}/{fold}", labels=labels,
                           ordered=spec.ordered, gates=[], noise_floor=spec.noise_floor)
    else:
        y = [r[spec.label_field] for r in rows]
        vrows, fspec = rows, spec

    pred = [labels[i] for i in P.argmax(1)]
    m = {}
    m.update(M.core(y, pred, labels))
    m.update(M.label_ceiling(vrows, pred, fspec))
    m.update(M.gates(y, P, fspec))
    m.update(M.risk_coverage(y, P, fspec))
    m.update(M.confidence_vs_disagreement(vrows, P, fspec))

    sep = next((d.separability_from_production for d in spec.datasets
                if d.role == "eval" and d.separability_from_production is not None), None)
    checks = [
        C.baseline_lift(m),
        C.seed_stability(seed_scores or [m["accuracy"]], spec.noise_floor),
        C.corpus_distribution(sep, None),
    ]
    if "selective/accuracy@0.90" in m:
        checks.append(C.holdout_selection(m["accuracy"], m["accuracy"]))
    return m, C.run_all(checks)
