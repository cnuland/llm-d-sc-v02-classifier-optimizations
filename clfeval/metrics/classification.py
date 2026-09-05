"""The metric contract — domain-agnostic, and complete by construction.

A run that emits only accuracy is rejected. That is the whole design. On this
project accuracy alone reordered the results table three separate times and once
recommended a model that releases 14.88% of live secrets over one that releases
none, so `accuracy` is computed here as one field among many rather than as the
answer.

Nothing below knows what a "tier" means. Everything domain-specific arrives
through DomainSpec.
"""
from __future__ import annotations
import collections
import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))


def core(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Scale-aware core metrics. `lift_over_chance` is the one to read."""
    n = len(y_true)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    acc = correct / max(1, n)
    counts = collections.Counter(y_true)
    majority = max(counts.values()) / max(1, n) if counts else 0.0
    minority = min(counts.values()) / max(1, n) if counts else 0.0
    lo, hi = wilson(correct, n)
    out = {
        "accuracy": acc,
        "majority_baseline": majority,
        "lift_over_chance": acc - majority,
        "minority_share": minority,
        "n": float(n),
        "wilson95_lo": lo, "wilson95_hi": hi,
    }
    f1s = []
    for l in labels:
        sup = counts.get(l, 0)
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == b == l)
        fp = sum(1 for a, b in zip(y_true, y_pred) if b == l and a != l)
        rec = tp / sup if sup else float("nan")
        pre = tp / (tp + fp) if (tp + fp) else float("nan")
        out[f"recall/{l}"] = rec
        out[f"precision/{l}"] = pre
        out[f"support/{l}"] = float(sup)
        if sup and (tp + fp):
            f1s.append(0.0 if (rec + pre) == 0 else 2*rec*pre/(rec+pre))
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return out


def label_ceiling(rows: list[dict], y_pred: list[str], spec) -> dict:
    """Is the ceiling the LABELS or the MODEL?

    Split by whether the jury was unanimous. This diagnostic gave OPPOSITE answers
    on two signals in the same corpus -- one rubric-limited, one model-limited --
    and it decides whether a relabelling campaign is worth funding. It costs
    nothing beyond vote data you already have.
    """
    v = spec.votes_field; lab = spec.label_field
    have = [(r, p) for r, p in zip(rows, y_pred) if isinstance(r.get(v), list) and r[v]]
    if not have:
        return {"labels/status": float("nan")}
    un = [(r, p) for r, p in have if len(set(r[v])) == 1]
    co = [(r, p) for r, p in have if len(set(r[v])) > 1]
    cu = sum(1 for r, p in un if p == r[lab])
    cc = sum(1 for r, p in co if p == r[lab])
    n = len(have)
    return {
        "labels/jury_agreement": len(un) / n,
        "labels/accuracy_unanimous": cu / max(1, len(un)),
        "labels/accuracy_contested": cc / max(1, len(co)),
        "labels/contested_share": len(co) / n,
        # what the model would score if every contested row were labelled
        # perfectly AND it got them all right: an upper bound on the entire
        # label-quality axis, computable before spending anything on it
        "labels/ceiling_from_perfect_labels": (cu + len(co)) / n,
        "labels/error_on_unanimous": (len(un) - cu) / n,
    }


def gates(y_true: list[str], probs: np.ndarray, spec) -> dict:
    """Containment / over-block per gate. Ordered taxonomies only.

    Reported at argmax here; `controls.matched_operating_point` is what makes two
    models comparable, because an argmax comparison on a gated taxonomy compares
    two arbitrary operating points rather than two models -- which overturned
    seven of eight apparent wins on this project.
    """
    if not spec.gates: return {}
    rank = spec.rank
    pred = [spec.labels[i] for i in probs.argmax(1)]
    out = {}
    # Gates carry an action and an optional target now (specs.task.Gate); this
    # module predates that and accepted bare label strings. Accept both so a task
    # spec written either way qualifies identically.
    for gate in spec.gates:
        g = gate if isinstance(gate, str) else gate.at
        thr = rank[g]
        hi = np.array([rank[a] >= thr for a in y_true])
        blk = np.array([rank[a] >= thr for a in pred])
        if hi.any():
            out[f"gate/{g}/containment"] = float(blk[hi].mean())
        if (~hi).any():
            out[f"gate/{g}/over_block"] = float(blk[~hi].mean())
        # the failure a security owner cares about: at-or-above content emitted
        # to the LOWEST tier, i.e. released rather than merely mis-tiered
        if hi.any():
            out[f"gate/{g}/released_to_lowest"] = float(
                np.mean([p == spec.labels[0] for p, h in zip(pred, hi) if h]))
    return out


def risk_coverage(y_true: list[str], probs: np.ndarray, spec,
                  coverages=(0.95, 0.90, 0.80, 0.70), repeats=40) -> dict:
    """Held-out abstention curve, scored in the DEPLOYED label space.

    Held out because the abstention threshold is an operating point, and choosing
    one on the rows you then score was worth about a point of optimism on accuracy
    and roughly 2x on over-block at high containment.

    Deployed because confidence is what the threshold reads, and a taxonomy finer
    than the routing decision puts uncertainty on boundaries the router discards.
    On a 3-way head deployed as a 2-way gate, half the rows in the least-confident
    5% were rows the deployed gate was already correct on -- 100.0% of them, on
    both seeds -- because their uncertainty was entirely interior to one folded
    class. Thresholding raw softmax.max() there cost 2.3-3.0 points of kept
    accuracy at 80% coverage. `spec.deployment_fold` selects the space; when a
    task declares none, this is the trained space and nothing changes.

    Also reports `waste`: the share of escalated rows the gate had ALREADY got
    right. Kept-accuracy alone cannot distinguish a threshold that finds real
    errors from one that escalates at random, and escalation is not free.
    """
    spec, probs, y_true = spec.as_deployed(probs, y_true)
    pred = np.array([spec.labels[i] for i in probs.argmax(1)])
    conf = probs.max(1)
    correct = (pred == np.array(y_true))
    rng = np.random.default_rng(0)
    out = {}
    for cov in coverages:
        held, wasted = [], []
        for _ in range(repeats):
            idx = rng.permutation(len(correct))
            for a, b in ((idx[:len(idx)//2], idx[len(idx)//2:]),
                         (idx[len(idx)//2:], idx[:len(idx)//2])):
                t = np.quantile(conf[a], 1-cov)
                keep = conf[b] >= t
                if keep.sum(): held.append(correct[b][keep].mean())
                if (~keep).sum(): wasted.append(correct[b][~keep].mean())
        out[f"selective/accuracy@{cov:.2f}"] = float(np.mean(held)) if held else float("nan")
        out[f"selective/waste@{cov:.2f}"] = float(np.mean(wasted)) if wasted else float("nan")
    return out


def confidence_vs_disagreement(rows: list[dict], probs: np.ndarray, spec,
                               coverage=0.90) -> dict:
    """Does abstention drop the rows the JURY could not agree on?

    Enrichment near 1.0 means confidence is BLIND to disagreement -- the model is
    confidently wrong on hard rows rather than uncertain about them, and no
    threshold rescues it. That distinction explained why one published gate had
    the worst abstention curve in its family despite the second-highest accuracy.

    NOT A SELECTION CRITERION. Across three arms of one matched experiment it
    ranked them in exactly the reverse of their deployed kept-accuracy: the arm
    with the best enrichment (2.0x) had the worst accuracy at 80% coverage, and
    the arm with the worst enrichment (1.0x) beat it by 2.3 points. Enrichment
    rewards a head for being uncertain without asking whether the uncertainty is
    about anything the deployment acts on. Read it as a diagnostic for WHY an
    abstention curve is shaped as it is, and rank arms on `selective/accuracy`
    and `selective/waste`.

    Scored in the deployed space for the same reason risk_coverage is -- an
    enrichment computed on a taxonomy the router never sees describes a threshold
    nobody applies.
    """
    v = spec.votes_field
    flags = np.array([1.0 if isinstance(r.get(v), list) and len(set(r[v])) > 1 else 0.0
                      for r in rows])
    if flags.sum() == 0: return {}
    spec, probs, _ = spec.as_deployed(probs)
    conf = probs.max(1)
    t = np.quantile(conf, 1-coverage)
    dropped = conf < t
    if dropped.sum() == 0: return {}
    base = flags.mean()
    return {
        "selective/contested_base_rate": float(base),
        "selective/contested_in_dropped": float(flags[dropped].mean()),
        "selective/contested_enrichment": float(flags[dropped].mean() / max(1e-9, base)),
    }
