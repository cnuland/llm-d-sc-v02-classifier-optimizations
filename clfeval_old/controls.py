"""Controls — the run-blocking checks, domain-agnostic.

Each control corresponds to a specific occasion where a naive evaluation on this
project gave the wrong answer. They are gates rather than metrics: a run that
fails one is tagged and cannot be promoted, because every one of these failures
produced a number that looked publishable.
"""
from __future__ import annotations
import numpy as np, collections


class Result(dict):
    def __init__(self, name, passed, detail, **kw):
        super().__init__(control=name, passed=bool(passed), detail=detail, **kw)


def baseline_lift(m: dict, min_lift=0.02, min_minority=0.06) -> Result:
    """A number without a baseline is not a result.

    Caught: a gate scoring 99.65% on an eval containing 2 positive rows, and a
    candidate taxonomy scoring 93.78% against a 93.45% majority baseline.
    """
    lift, minor = m.get("lift_over_chance", 0), m.get("minority_share", 0)
    if minor < min_minority:
        return Result("baseline_lift", False,
                      f"minority class is {minor:.1%} of the eval (<{min_minority:.0%}): "
                      f"accuracy is mostly free and the fold is degenerate",
                      lift=lift, minority_share=minor)
    if lift < min_lift:
        return Result("baseline_lift", False,
                      f"lift over chance is {lift:+.2%} (<{min_lift:.0%}): the model is "
                      f"barely beating 'always answer the majority class'",
                      lift=lift, minority_share=minor)
    return Result("baseline_lift", True, f"lift {lift:+.2%}, minority {minor:.1%}",
                  lift=lift, minority_share=minor)


def matched_operating_point(probs_a, probs_b, y_true, spec, gate=None,
                            targets=(0.85, 0.90, 0.95)) -> Result:
    """Compare two models at MATCHED containment, not at argmax.

    Seven of eight interventions on this project looked like wins at argmax and
    lost here. An argmax comparison on a gated ordered taxonomy compares two
    arbitrary operating points, not two models.
    """
    if not spec.gates:
        return Result("matched_operating_point", True, "not an ordered gated taxonomy")
    gate = gate or spec.gates[-1]
    rank, thr = spec.rank, spec.rank[gate]
    up = np.array([1.0 if rank[l] >= thr else 0.0 for l in spec.labels])
    hi = np.array([rank[a] >= thr for a in y_true])
    def curve(P):
        L = np.log(np.clip(P, 1e-9, 1)); pts = {}
        for tgt in targets:
            best = None
            for b in np.arange(-4, 8.01, 0.05):
                idx = (L + b*up).argmax(1)
                blk = np.array([rank[spec.labels[i]] >= thr for i in idx])
                if hi.any() and blk[hi].mean() >= tgt:
                    o = blk[~hi].mean()
                    if best is None or o < best: best = o
            pts[tgt] = best
        return pts
    ca, cb = curve(probs_a), curve(probs_b)
    wins = sum(1 for t in targets
               if ca[t] is not None and cb[t] is not None and ca[t] < cb[t])
    comparable = sum(1 for t in targets if ca[t] is not None and cb[t] is not None)
    detail = "  ".join(
        f"@{t:.0%}: {ca[t]:.2%} vs {cb[t]:.2%}" if ca[t] is not None and cb[t] is not None
        else f"@{t:.0%}: n/a" for t in targets)
    return Result("matched_operating_point", wins > comparable/2,
                  f"gate {gate} — candidate wins {wins}/{comparable} cells. {detail}",
                  cells_won=wins, cells=comparable)


def holdout_selection(fitted: float, heldout: float, tol=0.005) -> Result:
    """Any operating point chosen on the rows it is scored on is fitted.

    Caught: 98.14% fitted vs 97.89% held out on accuracy, and 8.87% vs 17.13% on
    over-block at high containment -- roughly 2x optimistic where it matters most.
    """
    gap = fitted - heldout
    return Result("holdout_selection", gap <= tol,
                  f"fitted {fitted:.2%} vs held out {heldout:.2%} (optimism {gap:+.2%}); "
                  f"report the held-out figure",
                  optimism=gap)


def seed_stability(scores: list[float], noise_floor: float | None) -> Result:
    """One seed is not a measurement.

    Caught: a '+0.42 gain' that vanished on the second seed, and a published
    figure that had to be corrected from best-of-seeds to median.
    """
    if len(scores) < 2:
        return Result("seed_stability", False,
                      f"only {len(scores)} seed(s); a difference cannot be "
                      f"distinguished from initialisation noise", n_seeds=len(scores))
    spread = max(scores) - min(scores)
    if noise_floor and spread > 3*noise_floor:
        return Result("seed_stability", False,
                      f"seed spread {spread:.4f} exceeds 3x the noise floor "
                      f"({noise_floor:.4f}): this configuration is unstable",
                      spread=spread)
    return Result("seed_stability", True,
                  f"{len(scores)} seeds, spread {spread:.4f}, median "
                  f"{sorted(scores)[len(scores)//2]:.4f}", spread=spread)


def corpus_distribution(separability: float | None, prior_mismatch: float | None) -> Result:
    """Is the eval evidence about the traffic being served?

    The single largest unresolved risk on this project: a signal validated
    entirely against data 95.8% distinguishable from real traffic. Every accuracy
    figure was correct and none of it was evidence about production.

    Prior mismatch is reported rather than gated -- resampling toward the eval
    prior paid above ~1.5x mismatch and LOST below it, because resampling always
    discards rows.
    """
    if separability is None:
        return Result("corpus_distribution", False,
                      "separability from production traffic is UNMEASURED — the "
                      "eval may not be evidence about the deployment",
                      separability=None, prior_mismatch=prior_mismatch)
    if separability > 0.90:
        return Result("corpus_distribution", False,
                      f"eval is {separability:.1%} distinguishable from production "
                      f"traffic: these metrics describe a different population",
                      separability=separability, prior_mismatch=prior_mismatch)
    return Result("corpus_distribution", True,
                  f"separability {separability:.1%}"
                  + (f", prior mismatch {prior_mismatch:.2f}x" if prior_mismatch else ""),
                  separability=separability, prior_mismatch=prior_mismatch)


def judge_integrity(judge_is_contestant: bool, position_randomised: bool,
                    longer_answer_win_rate: float | None,
                    second_judge_agrees_in_sign: bool | None) -> Result:
    """LLM-judged comparisons need controls or they measure the wrong thing.

    Caught: a headline finding retracted after the judge was shown to pick the
    longer answer in 70.2% of decided pairs, and a second judge reversed the
    result.
    """
    fails = []
    if judge_is_contestant: fails.append("judge is also a contestant")
    if not position_randomised: fails.append("answer position not randomised")
    if longer_answer_win_rate is not None and longer_answer_win_rate > 0.60:
        fails.append(f"longer answer wins {longer_answer_win_rate:.1%} (>60%): "
                     f"the judge is substantially scoring length")
    if second_judge_agrees_in_sign is False:
        fails.append("a second judge disagrees in sign")
    return Result("judge_integrity", not fails,
                  "; ".join(fails) if fails else "position randomised, judge neutral, "
                  "length bias within tolerance")


def run_all(checks: list[Result]) -> dict:
    failed = [c["control"] for c in checks if not c["passed"]]
    return {"controls_passed": not failed, "controls_failed": failed, "checks": checks}
