"""Tests for the controls, written as REGRESSIONS against real mistakes.

Each test names the incident it prevents. A control that silently stops firing is
worse than no control, and these are the specific failures that produced numbers
which looked publishable.
"""
import sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from clfeval.controls import (BaselineControl, SeedStabilityControl, HoldoutControl,
                              TrafficAlignmentControl, RuntimeSLOControl,
                              CorpusImmutabilityControl, JudgeIntegrityControl, Status)
from clfeval.specs import ClassifierTaskSpec

TASK = ClassifierTaskSpec(signal="t", labels=["A", "B"], noise_floor=0.002)


def _ctx(**kw):
    base = {"metrics": {}, "task": TASK, "seed_scores": [0.9, 0.9],
            "dataset_manifest": [{"id": "d", "digest": "sha256:abc"}]}
    return {**base, **kw}


def test_degenerate_split_fails():
    """A gate scored 99.65% on an eval holding 2 positive rows."""
    r = BaselineControl().run(_ctx(metrics={"lift_over_chance": 0.003,
                                            "minority_share": 0.004}))
    assert r.status is Status.FAIL and "degenerate" in r.detail


def test_lift_below_baseline_fails():
    """A candidate scored 93.78% against a 93.45% majority baseline."""
    r = BaselineControl().run(_ctx(metrics={"lift_over_chance": 0.003,
                                            "minority_share": 0.3}))
    assert r.status is Status.FAIL


def test_single_seed_fails():
    """A '+0.42 gain' vanished on the second seed."""
    assert SeedStabilityControl().run(_ctx(seed_scores=[0.9])).status is Status.FAIL


def test_inherited_floor_is_flagged():
    """The floor was measured once on one config and carried across all others,
    inflating every 'Nx the noise floor' claim by about 4x."""
    r = SeedStabilityControl().run(_ctx(seed_scores=[0.90, 0.9025],
                                        noise_floor=0.002,
                                        floor_measured_on="other-config",
                                        config_id="this-config"))
    assert "property of a config" in r.detail


def test_no_floor_warns_not_passes():
    """Omitting the floor must not be the easy way to pass stability."""
    t = ClassifierTaskSpec(signal="t", labels=["A", "B"])
    r = SeedStabilityControl().run({**_ctx(), "task": t, "seed_scores": [0.9, 0.91]})
    assert r.status is Status.WARN


def test_fitted_operating_point_fails():
    """Threshold tuning read 98.14% fitted and 97.89% held out; over-block read
    8.87% fitted and 17.13% held out."""
    r = HoldoutControl().run(_ctx(metrics={"threshold/fitted_accuracy": 0.9814,
                                           "threshold/heldout_accuracy": 0.9689}))
    assert r.status is Status.FAIL


def test_offdistribution_corpus_fails():
    """A signal was qualified entirely against data 95.8% distinguishable from
    the traffic it would serve."""
    r = TrafficAlignmentControl().run(_ctx(traffic={"separability": 0.958}))
    assert r.status is Status.FAIL and r.evidence["transfer_confidence"] == "LOW"


def test_unmeasured_alignment_warns_not_passes():
    r = TrafficAlignmentControl().run(_ctx(traffic={}))
    assert r.status is Status.WARN


def test_coverage_gates_before_latency():
    """A classifier scoring ~96% offline sat in CrashLoopBackOff for 32 hours.
    Latency percentiles over an empty sample look excellent."""
    r = RuntimeSLOControl().run(_ctx(runtime={"classification_coverage": 0.0,
                                              "p95_ms": 1.0, "p99_ms": 1.0},
                                     slo={"max_p95_ms": 20}))
    assert r.status is Status.FAIL and "coverage" in r.detail


def test_unpinned_corpus_fails():
    """A relabel overwrote 59,582 labels in place, unrecoverably."""
    r = CorpusImmutabilityControl().run(
        _ctx(dataset_manifest=[{"id": "d", "digest": "sha256:MISSING"}]))
    assert r.status is Status.FAIL


def test_length_biased_judge_fails():
    """A headline finding was retracted: the judge picked the longer answer in
    70.2% of decided pairs."""
    r = JudgeIntegrityControl().run(_ctx(judge={
        "judge_is_contestant": False, "position_randomised": True,
        "longer_answer_win_rate": 0.702}))
    assert r.status is Status.FAIL and "scoring length" in r.detail


def test_not_applicable_is_not_pass():
    """'Unmeasured' must never render as 'passed'."""
    r = JudgeIntegrityControl().run(_ctx())
    assert r.status is Status.NOT_APPLICABLE and not r.blocks_promotion


def _curve(**pts):
    return {f"selective/accuracy@{c}": v for c, v in pts.items()}


def test_non_monotone_risk_coverage_fails():
    """§132's real curve. The gate scored 88.41% -- indistinguishable from the
    folded alternative's 88.95% -- while its accuracy PEAKED at 60% coverage and
    fell to 91.57% at 30%, worse than keeping 70%. Accuracy, macro-F1 and ECE all
    looked normal. Only the shape of the curve showed it."""
    from clfeval.controls.quality import ConfidenceOrderingControl
    r = ConfidenceOrderingControl().run({"metrics": _curve(**{
        "0.90": 0.9133, "0.80": 0.9320, "0.70": 0.9534,
        "0.60": 0.9577, "0.50": 0.9493, "0.40": 0.9367, "0.30": 0.9157})})
    assert r.status is Status.FAIL
    assert "NON-MONOTONE" in r.detail


def test_monotone_risk_coverage_passes():
    """The folded 4-way head from the same experiment, same rows."""
    from clfeval.controls.quality import ConfidenceOrderingControl
    r = ConfidenceOrderingControl().run({"metrics": _curve(**{
        "0.90": 0.9234, "0.80": 0.9524, "0.70": 0.9715,
        "0.60": 0.9970, "0.50": 0.9964, "0.40": 1.0, "0.30": 1.0}) |
        {"calibration/confident_half_spread": 0.19}})
    assert r.status is Status.PASS


def test_tiny_dip_is_not_a_failure():
    """Held-out thresholds resample; a curve that wobbles 0.2% between adjacent
    coverages has not inverted, and failing it would make the control unusable."""
    from clfeval.controls.quality import ConfidenceOrderingControl
    r = ConfidenceOrderingControl().run({"metrics": _curve(**{
        "0.90": 0.920, "0.80": 0.940, "0.70": 0.938, "0.60": 0.960}) |
        {"calibration/confident_half_spread": 0.19}})
    assert r.status is Status.PASS


def test_narrow_confidence_range_alone_does_not_warn():
    """§133 retracts the saturation warning. The audited arm with the NARROWEST
    confident-half spread (0.004, genlen) has a perfect monotone curve to 100% on
    both seeds, while every arm that actually inverts has a WIDER spread
    (0.017-0.053). Warning on narrowness fires on the cleanest gates we have."""
    from clfeval.controls.quality import ConfidenceOrderingControl
    r = ConfidenceOrderingControl().run({"metrics": _curve(**{
        "0.90": 0.961, "0.80": 0.977, "0.70": 0.992, "0.60": 0.994,
        "0.50": 0.996, "0.40": 0.995, "0.30": 1.0}) |
        {"calibration/confident_half_spread": 0.004}})
    assert r.status is Status.PASS
    assert "0.0040" in r.detail          # still reported, just not a verdict


def test_confidence_ordering_is_blocking():
    """A router that cannot order its own confidence cannot be given an escalation
    policy, which is the operational point of a gate. This must not be advisory."""
    from clfeval.controls.quality import ConfidenceOrderingControl
    assert ConfidenceOrderingControl().blocking is True


def test_confident_half_spread_measures_the_top_half():
    import numpy as np
    from clfeval.metrics.calibration import confident_half_spread
    # bottom half spans widely; the top half is pinned in a 0.01 band
    P = np.array([[0.5, 0.5], [0.6, 0.4], [0.95, 0.05], [0.96, 0.04]])
    s = confident_half_spread(P)["calibration/confident_half_spread"]
    assert s == pytest.approx(0.01, abs=1e-6)
