import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from clfeval.specs import ClassifierTaskSpec, EvalSuite, PromotionPolicy


def test_gates_require_order():
    with pytest.raises(ValueError, match="ordered"):
        ClassifierTaskSpec(signal="t", labels=["A","B"], ordered=False,
                           gates=[{"at": "B"}])


def test_identical_folds_share_a_digest():
    """`route` and `cx2` shipped as separate models and were the same fold under
    two label vocabularies; error overlap was Jaccard 1.000."""
    t = ClassifierTaskSpec(signal="t", labels=["A","B","C"], folds={
        "x": {"A":"lo","B":"lo","C":"hi"}, "y": {"A":"lo","B":"lo","C":"hi"}})
    assert t.fold_digest("x") == t.fold_digest("y")


def test_builtin_tasks_load():
    for n in ("complexity","sensitivity","cost"):
        assert ClassifierTaskSpec.builtin(n).digest.startswith("sha256:")


def test_suites_load_and_resolve_controls():
    root = pathlib.Path(__file__).resolve().parents[1] / "clfeval/suites"
    for f in root.glob("*.yaml"):
        EvalSuite.load(f).active_controls()


def test_promotion_rejects_unreproducible():
    class R:
        verdict="QUALIFIED"; metrics={"lift_over_chance":0.3}
        controls={"blocking_failures":[],"warnings":[]}
        planes=[]; unpinned=["runtime_revision"]
    d = PromotionPolicy(require_reproducible=True).decide(R())
    assert d["decision"] in ("REJECT","INCOMPLETE")


def test_incomplete_is_not_reject():
    """A required plane that was never exercised is UNFINISHED, not failed.
    Collapsing the two either blocks unevaluated work or lets it pass as good."""
    from clfeval.specs import PromotionPolicy
    class R:
        verdict="INCOMPLETE"; metrics={"lift_over_chance":0.3}
        controls={"blocking_failures":[],"warnings":[]}
        planes=[{"plane":"runtime_quality","status":"NOT_EVALUATED"}]
        unpinned=[]
    d = PromotionPolicy(require_reproducible=False,
                        require_planes=["runtime_quality"]).decide(R())
    assert d["decision"] == "INCOMPLETE"


def test_unknown_runtime_identity_fails():
    """63.67% deployed vs 81.82% candidate raised 'which model IS that?' -- an
    unidentifiable artifact cannot be qualified."""
    from clfeval.controls import ArtifactIdentityControl, Status
    from clfeval.specs import ClassifierTaskSpec
    class Ad: plane="runtime"; revision="unknown"; taxonomy_revision_reported=None
    r = ArtifactIdentityControl().run({"adapter": Ad(), "metrics": {},
                                       "task": ClassifierTaskSpec(signal="t", labels=["A","B"])})
    assert r.status is Status.FAIL and "cannot be attributed" in r.detail


def test_coverage_unmeasured_fails():
    """A runtime plane without coverage cannot tell a fast classifier from a
    bypassed one."""
    from clfeval.controls import RuntimeSLOControl, Status
    from clfeval.specs import ClassifierTaskSpec
    r = RuntimeSLOControl().run({"runtime": {"p95_ms": 5.0}, "slo": {"max_p95_ms": 20},
                                 "metrics": {}, "task": ClassifierTaskSpec(signal="t", labels=["A","B"])})
    assert r.status is Status.FAIL and "not measured" in r.detail


def test_ledger_accepts_a_real_report_shape():
    """The ledger read `planes_evaluated` after the report moved to `planes`, and
    it crashed only when a real qualification tried to log. A sink that drifts
    from its report schema fails at the worst possible moment: after the work."""
    import inspect
    from clfeval.sinks import mlflow_ledger
    from clfeval.reports import QualificationReport
    src = inspect.getsource(mlflow_ledger)
    fields = set(QualificationReport.__dataclass_fields__)
    for attr in ("planes_evaluated", "planes_unevaluated"):
        assert attr not in src, f"ledger references removed field {attr}"
    for ref in ("report.planes", "report.verdict", "report.metrics"):
        assert ref in src
        assert ref.split(".", 1)[1] in fields


def test_fold_sums_probabilities_and_cancels_interior_uncertainty():
    """§130: a row split 0.45/0.45/0.10 is uncertain 3-way, confident once folded.

    This is the whole mechanism. Half the rows in a 3-way head's least-confident
    slice were rows the deployed binary gate was already correct on -- 100.0% of
    them, both seeds -- because their uncertainty lived entirely inside one folded
    class. If fold_probs ever stops summing, that regression is silent: accuracy
    is unchanged and only the abstention threshold reads nonsense.
    """
    import numpy as np
    from clfeval.specs.task import ClassifierTaskSpec
    spec = ClassifierTaskSpec(
        signal="triage3cx", labels=["HARD", "STANDARD", "TRIVIAL"],
        folds={"deployed": {"HARD": "WORK", "STANDARD": "WORK", "TRIVIAL": "TRIVIAL"}},
        deployment_fold="deployed")
    probs = np.array([[0.45, 0.45, 0.10]])
    assert probs.max() == 0.45                      # maximally uncertain 3-way
    sub, fp = spec.fold_probs("deployed", probs)
    assert sub.labels == ["TRIVIAL", "WORK"]
    assert fp[0, sub.labels.index("WORK")] == pytest.approx(0.90)
    assert fp.max() == pytest.approx(0.90)          # confident once deployed


def test_as_deployed_folds_labels_alongside_probs():
    """A fold that moves probs but not y_true scores every row against the wrong
    key -- and still returns a plausible-looking accuracy."""
    import numpy as np
    from clfeval.specs.task import ClassifierTaskSpec
    spec = ClassifierTaskSpec(
        signal="t", labels=["A", "B", "C"],
        folds={"d": {"A": "X", "B": "X", "C": "Y"}}, deployment_fold="d")
    sub, fp, y = spec.as_deployed(np.array([[0.5, 0.3, 0.2]]), ["B"])
    assert y == ["X"] and sub.labels == ["X", "Y"]


def test_no_deployment_fold_is_a_passthrough():
    """Most tasks declare no fold; as_deployed must not perturb them."""
    import numpy as np
    from clfeval.specs.task import ClassifierTaskSpec
    spec = ClassifierTaskSpec(signal="t", labels=["A", "B"])
    p = np.array([[0.7, 0.3]])
    s2, p2, y2 = spec.as_deployed(p, ["A"])
    assert s2 is spec and p2 is p and y2 == ["A"]


def test_deployment_fold_must_name_a_declared_fold():
    from clfeval.specs.task import ClassifierTaskSpec
    with pytest.raises(ValueError, match="not a declared fold"):
        ClassifierTaskSpec(signal="t", labels=["A", "B"], deployment_fold="ghost")
