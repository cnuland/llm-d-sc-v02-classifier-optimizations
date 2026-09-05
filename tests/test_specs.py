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
        planes_unevaluated=[]; unpinned=["runtime_revision"]
    d = PromotionPolicy(require_reproducible=True).decide(R())
    assert d["decision"] == "REJECT"
