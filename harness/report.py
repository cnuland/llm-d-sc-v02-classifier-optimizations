"""Aggregate every trained arm into one comparison table.

Sorted by REAL-TRAFFIC accuracy, not by the legacy held-out number. Ranking by
the legacy set is exactly what produced a shipped sensitivity model scoring
0.8933 there and 0.2903 on enterprise text.
"""
import json, sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import (ROOT, SIGNALS, load_taxonomy, load_jsonl, heldout, metrics,
                     embed, anchor_topk_predict)

SHIPPED = {"complexity": "cnuland/llm-d-sc-complexity",
           "cost": "cnuland/llm-d-sc-cost",
           "sensitivity": "cnuland/llm-d-sc-sensitivity"}
COLS = ["real-gold", "enterprise-gold", "heldout-v1", "real-contested"]


def evalsets(sig):
    out = [("heldout-v1", heldout(sig))]
    for nm in ("real-gold", "enterprise-gold", "real-contested"):
        p = ROOT / f"data/eval/{sig}-{nm}.jsonl"
        if p.exists():
            out.append((nm, load_jsonl(p)))
    return out


def shipped_scores(sig):
    """Score the currently shipped model on the same sets, as the baseline row."""
    tax = load_taxonomy(sig); labels = tax["labels"]
    mid = SHIPPED[sig]
    rev = tax.get("model_revision") if mid == tax["model_repo"] else None
    a_txt = [t for l in labels for t in tax["anchors"][l]]
    a_lab = [l for l in labels for _ in tax["anchors"][l]]
    av = embed(mid, a_txt, revision=rev)
    scores = {}
    for nm, rows in evalsets(sig):
        qv = embed(mid, [r["text"] for r in rows], revision=rev)
        pred, _ = anchor_topk_predict(qv, av, a_lab, labels, tax.get("top_k", 3))
        scores[nm] = metrics([r["tier"] for r in rows], pred, labels)
    return scores


def cell(d):
    return f"{d['accuracy']:.4f}/{d['macro_f1']:.3f}" if d else "-"


def main():
    reps = []
    for f in sorted((ROOT / "reports").glob("*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "signal" in r and not r["tag"].startswith(("smoke", "probe")):
            reps.append(r)

    for sig in SIGNALS:
        rows = [r for r in reps if r["signal"] == sig]
        if not rows:
            continue
        print(f"\n########## {sig}   (accuracy/macroF1) ##########")
        print(f"  {'arm':<34}{'arch':<7}" + "".join(f"{c:>18}" for c in COLS) + f"{'p50ms':>8}")
        try:
            s = shipped_scores(sig)
            print(f"  {'SHIPPED (baseline)':<34}{'embed':<7}"
                  + "".join(f"{cell(s.get(c)):>18}" for c in COLS) + f"{'-':>8}")
        except Exception as e:
            print(f"  (shipped baseline unavailable: {type(e).__name__})")
        for r in sorted(rows, key=lambda x: -x.get("real-gold", {}).get("accuracy", 0)):
            print(f"  {r['tag'][:33]:<34}{r['arch']:<7}"
                  + "".join(f"{cell(r.get(c)):>18}" for c in COLS)
                  + f"{r.get('cpu_p50_ms','-'):>8}")


main()
