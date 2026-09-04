"""Reproduce the committed Rust benchmark numbers in Python.

If these do not match docs/benchmarks/classification-accuracy.md, the Python
harness is not measuring what production measures and nothing downstream of it
can be trusted.
"""
import sys; sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import *

EXPECTED = {"complexity": 0.9750, "cost": 0.8333, "sensitivity": 0.8933}

print("Reproducing committed Rust numbers with the shipped taxonomies\n")
for sig in SIGNALS:
    tax = load_taxonomy(sig); rows = heldout(sig)
    labels = tax["labels"]
    a_txt, a_lab = [], []
    for lab in labels:
        for t in tax["anchors"][lab]:
            a_txt.append(t); a_lab.append(lab)
    mid, rev = tax["model_repo"], tax.get("model_revision")
    qv = embed(mid, [r["text"] for r in rows], revision=rev)
    av = embed(mid, a_txt, revision=rev)
    pred, _ = anchor_topk_predict(qv, av, a_lab, labels, tax.get("top_k", 3))
    m = metrics([r["tier"] for r in rows], pred, labels, [r.get("hard") for r in rows])
    exp = EXPECTED[sig]; delta = m["accuracy"] - exp
    flag = "MATCH" if abs(delta) < 1e-3 else f"DIFFERS by {delta:+.4f}"
    print(fmt(sig, f"{mid.split('/')[-1]}", m))
    print(f"      anchors={len(a_txt)} top_k={tax.get('top_k')} | committed={exp:.4f} -> {flag}\n")
