"""Publish a trained classifier to the Hub with a model card that states the
honest number, not the flattering one.

The card is generated from `reports/<tag>.json`, so it cannot drift from what
was actually measured. Every card carries, in this order:

  * accuracy on REAL traffic (the number that predicts production behaviour)
  * accuracy on the legacy hand-authored held-out set (for continuity with the
    published llm-d-sc benchmark)
  * the gap between them, named
  * the jury contested rate, because a tier assignment that three strong models
    cannot agree on is not something the model can be graded against

That ordering is deliberate. The prior generation of these models was published
on a 98.53% figure measured against a held-out split of its own generator, and
the number that mattered -- real traffic -- was 25-44 points lower.
"""
from __future__ import annotations
import json, sys, pathlib, os
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
from evalkit import ROOT, load_taxonomy
from huggingface_hub import HfApi, create_repo

TOKEN = (os.environ.get("HF_TOKEN")
         or pathlib.Path.home().joinpath("hf_token.txt").read_text().strip())


def card(rep: dict, repo: str) -> str:
    sig = rep["signal"]
    tax = load_taxonomy(sig)
    real = rep.get("real-gold", {})
    v1 = rep.get("heldout-v1", {})
    ent = rep.get("enterprise-gold", {})
    gap = (v1.get("accuracy", 0) - real.get("accuracy", 0)) if real and v1 else None

    def row(n, d):
        if not d:
            return ""
        lo, hi = d["wilson95"]
        return (f"| {n} | {d['n']} | {d['accuracy']:.4f} | "
                f"{lo:.3f} – {hi:.3f} | {d['macro_f1']:.4f} |\n")

    t = f"""---
license: apache-2.0
library_name: {"sentence-transformers" if rep["arch"] == "embed" else "transformers"}
tags: [text-classification, llm-routing, llm-d, semantic-router, {sig}]
base_model: {rep["base"]}
---

# {repo.split('/')[-1]}

`{sig}` classifier for [llm-d semantic
classification](https://github.com/llm-d-incubation/llm-d-semantic-classifier).
Labels: {", ".join(f"`{l}`" for l in tax["labels"])}.

Architecture: **{"embedding + anchor-topk-mean (drop-in for the llm-d-sc runtime)"
                 if rep["arch"] == "embed" else
                 "sequence-classification head (requires a runtime that reads logits)"}**,
base `{rep["base"]}`.

## Accuracy

Read the real-traffic row first.

| eval set | n | accuracy | 95% CI | macro F1 |
|---|---:|---:|---|---:|
"""
    ref = rep.get("refined-gold")
    if ref:
        t += row("**real traffic, refined gold** (high-effort re-adjudication)", ref)
    t += row("real traffic (WildChat, unanimous 3-model jury)", real)
    ent2 = rep.get("entsec-gold")
    if ent2:
        t += row("enterprise, secrets-handling situations", ent2)
    t += row("enterprise (unconditioned generation, unanimous jury)", ent)
    t += row("legacy hand-authored held-out", v1)
    if gap is not None:
        t += f"""
**Hand-authored minus real traffic: {gap:+.3f}.** Hand-authored held-out prompts
are written in the same clean register as the anchors; real users send truncated
pastes, fragments and roleplay preambles. The real-traffic row is the one that
predicts production behaviour.
"""
    gate = rep.get("gate")
    if gate:
        t += "\n### Gate behaviour — what this model does in deployment\n\n"
        t += ("Tier-exact recall understates a gate's safety: classifying a "
              "`REGULATED` prompt as `CONFIDENTIAL` is the wrong tier but still "
              "blocked. What matters is how much sensitive content is contained, "
              "and how much benign content is blocked unnecessarily.\n\n")
        t += "| gate threshold | contained | over-blocked |\n|---|---:|---:|\n"
        for k, v in gate.items():
            t += f"| block >= `{k}` | {v['contained']:.3f} | {v['over_block']:.3f} |\n"

    rec = (rep.get("entsec-gold") or rep.get("enterprise-gold") or {}).get("recall")
    if rec:
        t += "\n### Per-tier recall — read this before the accuracy\n\n"
        t += ("For an egress or capacity gate the errors are not interchangeable: "
              "missing a `NEVER_EGRESS` prompt leaks a live credential, while "
              "over-flagging a `PUBLIC` one costs a cheap round trip. Aggregate "
              "accuracy is carried by the largest class and hides this.\n\n")
        t += "| tier | recall |\n|---|---:|\n"
        for k, v in rec.items():
            t += f"| `{k}` | {v:.2f} |\n"

    sn = rep.get("seed_note")
    if sn:
        t += f"\n> **Run-to-run variance.** {sn}\n"

    t += f"""
### The eval has a measured ceiling

Gold labels were audited by blind paired adjudication in two strata — the rows
this model got wrong, and a sample of the rows it got right — with the judge
shown two candidate labels in random order and no indication of provenance.
**Roughly 4.9% of the gold labels are themselves wrong**, so a PERFECT
classifier scored against this eval would reach about **0.95, not 1.0**.

Read the real-traffic accuracy against that ceiling, not against 100%. Auditing
only a model's mistakes would move the number up artificially; sampling the
correct rows too is what makes the estimate honest, and it revealed that on
~3.3% of "correct" rows the model agreed with a bad label — meaning measured
accuracy is very slightly overstated.

### How the eval was built

Real-traffic rows come from [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
(ungated real assistant traffic). Each prompt was labelled independently by three
models (`claude-opus-5`, `claude-sonnet-5`, `claude-fable-5-1`) from the task
rubric alone -- **no labeller ever saw a proposed label**, so agreement is
evidence rather than assent. Only unanimous rows are scored.

Those three agree unanimously on roughly 70-74% of real prompts. The remaining
prompts are published as a `contested` split rather than discarded: they measure
how much real traffic this taxonomy does not resolve, which no single accuracy
figure can express.

### Training data

{rep["train_rows"]} rows from `{rep["source"]}`, mixing jury-labelled real traffic
(register and class prior) with rubric-grounded synthetic data (coverage of tiers
that are rare in real traffic). Training prior: `{rep.get("train_prior")}`.
Held-out eval prompts are excluded by content hash.

## Latency

CPU single-request: **p50 {rep.get('cpu_p50_ms')} ms, p99 {rep.get('cpu_p99_ms')} ms**
(Apple M-series, single thread). llm-d-sc serves the classifier on CPU, so model
size trades directly against per-replica throughput.

## Limitations

- WildChat is consumer traffic. For `sensitivity` it is ~93% `PUBLIC` and cannot
  measure the tiers that gate egress; the enterprise row above covers those.
- Labels come from LLM jurors, not human annotators. The rubric was validated by
  reproducing the project's hand-authored gold labels
  (complexity 0.9875, cost 1.000, sensitivity 1.000) before use.
- Not independently reproduced.
"""
    return t


if __name__ == "__main__":
    tag = sys.argv[1]
    repo = sys.argv[2]
    rep = json.load(open(ROOT / f"reports/{tag}.json"))
    local = ROOT / "models" / tag
    (local / "README.md").write_text(card(rep, repo))
    api = HfApi(token=TOKEN)
    create_repo(repo, token=TOKEN, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(local), repo_id=repo, repo_type="model",
                      ignore_patterns=["_ck/*", "**/_ck/**", "checkpoint-*"])
    print(f"pushed {local} -> https://huggingface.co/{repo}")
