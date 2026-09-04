# llm-d-sc v0.2 — classifier accuracy optimisation

Everything measured while trying to raise the accuracy of the three llm-d-sc
routing classifiers — **complexity**, **cost** and **sensitivity** — on real and
enterprise traffic. Companion to
[llm-d-sc-v0.2-benchmarking](https://github.com/cnuland/llm-d-sc-v0.2-benchmarking),
which covers throughput and latency; this repo is about whether the router is
*right*, not how fast it is wrong.

**[`reports/FINDINGS.md`](reports/FINDINGS.md) is the substance** — 72 numbered
sections, every experiment including the ones that failed, with the measurement
that settled each. Negative results are kept deliberately: about two thirds of
what is recorded here is something that did **not** work, and several entries
retract an earlier entry in the same file.

## Headline

| signal | shipped v0.1 | current best | eval |
|---|---:|---:|---|
| complexity | 0.6411 | **0.8963** | refined gold, real traffic |
| cost | 0.3910 | **0.8976** | refined gold, real traffic |
| sensitivity | 0.2903 | **0.8034** | enterprise + secrets (synthetic) |

Models on the Hub: [`complexity-v2`](https://huggingface.co/cnuland/llm-d-sc-complexity-v2),
[`cost-v2`](https://huggingface.co/cnuland/llm-d-sc-cost-v2),
[`sensitivity-v2.1`](https://huggingface.co/cnuland/llm-d-sc-sensitivity-v2.1),
[`complexity-v2-mini`](https://huggingface.co/cnuland/llm-d-sc-complexity-v2-mini).

### Tier-exact accuracy is the wrong metric, and that is the main result

**None of these reach the high 90s and none can** (§69). Split each eval by whether
its three-juror jury was unanimous:

| signal | unanimous rows | contested rows | ceiling if every contested row were labelled perfectly |
|---|---:|---:|---:|
| complexity | 89.47% | 63.64% | 92.59% |
| cost | 88.06% | 58.33% | 91.43% |
| sensitivity | 77.93% | 51.24% | 83.56% |

Worse, **model accuracy tracks jury agreement within 1.3-2.6 points at every fold
of the same eval** (§74). Eighteen mechanisms, five encoders, six corpora and
three rubric rewrites later, the models have been pinned to how decidable the
question is, not to their own capability.

**But llm-d-sc does not consume four complexity tiers.** The deployed Praxis table
routes SIMPLE/MEDIUM to the small model and COMPLEX/REASONING to the large one. A
classifier saying MEDIUM where the jury said SIMPLE picks the same backend and has
made no routing error — tier-exact accuracy charges it for one anyway. Scored on
the decisions that actually exist (§75, §79):

| signal | deployed decision | jury agr | accuracy | majority baseline |
|---|---|---:|---:|---:|
| complexity | is reasoning needed | 94.6% | **95.79%** | 86.53% |
| sensitivity | block at NEVER_EGRESS | 96.4% | **95.62%** | 82.89% |
| cost | short vs long generation | 88.1% | **92.44%** | 50.42% |
| complexity | route: small vs large | 82.0% | 88.22% | 75.42% |

Majority baselines are in every table for a reason: one candidate fold scores
93.78% against a 93.45% baseline, and the binary egress model scores 99.65% on an
eval containing two positive rows. Both would otherwise read as the best results
here.

The honest summary: **three of the router's four real decisions sit at 92-96%;
the small-vs-large route at 88.22% is the gap.**

## Findings worth reading first

- **§66 — tier escalation should be removed.** It was added to sensitivity as a
  safety measure and never audited against the safety metric that justified it.
  Measured: −1.91 accuracy, +1 to +2.4 unnecessary blocking, under a point of
  containment bought.
- **§60 — the matched-containment control.** Three separate interventions
  (post-hoc logit adjustment, span-max inference, a 3-model ensemble) produced
  impressive containment tables that dissolved once over-block was read at
  *equal* containment. The control did more work than anything it tested.
- **§63 — synthetic data with 98.4% blind-check agreement was 8 points per row
  WORSE than real data.** High agreement is not a quality signal; it means the
  items are unambiguous, and the real failures are the ambiguous ones.
- **§70 — the cost rubric describes cues that are nearly absent from the
  traffic.** Its brevity cue fires on 8.7% of LOW rows and 6.4% of MODERATE.
- **§55 — one diagnostic, opposite answers on two signals.** Complexity is
  rubric-limited; sensitivity is model-limited. Run it before any relabelling
  campaign, not after.
- **§65 — a retraction.** An earlier section reported the complexity rubric
  rewrite as "+7.0 points, the largest lever in the project". At corpus scale it
  is +0.2. The subset effect was presented as the population effect.

## Reusable diagnostics (`harness/`)

The methodology is the transferable part. Each is signal-agnostic:

| tool | question it answers |
|---|---|
| `headroom.py` | is this signal capped by its LABELS or by the MODEL? |
| `roc_match.py` | does this intervention beat a plain threshold bias at matched containment? |
| `pair_encoders2.py` | how much of a confusable pair does each encoder's representation carry? |
| `rubric_ab.py` | does a rubric rewrite raise labeller agreement — measurable *before* relabelling |
| `gate_metric.py`, `gate_compare.py` | containment vs over-block per gating threshold |
| `secret_rules.py` | high-precision credential detection (0.25% FP) as a classifier pre-check |
| `compress.py` | extractive prompt compression (arXiv 2603.12646) — measured, rejected |
| `sdg.py` | synthetic generation: minimal pairs, anti-cues, blind relabelling |

`train.py` carries the fine-tuning mechanisms tried, including soft targets from
jury vote distributions, ordinal smoothing, LoRA, class weighting with tier
escalation, and a hierarchy-aware supervised contrastive loss
(Khosla et al. 2020; hierarchy weighting after Findings of EACL 2024).

## Hardware and tools

All training and evaluation ran locally:

| | |
|---|---|
| machine | Apple M4 Max, 16 cores, 128 GB unified memory, macOS 26.6.2 |
| accelerator | Apple MPS (no CUDA; llm-d-sc serves the classifier on CPU) |
| python | 3.11.11 |
| torch | 2.14.0 |
| transformers | 5.16.1 |
| scikit-learn | 1.9.0 |
| encoders | `all-MiniLM-L6-v2`, `bge-base-en-v1.5`, `e5-base-v2`, `all-mpnet-base-v2`, ModernBERT, DeBERTa-v3 |
| labelling / generation | Anthropic API — `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5-1` as independent jurors |

Latency figures in the findings are single-request CPU p50 with the model on CPU,
which is how llm-d-sc serves it.

## What is NOT in this repo, and why

- **`models/`** — 13 GB of checkpoints. The published ones are on HuggingFace.
- **`data/eval/`, `data/train/`** — ~200 MB. Derived from
  [WildChat](https://huggingface.co/datasets/allenai/WildChat) under its own
  licence, and several files contain synthetic API keys and private keys **by
  design** — they are the sensitivity taxonomy's NEVER_EGRESS tier, and shipping
  them would trip secret scanners for no benefit. Every generator, labeller and
  adjudicator needed to rebuild them is in `harness/`.
- **`data/.llmcache/`** — raw API responses, same reason.

`reports/*.json` holds the per-run metrics behind every table in the findings:
full confusion matrices, Wilson intervals, per-tier recall, latency, and the
exact recipe each model was trained with.

## Reproducing

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # generation, labelling, adjudication
export HF_TOKEN=...                 # only needed for harness/upload.py

./.venv/bin/python harness/build_eval_real.py          # rebuild the evals
./.venv/bin/python harness/build_train_real.py 60000 complexity
./harness/round_w.sh                                   # any round_*.sh is one experiment
```

Each `round_*.sh` opens with a comment stating what it varies, what the
prediction was, and which earlier finding motivated it. Where a round exists to
overturn a previous conclusion, it says so.

## Caveat on comparisons

The vLLM Semantic Router numbers this work was originally benchmarked against are
**not like-for-like**. Those are MMLU-Pro category classification, where the label
ships with the dataset. These taxonomies are subjective tiers adjudicated by a
jury, with inter-juror agreement of 87.7% (complexity), 74.5% (sensitivity) and
73.4% (cost). Tier-exact accuracy against labels that noisy has a hard ceiling
well below either project's headline number.
