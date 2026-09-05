# llm-d-sc classifier accuracy — findings log

Working dir `/Users/cnuland/llm-d-sc-accuracy`. Every number here is reproducible
from `harness/`; experiments are numbered and their scripts keep their names.

## 0. Where we started

`docs/benchmarks/classification-accuracy.md` in llm-d-sc-genesis reports, on
hand-authored held-out sets:

| signal | shipped accuracy | n | 95% CI (Wilson) |
|---|---:|---:|---|
| complexity | 0.9750 | 80 | 0.913 – 0.993 |
| sensitivity | 0.8933 | 75 | 0.803 – 0.945 |
| cost | 0.8333 | 60 | 0.720 – 0.907 |

The Python harness reproduces complexity and sensitivity **exactly**, so it is
measuring what production measures and the rest of this work can be trusted.

**Those n's cannot certify a "high 90s" claim.** At n=75 the 95% CI is ±7 points.
Even a perfect 75/75 only proves ">0.951". Enlarging the eval is a prerequisite,
not a nicety.

## 1. A shipping bug: cost.json points at the wrong model

`classifiers/cost.json` has `model_repo: sentence-transformers/all-MiniLM-L6-v2`
— the *un-finetuned* base. The retrained `cnuland/llm-d-sc-cost` exists on the
Hub and is what the committed 0.8333 was measured with.

| cost arm | accuracy | hard-case accuracy |
|---|---:|---:|
| shipped `cost.json` (baseline MiniLM) | 0.7500 | 0.400 |
| `cnuland/llm-d-sc-cost` | 0.8333 | 0.750 |

8.3 points, and nearly double on boundary cases, lost to a config field.

## 2. The prior 98.53% does not mean what it looks like

`hello-chris-sr-finetuned` reports 98.53% on complexity. Train and test there are
both drawn from ONE synthetic pipeline seeded by the same 48 anchors and written
by one generator, so the test split is in-distribution with its own training
data — the model can score on the generator's stylistic tells. SIMPLE and
REASONING both hit a perfect 1.0000, which real traffic does not do. The honest
re-measurement on hand-authored data is the 0.9750 above.

## 3. What the reference systems actually do (and llm-d-sc does not)

- **ComplianceGate** (arXiv 2606.31163) — the closest published system: routes on
  complexity *and* sensitivity. BERT-base encoder, 12L/768h/12heads/3072FFN,
  ~134M params, 128-token truncation, **two-layer FFN head with tanh over the
  first token → softmax over 6 labels**. 50,000 fine-tuning examples, 12,000
  steps. **99.2% accuracy, 7 ms overhead.**
- **vLLM Semantic Router** — ModernBERT / mmBERT-32K with
  `ModernBertForSequenceClassification` + **LoRA** adapters per task
  (`src/training/model_classifier/`), exported to ONNX.
- **llm-d-sc** — embed → masked mean-pool → cosine against anchors → mean of the
  top-3 per label → argmax. Anchors are **text**
  (`taxonomy.rs: anchors: BTreeMap<String, Vec<String>>`), embedded at load.

Both reference systems use a learned softmax head. llm-d-sc cannot: a head would
be a runtime change, so the gain has to be measured before it is proposed.

## 4. Experiment 04 — the decision rule is worth points, when the data is right

Frozen neutral encoders (never fine-tuned on this task), swapping only the
decision rule. `head` is logistic regression on the same frozen vectors.

| encoder | sensitivity anchor → head | cost anchor → head |
|---|---|---|
| all-MiniLM-L6-v2 | 0.8267 → 0.8933 (**+6.7**) | 0.7500 → 0.7333 (−1.7) |
| all-mpnet-base-v2 | 0.8533 → 0.9067 (**+5.3**) | 0.8000 → 0.7000 (−10.0) |
| bge-base-en-v1.5 | 0.8400 → 0.8667 (**+2.7**) | 0.8833 → 0.6833 (−20.0) |
| e5-base-v2 | 0.8400 → 0.8933 (**+5.3**) | 0.8667 → 0.7167 (−15.0) |
| mxbai-embed-large-v1 | 0.8267 → 0.8800 (**+5.3**) | 0.7167 → 0.7000 (−1.7) |

Consistent sign across five independent encoders in both directions:

- **sensitivity** — a learned boundary beats cosine-to-anchors everywhere. The
  decision rule is leaving real accuracy on the table.
- **cost** — the head loses everywhere. Five encoders cannot overfit identically;
  this is **label drift**: `cost-train.jsonl` teaches a different task from the
  one the eval measures. Corroborated by 5-fold CV *within* the cost corpus of
  only 0.8637 (a synthetic corpus should separate near-perfectly), and by a
  systematic shift where 10/15 held-out HIGH collapse to MODERATE.

The best cost number anywhere in this table is `bge-base` + **hand-written
anchors** (0.8833). The anchors are right; the training data is wrong.

## 5. Root cause: the labels were never operationally defined

Written `rubrics/{complexity,sensitivity,cost}.md`, then blind-labelled the
existing held-out sets from text + rubric alone (judge never sees the gold label).
Agreement with gold:

| signal | first draft | after calibration |
|---|---:|---:|
| complexity | 0.8875 | **0.9875** |
| cost | 0.6833 | **1.0000** |
| sensitivity | 0.9600 | **1.0000** |

The first-draft failures localised the defect precisely:

- **cost 0.6833** — 13/15 MODERATE mislabelled HIGH. My draft scaled cost by
  output-token count; the gold set scales it by **bulk**: HIGH means a large
  supplied corpus or exhaustive coverage of a whole domain ("these thirty
  papers", "every page", "this whole monorepo"), MODERATE means one substantial
  bounded artifact. Once written down, agreement is perfect.
- **complexity 0.8875** — every error in MEDIUM, 7 of 9 leaking to COMPLEX,
  because "involves tradeoffs" is too loose. MEDIUM is *one conventional
  deliverable*; COMPLEX is *system design with no single right answer*. This is
  the same boundary the trained model fails on.
- **sensitivity 0.9600** — INTERNAL vs CONFIDENTIAL (headcount plans, vendor
  evaluations are ordinary business = INTERNAL) and controls-for-regulated-data
  (a retention schedule for patient records is REGULATED even with no record
  pasted).

The rubrics now reproduce the product's own labels, so they can ground both
generation and verification.

## 6. Real traffic is far more ambiguous than either eval suggests

Built a real-traffic eval from **WildChat-1M** (ungated real ChatGPT traffic):
24,000 English first-turn prompts harvested, 6,000 screened, 150/tier sampled,
then **three independent blind labellers** (opus-5, sonnet-5, fable-5-1).

complexity, n=600:

- unanimous **442**, contested (2–1) **155**, unresolved (1–1–1) **3**
- pairwise agreement **0.845 / 0.830 / 0.793**

Three strong models sharing a rubric that reproduces gold at 0.9875 agree
unanimously on only **74%** of real prompts. Any "high 90s" claim is meaningful
only against the unanimous subset, and must be reported alongside the contested
rate.

Screened tier mix of real traffic (6,000 prompts) is itself a routing finding:

| signal | distribution |
|---|---|
| complexity | SIMPLE 2154, MEDIUM 3150, COMPLEX 387, REASONING 309 |
| cost | MINIMAL 1532, LOW 2465, MODERATE 1776, HIGH 227 |

The expensive tiers are ~11% of complexity traffic and ~4% of cost traffic.

## 7. THE HEADLINE: the shipped classifiers lose 25–44 points on real traffic

Experiment 05. Same models, same anchors, same scoring rule — only the eval set
changes. `real-gold` is the unanimous three-model jury over WildChat prompts.

| signal | arm | heldout-v1 | **real-gold** | real-contested |
|---|---|---:|---:|---:|
| complexity | shipped `cnuland/llm-d-sc-complexity` | 0.975 | **0.641** | 0.432 |
| complexity | baseline MiniLM | 0.825 | 0.371 | 0.324 |
| cost | shipped `cost.json` (baseline MiniLM) | 0.750 | 0.347 | 0.310 |
| cost | retrained `cnuland/llm-d-sc-cost` | 0.833 | **0.391** | 0.339 |
| sensitivity | shipped `cnuland/llm-d-sc-sensitivity` | 0.893 | **0.637** | 0.420 |
| sensitivity | baseline MiniLM | 0.827 | 0.296 | 0.252 |

**complexity −33.4, cost −44.2, sensitivity −25.6 points.**

Fine-tuning is still clearly doing real work — every finetuned arm roughly
doubles its baseline on real traffic (complexity 0.371→0.641, sensitivity
0.296→0.637). The problem is not that the models are useless. It is that
**the published numbers are measured on hand-authored prompts written in the
same register as the anchors**, and real users do not write like that.

Two distinct failure modes are mixed together here, and they need different fixes:

1. **Register gap.** Held-out prompts are clean, single-clause, and well-formed.
   Real prompts are truncated pastes, roleplay preambles, and fragments
   ("using the internet in 2006"). Fix: train on data in the real register —
   the `real` source in the v2 generator rewrites into WildChat voice.

2. **Prior mismatch.** Training corpora are balanced by construction; real
   traffic is not. Sensitivity real-gold is 242/284 PUBLIC, and the shipped
   model scores accuracy 0.637 with **macro-F1 0.227** — it fires the sensitive
   tiers far too often. A balanced-trained head is worse still: ModernBERT
   trained on v1 sensitivity reaches 0.9694 on its own split and **0.4049** on
   real traffic. Fix: match the prior, or correct for it at inference.

Real-traffic priors, screened over 6,000 prompts:

| signal | distribution |
|---|---|
| complexity | SIMPLE 2154, MEDIUM 3150, COMPLEX 387, REASONING 309 |
| cost | MINIMAL 1532, LOW 2465, MODERATE 1776, HIGH 227 |
| sensitivity | PUBLIC 5600, INTERNAL 282, REGULATED 71, CONFIDENTIAL 26, NEVER_EGRESS 21 |

### The ceiling this sets

Three strong models sharing a rubric that reproduces gold at 0.9875–1.000 reach
unanimity on only 70–74% of real prompts (pairwise 0.76–0.85). So "high 90s on
real traffic" is **not attainable as stated** — the target has to be:

- high 90s on the **unanimous** subset (where the task is well-posed), reported
  together with
- the **contested rate**, which measures how much real traffic the taxonomy
  does not resolve at all.

Any single number quoted without that second figure is not measuring the router.

### Known limitation of the sensitivity real eval

WildChat is consumer traffic. Its sensitivity gold set has 242 PUBLIC but only
6 CONFIDENTIAL, 7 REGULATED and 2 NEVER_EGRESS — too few to measure the tiers
that actually gate egress. A separate enterprise eval is being built by
UNCONDITIONED generation: a model is given a role, a company and a moment and
asked what that person would type, is never shown the taxonomy, and the same
blind jury assigns tiers afterwards.

## 8. Better synthetic data alone does NOT fix real-traffic accuracy

Trained on the v2 complexity corpus (2,260 rows, rubric-grounded, blind-verified
at 94.0% generator/verifier agreement, with minimal pairs, anti-cue items and
WildChat-voice rewrites) and measured against the honest eval:

| arm | heldout-v1 | **real-gold** | val-internal | CPU p50 |
|---|---:|---:|---:|---:|
| shipped `cnuland/llm-d-sc-complexity` | 0.975 | **0.641** | — | ~9 ms |
| embed/MiniLM on v2 synthetic | 0.950 | 0.550 | 0.925 | 3.4 ms |
| head/ModernBERT-base on v2 synthetic | 0.550 | 0.574 | **0.965** | 16.2 ms |

Both v2 arms land **below the shipped model on real traffic**, despite the
corpus being materially better by every internal measure. Cleaner synthetic data
buys in-distribution accuracy, not transfer.

The head arm is the clearest illustration: **0.965 on its own validation split,
0.550 on hand-authored prompts.** It learned the generator's register, which is
the same mechanism that produced the original 98.53% headline.

An unplanned observation worth keeping: the embed arm holds 0.950 on
heldout-v1 while the head collapses to 0.550. Anchor-topk-mean scores against
**hand-written anchor text**, so the anchors act as a register anchor that pins
the decision boundary to human-written prompts. A softmax head has no equivalent
and drifts with whatever wrote its training data. That is a genuine robustness
argument for the shipped architecture — one that pure accuracy numbers hide.

### Consequence for the plan

Real traffic has to be in the training mix, but it cannot be the whole mix,
because the tiers that matter are rare in it: COMPLEX+REASONING are 11% of
complexity traffic, HIGH is 4% of cost traffic, and the three sensitive tiers
together are ~2% of WildChat. The corpus therefore becomes:

- **real, jury-labelled WildChat** — supplies register and prior
- **synthetic** — supplies coverage of the rare tiers and the boundary cases
  that real traffic contains too few of

Two labellers (opus-5, sonnet-5) are used for training rows rather than the
three used for eval: training tolerates label noise, evaluation does not, and
the weaker bar roughly doubles the volume.

## 9. Anchor selection: two failed approaches, and what they prove

Anchors are the cheapest thing to change — they are plain text in
`classifiers/<sig>.json`, embedded at load, so a better anchor set ships as
configuration with no retraining and no runtime change. Two attempts:

**Experiment 06 — greedy forward selection maximising dev accuracy.**
Candidate pool 1,280 (synthetic + incumbents), 48 slots, selection on a 209-row
dev half of real-gold, scored on the unseen test half.

| eval | incumbent | optimised | delta |
|---|---:|---:|---:|
| dev (selection saw this) | 0.6220 | 0.7129 | **+0.0909** |
| test (unseen) | 0.6603 | 0.6555 | −0.0048 |
| heldout-v1 | 0.9750 | 0.7375 | **−0.2375** |

Textbook selection overfitting: all the gain is on the split the search
optimised, and it destroys the legacy set. Rejected.

**Experiment 07 — k-medoids per label, eval never consulted.**
Farthest-point init, Lloyd reassignment, medoid = the real example nearest its
cluster mean. Pool = 5,566 verified synthetic rows.

| eval | incumbent | medoid | delta |
|---|---:|---:|---:|
| heldout-v1 | 0.9750 | 0.7000 | −0.2750 |
| real-gold | 0.6411 | 0.5789 | −0.0622 |
| real-contested | 0.4318 | 0.4034 | −0.0284 |

Worse everywhere — but consistently with §8 rather than against it. Medoids of a
synthetic corpus inherit the synthetic corpus's register problem; the selection
method is not what failed.

**What both results actually establish:** 48 hand-written anchors beat the
medoids of 5,566 verified synthetic examples on every eval, and beat a search
that was allowed to optimise directly against the eval. The incumbent anchors
are a much stronger artifact than their provenance suggests, and they are doing
real work holding the decision boundary in human-written register (§8).

The version of this experiment that is still worth running is medoids over
**jury-labelled real traffic** — anchors drawn from the distribution that is
actually served, which is the register gap closed as a config change. That pool
is being labelled now.

## 10. The sensitivity classifier is worse than no fine-tuning on enterprise text

An enterprise eval was built by UNCONDITIONED generation: `claude-opus-5` was
given a role, an organisation and a moment in the working day and asked what
that person would type. It was never shown the taxonomy, so the text carries no
tier-shaped fingerprint. The same three-model blind jury assigned tiers
afterwards. 1,036 prompts → **744 unanimous**, 279 contested, 13 unresolved
(pairwise agreement 0.792–0.823).

Unlike WildChat, this set can actually measure the egress tiers:
INTERNAL 397, CONFIDENTIAL 154, REGULATED 114, PUBLIC 74, NEVER_EGRESS 5.

| sensitivity arm | heldout-v1 | real-gold | **enterprise-gold** | ent-contested |
|---|---:|---:|---:|---:|
| shipped `cnuland/llm-d-sc-sensitivity` | 0.8933 | 0.6373 | **0.2903** | 0.2760 |
| un-finetuned baseline MiniLM | 0.8267 | 0.2958 | **0.4113** | 0.3369 |

**Fine-tuning made it worse than the base model on realistic enterprise
prompts — 0.2903 against 0.4113.**

Per-tier recall, shipped model on enterprise-gold:

| tier | recall |
|---|---:|
| PUBLIC | 0.24 |
| INTERNAL | 0.14 |
| CONFIDENTIAL | 0.34 |
| REGULATED | 0.78 |
| NEVER_EGRESS | 0.40 (n=5, not reliable) |

Cause is visible in the v1 corpus. Its examples read like
*"the internal SOP for handling a mechanical failure on the Magic Carpet in
Sector 4"* — a fictional register. Fine-tuning pulled the embedding space
toward that register and away from the enterprise text the signal exists to
classify. On the hand-authored held-out set, written in the same fictional
register, it scores 0.8933 and looks healthy.

This is the same mechanism as §7 and §8, but with the sign flipped from
"overstated" to "actively harmful", and it is on the signal that decides whether
data may leave the network.

### Gap that still needs closing

`NEVER_EGRESS` has only 5 unanimous enterprise rows and 2 real ones. Unconditioned
generation rarely produces live credentials, which is realistic but leaves the
highest-consequence tier effectively unmeasured. Needs scenarios where secrets
appear naturally — pasting a CI config while debugging a failed deploy, rotating
keys, reviewing a `.env`, a privileged note from counsel — still without naming
the tier.

## 11. Complexity matrix: mixed corpus + softmax head, 0.641 → 0.845 on real traffic

Six arms, varying corpus, architecture and base model independently so a win can
be attributed. `mix` = 5,566 verified synthetic + 7,804 jury-labelled real.

| arm | heldout-v1 | **real-gold** | contested | CPU p50 |
|---|---:|---:|---:|---:|
| SHIPPED `cnuland/llm-d-sc-complexity` | 0.9750 | **0.6411** | 0.4318 | ~9 ms |
| embed / MiniLM / synthetic only | 0.9250 | 0.5383 | 0.4375 | 3.4 ms |
| embed / MiniLM / real only | 0.8250 | 0.5957 | 0.4375 | 3.4 ms |
| embed / MiniLM / mix | 0.9500 | 0.6531 | 0.5398 | 3.4 ms |
| embed / bge-base / mix | 0.9125 | 0.6962 | 0.5284 | 13.1 ms |
| head / MiniLM / mix | 0.8250 | **0.8230** | 0.6193 | **3.0 ms** |
| head / ModernBERT-base / mix | 0.8125 | **0.8445** | 0.6136 | 15.8 ms |

**+20.3 points on real traffic over the shipped model** (0.6411 → 0.8445).

1. **Neither corpus alone is enough.** For the drop-in embed architecture,
   mix (0.6531) beats real-only (0.5957) beats synthetic-only (0.5383). Real
   traffic supplies register and prior; synthetic supplies the rare tiers
   (COMPLEX+REASONING are 11% of traffic) and the boundary cases.

2. **The head result reverses once real data is present.** On synthetic alone
   (§8) the head lost badly — 0.550 on hand-authored prompts. On the mix it wins
   by 15–19 points on real traffic. The head has more capacity than
   cosine-to-anchors and will use it to fit whatever it is shown; the earlier
   failure was the corpus, not the architecture.

3. **head/MiniLM dominates embed/bge on both axes** — 0.8230 at 3.0 ms versus
   0.6962 at 13.1 ms — because it scores a 4-way softmax instead of 48 cosine
   comparisons. Given llm-d-sc's measured ~480 classifications/sec per replica
   with MiniLM, this is a throughput *gain*, not a trade.

4. **heldout-v1 and real-gold now rank arms differently.** The head arms score
   0.81-0.83 on the legacy set where embed arms reach 0.95. The legacy set is
   written in anchor register and structurally favours anchor scoring; it should
   be reported, but it must not be the selection criterion. Ranking by it is
   what shipped a sensitivity model at 0.2903 on enterprise text (§10).

### Caveat that must ship with these numbers

Training labels and eval labels both come from LLM jurors. The model is being
distilled from the jury, then graded by the jury. External grounding is that the
rubric driving both was validated by reproducing the project's own hand-authored
gold at 0.9875 / 1.000 / 1.000 (§5) — but a reader should know the eval is not
independent of the training signal in the way a human-annotated set would be.

## 12. Sensitivity matrix: 0.2903 → 0.7312 on enterprise text

Seven arms. `mix` = 6,783 synthetic + 8,359 jury-labelled real + 1,604
unconditioned enterprise rows.

| arm | heldout-v1 | real-gold | **enterprise-gold** | ent macroF1 | CPU p50 |
|---|---:|---:|---:|---:|---:|
| SHIPPED `cnuland/llm-d-sc-sensitivity` | 0.8933 | 0.6373 | **0.2903** | 0.2603 | ~9 ms |
| un-finetuned baseline MiniLM | 0.8267 | 0.2958 | 0.4113 | 0.3046 | — |
| embed / MiniLM / synthetic only | 0.7600 | 0.4155 | 0.3454 | 0.3447 | 3.3 ms |
| embed / MiniLM / real only | 0.8400 | 0.2958 | 0.2218 | 0.1632 | 3.3 ms |
| embed / MiniLM / mix | 0.8133 | 0.8345 | 0.3159 | 0.3343 | 3.4 ms |
| embed / bge-base / mix | 0.8400 | 0.8345 | 0.5430 | 0.4837 | 14.3 ms |
| head / MiniLM / mix | 0.8667 | 0.8768 | 0.6761 | 0.5279 | **2.9 ms** |
| head / ModernBERT / mix | 0.8667 | **0.8979** | **0.7312** | **0.6403** | 15.7 ms |
| head / MiniLM / **enterprise only** | 0.2000 | 0.0951 | 0.5470 | 0.2254 | 3.3 ms |

**+44.1 points on enterprise text** (0.2903 → 0.7312), and the shipped model's
deficit against its own un-finetuned base is erased.

1. **Single-corpus training fails in both directions, symmetrically.** Real-only
   collapses on enterprise (0.2218 — worse than the un-finetuned base, because
   WildChat is 85% PUBLIC and teaches the model to say PUBLIC). Enterprise-only
   collapses everywhere else (0.0951 real, 0.2000 legacy). Only the three-corpus
   mix is good on all of them. This is the §8/§11 result again: each corpus
   supplies a distribution none of the others contains.

2. **ModernBERT earns its 5× latency here, unlike on complexity.** +5.5 points of
   enterprise accuracy and +11.2 of macro-F1 over MiniLM. Sensitivity has five
   tiers with subtle distinctions (INTERNAL vs CONFIDENTIAL turns on business
   harm versus ordinary operations) where complexity has four coarser ones.

3. **Read macro-F1, not accuracy, on real-gold.** The head arms score 0.87-0.90
   accuracy there with macro-F1 0.34-0.59, because real-gold is 242/284 PUBLIC —
   most of that accuracy is "correctly said PUBLIC". enterprise-gold
   (397 INTERNAL / 154 CONFIDENTIAL / 114 REGULATED / 74 PUBLIC) is the honest
   balance measure for this signal.

### Still short of the target, and why

0.7312 is not high 90s. The corpus is the constraint: 1,604 enterprise rows
against 8,359 real and 6,783 synthetic, while the eval that matters is entirely
enterprise register. A 900-scene expansion is generating now. NEVER_EGRESS also
remains unmeasurable at n=5.

## 13. A bug in my own harness: farthest-point init selects outliers in a collapsed space

After fine-tuning, the embed arms were still being scored against the ORIGINAL
hand-written anchors. That is wrong on its face — fine-tuning moves the geometry
while the anchors stay put — so I added re-selection of anchors in the
fine-tuned space, using k-medoids with farthest-point initialisation.

It produced **below-chance** results: cost real-gold 0.0703 against 0.7190 for
the incumbent anchors, on a 4-class problem, predicting HIGH for 405 of 427 rows.

A random-selection control isolated the cause:

| anchor source (same model, same pool) | cost real-gold |
|---|---:|
| incumbent, hand-written | 0.7190 |
| **k-medoids, farthest-point init** | **0.0703** |
| k nearest the class centroid | 0.7096 |
| centroid-ranked with a diversity floor | 0.6487 |
| **random rows from the training pool** | **0.7447** |
| class centroid direction (not text) | 0.7354 |

Random beat everything, so the selection method was at fault, not the pool.
Measuring where the picks landed shows why:

| label | k-medoid picks, cosine to class centroid | class mean |
|---|---:|---:|
| MINIMAL | +0.310 | +0.976 |
| LOW | +0.053 | +0.781 |
| MODERATE | +0.141 | +0.894 |

**Triplet loss collapses each class into a tight cluster, so "the point maximally
distant from those already chosen" is pathological noise.** Lloyd reassignment
cannot recover: each outlier claims a singleton cluster and remains its own
medoid. A standard, correct clustering initialisation is exactly wrong on a
deliberately collapsed geometry.

Replaced with: rank by cosine to the class centroid, drop the outlier decile,
take an even spread across the remainder. **0.7424, deterministic, ahead of the
incumbent anchors, no clustering.** Note that "k nearest the centroid" (0.7096)
underperforms — it picks ten near-duplicates, which is effectively one anchor.
Spread matters as much as centrality.

Two things worth keeping from this:

- The re-anchored figures logged before this fix are **invalid** and are being
  re-run.
- This is the third measurement in a row (§9 twice, §13 here) where a more
  sophisticated anchor-selection method lost to something trivial. The
  hand-written anchors keep proving hard to beat, and where they are beaten it
  is by *random sampling of in-distribution text* — which says the useful
  property of an anchor set is that it covers the served distribution, not that
  its members are individually well-chosen.

## 14. DeBERTa-v3 is unusable in this environment (library bug, not a tuning problem)

The literature is clear that DeBERTa-v3 should be the right base here: 30-40%
higher sample efficiency than ModernBERT, equivalent F1 on 60-70% of the data,
and disentangled attention that helps precision-heavy tasks — which is exactly
what adjacent-tier boundaries are (IJCNLP 2025, *ModernBERT or DeBERTaV3?*).

It does not train. Traced step by step on `deberta-v3-small`:

| step | loss | grads finite | weights finite AFTER AdamW update |
|---|---|---|---|
| 0 | 1.1592 | yes | **NO** — `word_embeddings.weight`, `embeddings.LayerNorm.{weight,bias}` |
| 1 | nan | no | no |
| 2 | nan | no | no |

Ruled out, in order:

- **Not the device.** NaN on step 1 on MPS *and* CPU.
- **Not the learning rate.** Identical at 2e-5 and 5e-6, with gradient clipping at norm 1.0.
- **Not the tokenizer.** `DebertaV2Tokenizer`, max emitted id 68,254 against a
  128,100 vocab, round-trips correctly. (Worth checking — loading it demands
  `tiktoken`, which is suspicious for a SentencePiece model.)
- **Not the checkpoint.** Zero non-finite parameters at load, in both `-small`
  and `-base`; embedding stats are healthy (absmax 1.9, std 0.06).

So: finite weights, finite gradients, and the first optimiser update produces NaN
in precisely the three embedding tensors. That is a bug in this stack
(`transformers` 5.16.1 + `torch` 2.14), not something tuning can reach.

**Decision: skip it.** Fixing it means pinning an older `transformers` in a
side environment, and the payoff has shrunk — DeBERTa-v3's headline advantage is
SAMPLE EFFICIENCY, and sample scarcity is no longer the binding constraint now
that the corpora have doubled (complexity is at 21,171 training rows). Recorded
here so the option is not silently lost: on a pinned older `transformers` it is
the strongest candidate base to revisit.

## 15. Round B — corpus doubling and soft targets both pay, differently

Baseline to beat: 0.8589 real-gold (`complexity-sw-minilm-e6lr5`, 7,804 real rows).

| arm | change | heldout-v1 | **real-gold** | contested |
|---|---|---:|---:|---:|
| previous champion | — | 0.8750 | 0.8589 | 0.6648 |
| `cx-b1-hard-256` | corpus doubled to 15,600 real | **0.9125** | 0.8636 | 0.6136 |
| `cx-b2-soft-256` | + soft targets, +2,358 contested rows | 0.8250 | **0.8780** | **0.6250** |

Two separable effects, and they pull in different directions:

- **Doubling the real corpus** bought +0.5 on real traffic but **+3.75 on the
  legacy hand-authored set**. More real data made the model generalise ACROSS
  registers, not merely fit more of the same one.
- **Soft targets plus the recovered contested rows** bought a further +1.4 on
  real traffic and +1.1 on contested — but cost 8.75 points on heldout-v1.

That trade is coherent rather than contradictory. The contested rows are, by
construction, prompts two strong labellers disagreed on; training toward a split
target teaches the model to hedge near boundaries. Real traffic is full of such
prompts and rewards hedging; the hand-authored held-out set contains almost none
and punishes it. Which arm to ship depends on which distribution is being served
— and for a router, that is real traffic.

## 16. The taxonomy is ordinal, and the loss has been ignoring it

78% of the champion's real-traffic errors are ADJACENT-tier confusions:

| | SIMPLE | MEDIUM | COMPLEX | REASONING |
|---|---:|---:|---:|---:|
| **SIMPLE** | 65 | 15 | 1 | 2 |
| **MEDIUM** | 11 | 206 | 14 | 7 |
| **COMPLEX** | 0 | 5 | 28 | 0 |
| **REASONING** | 1 | 2 | 1 | 60 |

46 adjacent errors against 13 at distance >= 2. Plain cross-entropy cannot see
this: predicting COMPLEX for a MEDIUM prompt is scored exactly as wrong as
predicting REASONING, when operationally it is a far smaller mistake.

Added `--ordinal`, which leaks 12% of the target mass to neighbouring tiers
(half again to next-nearest). It composes with the soft targets rather than
replacing them — a contested LOW/MODERATE row becomes
`[0.042, 0.458, 0.458, 0.042]` instead of `[0.007, 0.493, 0.493, 0.007]`.

**Deliberately gated to `cost` and `sensitivity` only.** Their tiers escalate
monotonically (MINIMAL<LOW<MODERATE<HIGH; PUBLIC<INTERNAL<CONFIDENTIAL<
REGULATED<NEVER_EGRESS). Complexity's do NOT: the rubric defines COMPLEX and
REASONING as different KINDS of hard — breadth of design versus depth of
derivation — so treating them as adjacent ranks would encode a relationship the
taxonomy does not claim. The flag prints a notice and no-ops if pointed at
complexity.

This also matters most where it is enabled: cost has the lowest labeller
agreement of the three (74.6%), and its disagreements are overwhelmingly
one-tier slips.

## 17. Active learning: random labelling was spending most of its budget on decided cases

Every round up to here labelled a RANDOM slice of traffic. But the errors are not
randomly distributed — 78% are adjacent-tier and concentrated on two boundaries
(§16) — so random labelling mostly buys prompts the model already gets right.

Scored 71,448 unlabelled prompts with the champion and selected by MARGIN (the
gap between the top two softmax probabilities):

| | median margin |
|---|---:|
| pool-wide, i.e. what random labelling buys | **0.9958** |
| the actively selected set | **0.4385** |

A margin of 0.996 is a decided case. Then jury-labelled the 9,000 selected rows
under the same two-labeller blind protocol, which produced two diagnostics that
matter more than the rows themselves:

- **On low-margin rows the model disagrees with the jury 55.4% of the time.**
  It is wrong on more than half of what it was unsure about, against roughly 13%
  on a random slice. That is the information-per-label ratio active learning is
  supposed to buy, measured rather than assumed.
- **Juror agreement on these hard prompts is 81.5%, versus 86.9% on random ones.**

The second number answers whether any headroom remains. Had agreement collapsed
toward 50% on the prompts the model finds hardest, the boundary would be
genuinely ill-posed and no further labelling could help — the ceiling would be
the taxonomy, not the model. At 81.5% the boundary is mostly LEARNABLE and the
model simply has not learned it yet.

Guards against the two standard failure modes of uncertainty sampling, both
applied: a per-tier cap on the selection (held exactly, 1,575 each) so one
ambiguous region cannot swallow the budget and skew the class prior, and 30% of
the budget held back as a random sample so the training distribution does not
drift away from real traffic. The class prior is part of what the model has to
learn — real sensitivity traffic really is ~85% PUBLIC.

Yield: 7,326 agreed + 1,660 contested = 8,986 new rows, all of them near the
decision boundary.

### Pool integrity

The pool was enlarged 24,000 -> 90,000 to make this possible. WildChat streams
deterministically, so re-harvesting with a larger target reproduces the original
prefix and appends only new prompts. Verified rather than assumed: all twelve
existing eval and training splits are still fully contained in the enlarged
pool, zero rows missing. Had the stream order shifted, every split built on the
old pool would have been silently invalidated.

## 18. Round B closed — ModernBERT buys register robustness, not just accuracy

| arm | heldout-v1 | **real-gold** | contested | train |
|---|---:|---:|---:|---:|
| b1 hard / 256 / MiniLM | **0.9125** | 0.8636 | 0.6136 | 533 s |
| b2 soft / 256 / MiniLM | 0.8250 | 0.8780 | 0.6250 | 593 s |
| b3 soft / 512 / MiniLM | 0.8000 | 0.8780 | 0.6307 | 930 s |
| **b4 soft / 512 / ModernBERT** | 0.8750 | **0.8852** | **0.6591** | 5,145 s |

Two things this settles.

**512 tokens buys nothing.** b3 equals b2 on real traffic to four decimals at
1.75x the training cost, despite 8.1% of real prompts being truncated at 256.
The discriminative signal lives in the OPENING of a prompt, not its tail —
"Write a brief summary of..." or "Design a multi-region..." declares its tier in
the first clause.

**The soft-target register trade was capacity, not the loss.** §15 recorded that
soft targets bought real-traffic accuracy while costing 8.75 points on the
hand-authored set, and read that as an inherent hedging trade. b4 falsifies the
"inherent" part: ModernBERT takes the same soft-target gain (0.8852, the best
figure so far) while holding heldout-v1 at 0.8750. A 23M model forced to hedge
near boundaries loses its grip on the clean register; a 149M model has room for
both. This is the first result in the project where ARCHITECTURE bought
something — every prior gain came from data.

The cost is not free: 8.7x training time, and ~16 ms inference against ~3 ms.
Set against the measured ~480 classifications/sec per llm-d-sc replica, that is
a throughput decision rather than an upgrade.

## 19. The eval has a ceiling, and it is 0.9514 — not 1.0

Before spending more budget chasing the residual, I checked whether the residual
is the MODEL's fault. Two strata, both adjudicated blind: the judge sees the
prompt and two candidate labels in randomised order, with no indication which
came from the jury and which from the model, and may answer NEITHER. Asking "is
this gold label right?" invites ratification; a blind pairwise choice does not.

**Stratum 1 — the 48 rows the model got WRONG:**

| judge prefers | count | share |
|---|---:|---:|
| gold | 40 | 83.3% |
| **model** | **8** | **16.7%** |
| neither | 0 | 0% |

**Stratum 2 — 120 sampled from the 370 rows the model got RIGHT** (alternative
label = the model's own second choice, so the comparison is real):

| judge prefers | count | share |
|---|---:|---:|
| gold | 116 | 96.7% |
| alternative | 3 | 2.5% |
| neither | 1 | 0.8% |

The second stratum is what makes this honest. Auditing only the model's mistakes
and then correcting the labels it disputes is fitting the eval to the model, and
can only move the number UP. Stratum 2 is the only one that can move it DOWN —
and it does: on 3.3% of the rows the model got "right", gold is wrong, meaning
the model AGREED with a bad label and its measured accuracy is overstated.

Combined:

| | |
|---|---:|
| gold errors among the 48 mistakes | 8 (16.7%) |
| gold errors among the 370 correct | 12 (3.3%) |
| total gold noise | **20 / 418 = 4.9%** |
| **ceiling — a perfect classifier scored against this gold** | **0.9514** |
| measured accuracy (`cx-b4`) | 0.8852 |
| estimated TRUE accuracy against clean labels | 0.8748 |
| remaining headroom | **+6.6 points** |

**"High 90s on real traffic" is not attainable on this eval.** It caps at 0.9514
because roughly one label in twenty is wrong. Raising that ceiling means cleaning
the gold set, which improves the measurement without improving the router. The
defensible claims are: 0.8852 measured against a 0.9514 ceiling, ~0.875 true
accuracy, from a shipped baseline of 0.641.

(An earlier note in this session quoted the ceiling as 0.9043 from stratum 1
alone. That was wrong — it ignored the noise among correct rows, which pushes in
the opposite direction. 0.9514 supersedes it.)

## 20. Margin-based active learning cannot reach the remaining errors

Of the 48 residual errors, **32 carry confidence above 0.90**, mean confidence
0.882. The model is not hesitating — it is confidently wrong.

This retro-explains Round D exactly. Adding 8,986 low-margin rows lifted
contested accuracy 0.6250 -> 0.6591 (+3.4, where the model IS unsure) and left
real-gold flat at 0.8756 vs 0.8780. Uncertainty sampling found what it is
designed to find, and the remaining errors are not in that set.

Round D also produced a clean negative worth keeping: training on the
actively-selected rows ALONE (14,552 rows, mostly hard) scored 0.7943 — nine
points worse than the mixed corpus. The easy majority is not filler; it carries
the class prior, and a corpus of only boundary cases forgets it. The 30%
random-holdback guard was doing real work.

So the acquisition function has to change from UNCERTAINTY to DISAGREEMENT:
take the prompts the model is most sure about, screen them with an independent
judge, and keep the contradictions. A confident model contradicted by a
competent judge is either a real error or a genuinely hard case — and neither is
reachable by margin.

## 21. Ensembling: +1.9 points free, and it isolates what is genuinely unfixable

Four already-trained models, averaged softmax, no training:

| ensemble | real-gold | best member |
|---|---:|---:|
| b4 + b2 + d2 | **0.9043** | 0.8852 |
| b4 + d2 | 0.9019 | 0.8852 |
| b2 + d2 | 0.8995 | 0.8852 |
| all four | 0.8995 | 0.8852 |

Diversity is what makes it work, and it is measurable. Pairwise error overlap
(Jaccard) is 0.35-0.40 between ModernBERT and the MiniLM arms but 0.53-0.56
among the MiniLM arms themselves. The architecturally different member carries
the ensemble; adding a fourth similar member makes it worse.

### The unanimous-failure set, and a coincidence that is not one

20 rows (4.8%) are missed by every member. That is almost exactly the 4.9% gold
noise from §19, which invited a tidy conclusion: irreducible error = label noise.
Checked directly, and it does not hold. Of the 8 rows the blind judge blamed on
gold, only **4** are unanimous model failures. So the 20 decompose as:

- **4 gold-label errors** — the models were right, the label was wrong
  (`"html code for select color red and green and blue"` labelled SIMPLE though
  it produces a working artifact; `"How to check if my router is available from
  WAN?"` labelled MEDIUM for one-step recall)
- **16 genuine model failures** — 3.8% of the eval that no current model gets

The other 4 gold errors were caught by SOME member, which is precisely why the
ensemble beats every individual.

**3.8% is the honest measure of remaining model headroom that ensembling cannot
reach.** That is the more useful figure than the tidy one: there is still real
room for data and architecture, not only label cleaning.

Cost: 3x inference. Against the measured ~480 classifications/sec per llm-d-sc
replica that is a capacity decision, not a free upgrade.

## 22. Disagreement-based acquisition finds the blind spot margin cannot

§20 established that uncertainty sampling cannot reach errors the model is
confident about. Inverting the search — take the MOST confident prompts, screen
them with an independent cheap judge, keep the contradictions:

| | |
|---|---:|
| candidates remaining after prior rounds | 62,462 |
| of those, high-confidence (margin >= 0.90) | **54,809 (88%)** |
| screened | 20,000 |
| **screen contradicted the confident model on** | **4,148 (20.7%)** |
| selected after per-tier capping | 2,070, mean margin **0.9955** |

The model is confident about 88% of everything, which is why margin-based
selection had so little to work with. And the disagreement profile reproduces
the eval's error profile without ever seeing the eval:

| pair | disagreements in pool | errors in eval |
|---|---:|---:|
| MEDIUM -> SIMPLE | 1,178 | 12 |
| SIMPLE -> MEDIUM | 407 | 10 |
| MEDIUM -> COMPLEX | 237 | 6 |

Finding the same failure modes in unlabelled data that the held-out set exposes
independently is the evidence that the acquisition function targets the right
blind spot rather than merely finding noise.

The cheap screen only nominates; the strong two-model jury adjudicates, and the
labelling step reports how often the screen's flags survived that adjudication.

### Disagreement acquisition — enriched, but noisier than the margin route

Jury adjudication of the 2,070 disagreement-selected rows:

| | |
|---|---:|
| juror agreement on these prompts | 82.3% |
| **jury contradicts the MODEL** | **563 / 2,059 = 27.3%** |
| **cheap screen agreed with the jury** | **511 / 2,059 = 24.8%** |

Two readings, both worth keeping.

**It works, at about half the efficiency of the margin route.** Against a ~13%
base error rate on random rows, 27.3% is roughly 2.1x enrichment — real, but
below the 55.4% (4.3x) that low-margin selection achieved. The value is not
efficiency, it is REACH: these are confident errors, and no margin-based method
can surface them.

**The cheap screen is a weak nominator.** Only 24.8% of its flags survived
adjudication, so three quarters of the disagreements were the screen being wrong
rather than the model. That is affordable here because the screen is Haiku and
the jury is what actually assigns labels, but it means the screen cannot be used
as a labeller — only as a funnel, and an expensive one in jury budget per useful
row. A stronger screen, or an ensemble-disagreement signal using the models
already trained, would likely nominate better.

Yield: 1,695 agreed + 364 contested = 2,059 rows, all from the confident region.

## 23. Eval ceilings, measured per signal — the target is not the same everywhere

Same two-stratum blind adjudication as §19, applied to cost:

| | complexity | **cost** |
|---|---:|---:|
| gold errors among the model's mistakes | 16.7% | 13.9% |
| gold errors among the model's correct rows | 3.3% | **5.0%** |
| **total gold noise** | 4.9% | **6.5%** |
| **ceiling** (perfect classifier vs this gold) | 0.9514 | **0.9350** |
| best measured | 0.9043 | 0.8314 |
| estimated TRUE accuracy | ~0.875 | **0.8132** |
| **remaining headroom** | +4.7 | **+10.4** |

Cost's gold is the noisiest, which is consistent from an independent direction:
its two labellers agreed only 74.6% of the time, against 86.9% for complexity
and 93.1% for sensitivity. A signal whose judges disagree produces a noisier
gold set, and both numbers say the same thing about the same taxonomy.

**Cost has more than twice complexity's headroom.** Its 0.8314 is a model
failure, not a measurement ceiling — which makes it the right place to spend
effort, and reverses the intuition that the highest number is the most promising
one to push.

The adjudicator's examples show why cost is hard, and are uncomfortable reading:
`"List of 60 prompts to create Dragon coloring book images for toddlers"` is
labelled MODERATE, and the model said HIGH — "quantity of 60 items signals bulk
enumeration", which is the rubric's own HIGH rule applied more faithfully than
the jury applied it. Several of cost's gold errors are the jury failing to
follow the rubric it was given, not genuine ambiguity.

### Consequence for the goal

"High 90s" is unreachable on any of these evals as they stand, and the ceiling
differs per signal: 0.9514 for complexity, 0.9350 for cost. Reaching high 90s
would require re-labelling the gold sets to remove 5-6.5% noise — which improves
the MEASUREMENT and not the router. The honest targets are:

| signal | now | ceiling | realistic goal |
|---|---:|---:|---|
| complexity | 0.9043 | 0.9514 | close the 4.7 |
| cost | 0.8314 | 0.9350 | close the 10.4 — the biggest prize |
| sensitivity | 0.7312 | not yet measured | measure, then close |

## 24. Techniques taken from vLLM Semantic Router and the local SDG project

Read vSR's `ft_linear_lora.py` rather than its docs. Three things it does that
this project had not:

- `weight_decay=0.1` — ten times what I used, commented there as "higher weight
  decay for better regularization"
- `lr_scheduler_type="cosine"` with `warmup_ratio=0.06` (I used linear / 0.10)
- **LoRA** (`TaskType.SEQ_CLS`, rank/alpha/dropout) as the finetuning mechanism

All three are now queued against cost, the signal with the most headroom. LoRA
is worth more here than in general: cost's gold carries 6.5% label noise and
full fine-tuning has ample capacity to MEMORISE that noise, whereas LoRA trains
only the adapter.

Notably vSR does nothing about class imbalance or label noise, so the
soft-target and ordinal work here extends their recipe rather than duplicating it.

### Semantic dedup — checked, and correctly not applied

The local SDG project filters near-duplicates at cosine > 0.95; this project
deduped only by exact hash. Measured before adopting it:

| corpus | near-dup > 0.95 | > 0.99 |
|---|---:|---:|
| cost-real (WildChat) | **9.2%** | 3.0% |
| cost-v2 (synthetic) | 2.5% | 0.2% |
| sensitivity-v2 (synthetic) | 1.1% | 0.3% |
| sensitivity-enterprise | **0.0%** | 0.0% |

The high rate is in REAL traffic, where repeats are genuine signal — real users
do ask similar things, and removing them would distort the class prior the model
has to learn. The local project deduped SYNTHETIC data, where repeats are
generation artifacts; my synthetic rate is 1-2.5%, so there is little to gain.
Not adopted, for a reason rather than by omission.

The enterprise corpus at 0.0% is incidental validation of the unconditioned
role x setting x moment generation design: it produced no near-duplicates at all.

## 25. Label-noise literature argues against the obvious next move

With gold noise measured at 4.9-6.5%, filtering mislabelled training rows
(Confident Learning, co-teaching) is the obvious thing to try. The evidence says
not to:

- **"On large corpora with a low error rate (< 4%), filtering does not improve
  classification quality."** Confident Learning's reported win (+0.0134 F1) came
  on a SMALL corpus with 35.5% noise. These corpora are large (20-35k rows) with
  low noise — the regime where filtering is reported not to pay.
- *Learning with Confidence: Training Better Classifiers from Soft Labels*
  (arXiv 2409.16071) endorses what is already implemented here: soft targets
  built from annotator disagreement are the recommended treatment for this
  regime.

So the per-example vote distributions in §15 are the literature's answer rather
than a stand-in for it, and Confident Learning / co-teaching are decided against
on evidence rather than left untried. LoRA remains the right lever because it
constrains CAPACITY instead of trying to identify individual bad labels.

## 26. Two label-audit protocols disagree, and the paired one is biased

§19 estimated 4.9% gold noise for complexity by BLIND PAIRED adjudication: the
judge sees a prompt and two candidate labels in random order and picks the
better. I then re-adjudicated the entire set a second way — model-blind, from
scratch, batches of 5 instead of 20, `effort="high"`, and each juror required to
cite the rubric clause its decision rests on.

The two disagree sharply:

| protocol | result |
|---|---|
| blind paired choice (§19) | **4.9%** of labels judged wrong |
| full model-blind re-adjudication | **1.1%** of labels changed (4 of 376) |

**The paired protocol is biased upward.** Presenting a plausible alternative
invites the judge to second-guess a label that was correct; labelling from
scratch mostly reproduces the original. Both are "blind" in the sense that
provenance is hidden, but only one of them avoids anchoring the judge on a
specific competing answer.

Consequences, and the second is a correction to this report:

- The refinement's real value was not fixing labels. It was **removing 42 rows
  (10%) that lost unanimity under careful adjudication** — prompts three careful
  jurors cannot agree on are not a fair test of a router.
- **The 0.9514 ceiling in §19 was too pessimistic.** Real gold noise is nearer
  1-2%, so the ceiling is nearer 0.98, and there is roughly 7 points of genuine
  MODEL headroom on complexity rather than the ~2 that §19 implied. That reverses
  the conclusion I drew from it: continued work on complexity is worthwhile, not
  capped.

The two protocols should have been cross-checked against each other before a
ceiling was published from either.

Re-measured against the refined gold (376 rows):

| model | v1 gold | refined | delta |
|---|---:|---:|---:|
| cx-b4-soft-512-mbert | 0.8989 | 0.8936 | -0.005 |
| cx-d2-everything | 0.8883 | 0.8883 | 0.000 |
| cx-b1-hard-256 | 0.8803 | 0.8697 | -0.011 |
| **ensemble b4 + d2** | — | **0.9122** | — |

Accuracy did not rise, which is the expected result once only 1.1% of labels
moved: the dropped rows were mostly ones the models got right.

## 27. Two self-inflicted pipeline faults, both worth recording

**A train/eval mismatch I created while fixing coverage.** To get NEVER_EGRESS
into TRAINING (5 rows -> 675) I steered the generator toward credential-prone
moments — rotating an expired credential, cleaning up an old `.env`. The
enterprise EVAL was generated from generic workplace moments with no steering.
Training therefore became denser in secrets situations than the eval, and adding
those 4,000 rows moved enterprise accuracy DOWN (0.6761 -> 0.6505) while real
traffic and the legacy set both improved. This is the same distribution-mismatch
failure the project has been documenting throughout, committed while fixing a
different problem. The remedy is an eval slice covering the same situation
space, not less training data.

**Refusals were being silently absorbed.** A juror returned
`stop_reason=refusal` on a WildChat prompt; the retry path treated it as
transient, burned six attempts and killed a ~400-call adjudication. Refusal is
now a terminal error that the batch bisects around, costing one row instead of
the run. Making refusals visible immediately exposed a second instance: the
enterprise-secrets generator was refusing outright, because the brief instructed
it to paste credential-like material. Rewriting the brief to ask for realism
with explicit placeholders — the SITUATION already implies the material —
dropped refusals to zero and produces an eval that does not contain
realistic-looking fake secrets, which is better on its own terms.

Any pipeline reading real traffic has to survive prompts a model will not
answer. That should have been the default rather than the second fix.

## 28. A silent harness bug: soft targets never reached the loss

`transformers.Trainer` defaults `remove_unused_columns=True`, which DROPS any
dataset column that is not a parameter of `model.forward()`. The `soft` column
is not a model input, so it was deleted before collation, the custom collator
never saw it, and `compute_loss` fell back to plain cross-entropy.

**Every `--soft` and `--ordinal` arm in this project trained on hard labels.**

Corrections this forces to earlier sections:

| previously reported | actually |
|---|---|
| §15: soft targets bought +1.4 (0.8636 -> 0.8780) | the gain was real but came from ADDING 2,358 contested rows to the corpus, not from the loss |
| §18: "the soft-target register trade was capacity, not the loss" | that reasoning rested on a loss that never ran |
| §16 follow-up: "ordinal smoothing is too weak to flip decisions" | the flag never applied |

The failure was silent — no warning, plausible numbers throughout. It surfaced
only because the ordinal and non-ordinal sensitivity arms produced BYTE-IDENTICAL
probabilities (mean |diff| = 0.00000), which is impossible if both trained as
configured. A flag that changes nothing is indistinguishable from a flag that
does not help, and I had already published the latter conclusion.

Fixed two ways: `remove_unused_columns=False` when soft targets are in use, and
`compute_loss` now RAISES if the targets are missing rather than degrading
quietly. Round H re-runs paired hard/soft arms on all three signals so the two
effects can finally be separated.

## 29. Sensitivity, measured where it actually matters

The original enterprise eval had 5 NEVER_EGRESS rows and could not measure the
tier that gates egress. The new enterprise-SECRETS slice — unconditioned
generation over credential-handling situations, blind three-model jury, disjoint
roles/settings/seeds — has 834 unanimous rows: INTERNAL 425, REGULATED 182,
NEVER_EGRESS 166, PUBLIC 46, CONFIDENTIAL 15.

| model | acc | macro F1 | PUBLIC | INTERNAL | CONFID | **REGULATED** | **NEVER_EGRESS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| se-c1-minilm | 0.7830 | 0.6528 | 0.67 | 0.94 | 0.33 | **0.51** | 0.75 |
| sensitivity-head-mbert-mix | 0.7818 | 0.6338 | 0.61 | 0.93 | 0.27 | **0.46** | **0.85** |
| sensitivity-head-minilm-mix | 0.7470 | 0.5877 | 0.48 | 0.94 | 0.20 | 0.46 | 0.69 |
| sensitivity-head-minilm-ent | 0.5180 | 0.1509 | 0.00 | 1.00 | 0.00 | 0.04 | 0.00 |

**About half of REGULATED content and 15-31% of NEVER_EGRESS would get through
the gate.** Accuracy of 0.78 hides this entirely: the models are carried by
INTERNAL recall of 0.93-0.94 on the largest class. For an egress gate, per-tier
recall is the metric and aggregate accuracy is close to meaningless.

ModernBERT trades REGULATED recall for NEVER_EGRESS recall (0.46/0.85 versus
0.51/0.75) — a genuine safety-posture choice rather than a strict ordering.

The enterprise-only arm is the cautionary one: 1.00 recall on INTERNAL and 0.00
on everything else. It learned to answer INTERNAL to everything, which scores
0.518 accuracy and is useless.

## 30. Cost gold refinement confirms the complexity pattern

Full model-blind re-adjudication of cost's 427 gold rows: **371 unanimous, 56
dropped, 7 labels changed (1.9%)**. Changes were MINIMAL->LOW (3),
HIGH->MODERATE (2), LOW->MODERATE (1), MODERATE->LOW (1) — all single-tier
slips, consistent with §16.

This reproduces the complexity result (1.1% changed) and confirms §26: the
paired-choice audit overestimates label noise. Cost's ceiling is therefore
nearer 0.97-0.98 than the 0.9350 estimated in §23, and its headroom is larger
than reported, not smaller.

## 31. The data axis has saturated; the encoder axis has barely been explored

| corpus | rows | real-gold |
|---|---:|---:|
| v2 + real | 21,171 | 0.8636 (hard) / 0.8780 (b2) |
| + active-selected | 30,157 | 0.8756 |
| + disagreement-selected | 34,569 | **0.8684** (worse) |

Adding the 2,059 disagreement-selected rows COST a point. That is consistent
with their measured quality rather than surprising: the cheap screen that
nominated them had only 24.8% precision, so three quarters of its flags were the
screen being wrong rather than the model. Those rows carry proportionally more
label noise than the rest of the corpus, and at 6% of the total that is enough
to hurt.

Read together with §17 (active selection lifted contested accuracy but not gold)
and §20 (32 of 48 residual errors are high-confidence), the pattern is clear:
**more data has stopped being the lever.** 5.5k -> 21k helped; 21k -> 34.5k hurt.

Meanwhile every arm in this project has used one of two encoders — MiniLM-L6
(23M) or ModernBERT-base (149M). The base-model axis is the one that has barely
been varied, and §18 showed it is the only axis where architecture ever bought
anything (ModernBERT taking the soft-target gain without the register penalty).

Round I therefore holds the corpus at the best-performing 21k configuration and
varies the encoder instead: `bge-base-en-v1.5`, `e5-base-v2`,
`all-mpnet-base-v2` — all strong retrieval encoders never tried here with a
classification head — plus `ModernBERT-large` to test whether the base-size gain
continues.

A caveat to carry: the disagreement rows should not simply be deleted. They
target the confident-error population that nothing else reaches (§22), and the
right fix is a better nominator — an ensemble-disagreement signal from the four
models already trained costs no API calls and would be far more precise than a
single cheap screen.

### Soft targets, finally measured (complexity, 31,112 rows, paired)

| eval | hard | soft | delta |
|---|---:|---:|---:|
| heldout-v1 | 0.8875 | 0.8875 | 0.000 |
| real-gold (unanimous) | 0.8684 | 0.8612 | **-0.007** |
| real-contested (2-1 splits) | 0.6193 | **0.6477** | **+0.028** |
| val-internal | 0.8230 | 0.8264 | +0.003 |

Soft targets are a TRADE, not a win, and exactly the trade the mechanism
predicts: training toward a split target teaches hedging near boundaries, which
helps on prompts the jurors disagreed about and costs a little on prompts they
agreed about.

This confirms the §28 correction empirically. The +1.4 originally credited to
soft targets came from ADDING the contested rows to the corpus; the loss itself
is neutral-to-slightly-negative on unanimous gold.

Which arm to prefer depends on what is being served. Real traffic contains both
populations, so the choice is between a model that is slightly better on clear
prompts and one that is meaningfully better on ambiguous ones — and for a router
that hedges toward a larger model when unsure, the second is arguably safer.

Open prediction: soft targets should help MORE where labels are noisier. Cost has
the lowest juror agreement of the three (74.6%) and the noisiest gold, so
`co-h1-hard` vs `co-h2-soft` is the sharper test of the mechanism.

## 32. Ensembling helps most where it matters most (sensitivity's egress tiers)

Free — inference only, no training.

**sensitivity, enterprise-secrets eval (n=834):**

| | best single | ensemble | delta |
|---|---:|---:|---:|
| accuracy | 0.7581 | **0.7793** | +0.021 |
| macro F1 | — | 0.6958 | — |
| CONFIDENTIAL recall | 0.33 | **0.56** | +0.23 |
| **REGULATED recall** | 0.51 | **0.61** | +0.10 |
| **NEVER_EGRESS recall** | 0.75 | **0.88** | +0.13 |

The accuracy figure badly understates this. For an egress gate, going from
1-in-4 credentials slipping through to 1-in-8 is worth far more than two points
of aggregate accuracy — and REGULATED, the largest sensitive tier at n=182,
gains 10 points of recall.

Diversity drives it, and is measurable: ModernBERT vs MiniLM overlap at Jaccard
0.398, and those two form the best pair. The `se-c1` / `se-c2-ord` pair overlaps
at **1.000** — byte-identical predictions, an independent confirmation of the
§28 `remove_unused_columns` bug.

**cost, real-gold (n=427):** 0.8290 -> 0.8361, only +0.007. Members overlap at
Jaccard 0.486 — too similar for averaging to buy much. Per-tier recall exposes
the real weakness: **HIGH recall 0.48** against MODERATE 0.92. Cost's failure is
concentrated on exactly the MODERATE/HIGH boundary the ordinal targets were
designed for, and no ordinal arm has yet run with the loss actually applied.

Practical consequence: the ensemble is the right thing to SHIP for sensitivity —
it is strictly better on every sensitive tier — at 2x inference. Against the
measured ~480 classifications/sec per llm-d-sc replica that is a real capacity
decision, but for a security control the recall is worth it.

## 33. Soft targets help or hurt depending on the signal's LABEL NOISE

Paired arms, same corpus and seed, soft targets finally reaching the loss:

| eval | complexity hard | complexity soft | cost hard | cost soft |
|---|---:|---:|---:|---:|
| heldout-v1 | 0.8875 | 0.8875 | 0.9500 | 0.9500 |
| real-gold | 0.8684 | 0.8612 (**-0.007**) | 0.8337 | **0.8501** (+0.016) |
| refined-gold | — | 0.8750 | 0.8491 | **0.8760** (+0.027) |
| real-contested | 0.6193 | **0.6477** (+0.028) | 0.5774 | 0.5119 (**-0.066**) |

**The two signals give mirror-image answers**, and the distinguishing factor is
label noise — as predicted in §33's precursor before these arms ran.

- **cost** has the lowest juror agreement (74.6%) and the noisiest gold. There,
  softening targets acts as REGULARISATION against memorising label noise, and
  it pays on the clean unanimous set (+2.7 on refined gold).
- **complexity** has cleaner labels (86.9% agreement). There, softening mostly
  teaches hedging, which pays only on genuinely ambiguous prompts (+2.8 on
  contested) and costs a little on clear ones.

So "do soft targets help?" has no signal-independent answer. It depends on how
noisy that signal's labels are, which is now measured for all three.

Cost's best is **0.8760 on refined gold**, up from 0.8314.

### Cost's weakest tier is also its least measurable

Per-tier recall, `co-h2-soft` on refined gold: MINIMAL 0.90, LOW 0.85,
MODERATE 0.90, **HIGH 0.62**. Soft targets lifted MINIMAL (+0.04) and LOW
(+0.07) but left HIGH untouched.

HIGH has only **13 rows** in the refined gold, so 0.62 is 8/13 with a 95% CI of
roughly 0.36-0.83. It is the rarest tier in real traffic (227 of 6,000 screened,
3.8%) and the one the ordinal targets are aimed at.

Any HIGH improvement therefore needs the treatment sensitivity's egress tiers
got in §29: a targeted eval slice with enough HIGH examples to separate signal
from noise. Reading a two-row swing on n=13 as a result would repeat exactly the
mistake this project has spent its time documenting.

### Ordinal smoothing: a well-motivated hypothesis that did not pay

First test where the loss actually applied (cost, paired against `co-h2-soft`):

| eval | soft | soft + ordinal |
|---|---:|---:|
| real-gold | 0.8501 | 0.8595 (+0.009) |
| **refined-gold** | **0.8760** | 0.8733 (-0.003) |
| real-contested | 0.5119 | 0.5238 (+0.012) |
| HIGH recall (refined) | 0.62 | **0.54** |

Neutral, and it did NOT fix the MODERATE/HIGH boundary it was designed for.
Refined gold is the better measuring stick and shows a small loss; quoting the
+0.009 from real-gold alone would be picking the favourable eval.

The motivation was sound — 78% of errors are adjacent-tier (§16), cost's tiers
are genuinely monotone, and the literature supports ordinal-aware objectives.
The probable defect is that the smoothing is SYMMETRIC: it leaks MODERATE mass
toward HIGH (helpful) and equally toward LOW (harmful), so the net effect on a
rare tier cancels. An asymmetric variant that leaks only UPWARD would match the
operational asymmetry, and that is what `--escalate` does for sensitivity —
where the same idea gets its second test.

Cost's best remains soft targets WITHOUT ordinal: **0.8760** on refined gold.

## 34. Cost's "HIGH-tier weakness" was measurement noise

Built a cost eval slice enriched around the volume boundary — unconditioned
generation over work situations involving large material volumes, deliberately
mixed with small ones so the boundary is exercised rather than made trivial.
727 unanimous rows: LOW 326, MODERATE 161, **HIGH 135**, MINIMAL 105.

| eval | HIGH rows | HIGH recall |
|---|---:|---:|
| refined gold | 13 | 0.62 |
| **volume eval** | **135** | **0.86** |

**The 0.62 was 8/13**, comfortably inside the CI flagged at the time. On real
support the tier is fine.

Two consequences, and the second is a process failure worth owning:

- The premise "cost's failure is concentrated on the MODERATE/HIGH boundary" was
  **false**. On the volume eval the weak tiers are MINIMAL (0.61) and MODERATE
  (0.66); HIGH is among the strongest.
- The ordinal arms were designed to fix that boundary. They came back neutral
  (§33) because there was nothing there to fix. I recorded the n=13 caveat and
  then let the hypothesis drive an experimental round anyway. Building the eval
  BEFORE forming the hypothesis would have saved the round.

Best cost on the volume eval: ensemble of `co-h2-soft` + `cost-head-mbert-mix`
at **0.7992** (macro F1 0.7782), against 0.7840 for the best single member.
Note this eval is deliberately harder than real-gold — it over-samples the
boundary region — so 0.7992 here and 0.8760 on refined gold are consistent.

## 35. Sensitivity: soft targets and class weighting each fix a different failure

With soft targets finally reaching the loss (§28), on the enterprise-secrets
eval (n=707):

| model | acc | macro F1 | CONFID | REGULATED | **NEVER_EGRESS** |
|---|---:|---:|---:|---:|---:|
| se-c1-minilm (pre-fix) | 0.7581 | 0.6528 | 0.33 | 0.51 | 0.75 |
| **se-h1-soft** | 0.7595 | **0.6831** | **0.68** | 0.56 | **0.90** |
| se-h2-soft-ord | 0.7638 | 0.6809 | 0.60 | 0.57 | 0.88 |
| se-h3-cw (class-weighted) | 0.7638 | 0.6731 | 0.52 | 0.57 | 0.89 |
| **ensemble h1+h3+mbert** | **0.7864** | **0.7025** | 0.64 | 0.58 | **0.90** |

**NEVER_EGRESS recall 0.75 -> 0.90 and CONFIDENTIAL 0.33 -> 0.68** from soft
targets alone — for an egress gate, credentials escaping at 1-in-10 rather than
1-in-4. Accuracy moved only +0.001, so aggregate accuracy would have shown this
as no change at all.

Class weighting did what it was designed for, on the set where it applies:
**real-gold macro F1 0.4360 -> 0.5326** (+0.10) while accuracy fell slightly
(0.8662 -> 0.8592). That set is 85% PUBLIC, so the gain is entirely in the
minority classes — invisible to accuracy, and the entire point of the mechanism.

The two mechanisms address different failures and do not stack cleanly: class
weighting trades CONFIDENTIAL recall (0.68 -> 0.52) for INTERNAL. The ensemble
recovers most of it.

## 36. Tier-escalated class weighting is the corrected ordinal idea, and it works

`se-h4-cw-esc` — inverse-sqrt class weights with an upward tier escalation —
is the best single sensitivity model produced:

| model | acc | macro F1 | CONFID | REGULATED | NEVER_EGRESS | PUBLIC |
|---|---:|---:|---:|---:|---:|---:|
| se-c1-minilm (session start) | 0.7581 | 0.6528 | 0.33 | 0.51 | 0.75 | 0.74 |
| se-h1-soft | 0.7595 | 0.6831 | 0.68 | 0.56 | 0.90 | 0.69 |
| **se-h4-cw-esc** | **0.7723** | **0.6935** | **0.72** | **0.62** | **0.91** | 0.60 |

Every sensitive tier improved, and the PUBLIC drop (0.69 -> 0.60) is the trade
working as intended: the model over-flags public content in order to catch more
sensitive content. For an egress control that is the correct asymmetry, and it
is exactly what the escalation term encodes.

**This vindicates the ordinal hypothesis that §33 recorded as neutral.** The idea
was right; the implementation was symmetric. Leaking target mass equally in both
directions cancels on a rare tier. Leaking it upward only — matching the
operational asymmetry — pays on every tier that matters.

Across the session, sensitivity's credential-catch rate moved 0.75 -> 0.91 and
CONFIDENTIAL recall more than doubled, while aggregate accuracy moved 0.7581 ->
0.7723. Anyone reading accuracy alone would have seen a 1.4-point change and
missed all of it.

Round J transfers the mechanism to the other two signals. Cost has a real
asymmetry (under-estimating cost under-provisions capacity, worse than
over-provisioning) so it gets the escalated variant; complexity has no such
direction, so it gets plain class weighting for the imbalance alone.

## 37. Mechanisms do not transfer between signals — the PATHOLOGY has to transfer

Tier-escalated class weighting was the best mechanism found for sensitivity
(§36). Applied to cost it is slightly NEGATIVE:

| cost arm | refined-gold | macro F1 |
|---|---:|---:|
| co-h2-soft (plain soft targets) | **0.8760** | **0.7996** |
| co-j2-cw-esc (tier-escalated) | 0.8733 | 0.7890 |
| co-j1-cw (class-weighted) | 0.8679 | 0.7848 |

The reason is that the pathology differs. Sensitivity's real-gold is **85%
PUBLIC**, a 20:1 imbalance, so an unweighted loss is optimised by answering the
majority class — re-weighting fixes a genuine failure. Cost's tiers are far more
balanced (105 / 326 / 161 / 135 on the volume eval), so re-weighting mostly
distorts a distribution the model was already fitting.

Transferring a validated mechanism without first checking that the underlying
PATHOLOGY transfers is a good way to produce a plausible-looking regression.

## 38. LoRA's value here is diversity, not peak accuracy

| cost arm | refined-gold | heldout-v1 | contested |
|---|---:|---:|---:|
| co-h2-soft (full fine-tune) | **0.8760** | 0.9500 | 0.5119 |
| co-g3-lora16 (r=16, wd 0.1, cosine) | 0.8706 | **0.9667** | **0.5595** |

LoRA is not the best single model, but it appears in the best ensemble, and it
holds the best held-out and contested scores. Training only an adapter
regularises differently from full fine-tuning, so its errors fall in different
places — which is what an ensemble needs. Against cost's 6.5% label noise the
capacity constraint also shows up as better performance on the hardest split.

**Best cost: ensemble of `co-g3-lora16` + `co-j2-cw-esc` + `cost-head-mbert-mix`
+ `cost-head-minilm-mix` = 0.8868 on refined gold** (best single 0.8760).
The winning members span LoRA, tier-escalation, ModernBERT and MiniLM — the
diverse set, not the top-4 by individual accuracy.

Cost overall: **0.3910 shipped -> 0.8868**, +49.6 points on real traffic.

## 39. Enough data removes the trade-off that the loss was compensating for

Round K combined the two things that worked on sensitivity — the enterprise
corpus expanded 5,582 -> 13,733 rows, and tier-escalated class weighting.

| model | acc | macro F1 | PUBLIC | CONFID | REGULATED | NEVER_EGRESS |
|---|---:|---:|---:|---:|---:|---:|
| se-c1-minilm (session start) | 0.7581 | 0.6528 | 0.74 | 0.33 | 0.51 | 0.75 |
| se-h1-soft | 0.7595 | 0.6831 | 0.69 | 0.68 | 0.56 | 0.90 |
| se-h4-cw-esc | 0.7723 | 0.6935 | **0.60** | 0.72 | 0.62 | **0.91** |
| **se-k1-big-esc** | **0.7808** | **0.7144** | **0.71** | **0.84** | **0.65** | 0.90 |

**The extra data removed the trade-off escalation had been forcing.**
`se-h4-cw-esc` bought sensitive-tier recall by sacrificing PUBLIC (0.74 -> 0.60):
the model over-flagged public content to catch more sensitive content, which was
the correct trade to make when it lacked evidence. With 2.5x the enterprise data
`se-k1` reaches CONFIDENTIAL 0.84 *and* recovers PUBLIC to 0.71.

Read together: the escalation term was compensating for missing training
support on the rare tiers. Given that support, it no longer has to, and the
asymmetric loss and the data are complements rather than substitutes.

This is also the best sensitivity model on every other eval — heldout-v1 0.9333
(best in the project), enterprise-gold 0.7151, real-gold macro F1 0.6174 — so it
is not trading one distribution against another.

Session totals for the tiers that gate egress:

| tier | start | now |
|---|---:|---:|
| CONFIDENTIAL | 0.33 | **0.84** |
| REGULATED | 0.51 | **0.65** |
| NEVER_EGRESS | 0.75 | **0.90** |

Aggregate accuracy over the same period moved 0.7581 -> 0.7808. Anyone tracking
accuracy alone would have seen 2.3 points and missed all of it.

## 40. Tier-exact recall is the wrong metric for a gate — measure CONTAINMENT

REGULATED's 42 errors on the best model split into two operationally opposite
kinds:

| REGULATED classified as | count | effect if the gate blocks CONFIDENTIAL and above |
|---|---:|---|
| CONFIDENTIAL | 15 | **still blocked** — wrong tier, safe |
| NEVER_EGRESS | 7 | **still blocked** — safe |
| INTERNAL | 18 | **leaks** |
| PUBLIC | 2 | **leaks** |

Tier-exact recall counts all 42 as failures and reports 0.65. Operationally,
100 of 120 REGULATED prompts (83%) are classified at or above the gate and are
blocked. **Exact recall understates deployed safety by 18 points**, because a
router that over-classifies still gates correctly.

So the right pair of numbers per gating threshold is:

- **containment** — fraction of at-or-above-threshold content correctly blocked
  (what leaks)
- **over-block** — fraction of below-threshold content wrongly blocked
  (unnecessary round trips)

| model | block>=CONFIDENTIAL | over-block | block>=REGULATED | over-block |
|---|---:|---:|---:|---:|
| se-c1-minilm (session start) | 0.872 | 0.177 | 0.788 | 0.116 |
| se-h4-cw-esc | 0.895 | 0.163 | 0.805 | 0.092 |
| **se-k1-big-esc** | **0.914** | 0.177 | **0.817** | **0.090** |

**91.4% containment at the CONFIDENTIAL gate**, with over-blocking unchanged at
17.7% — safety gained at no additional cost. At the stricter REGULATED gate,
over-blocking actually FELL from 11.6% to 9.0% while containment rose.

This is the number to put in front of a security reviewer, and it is the first
measurement in this project of what the classifier DOES in deployment rather
than how often it lands on the exact label. It does not change the model; it
changes what the existing model is understood to be worth.

Both metrics belong in the model card: tier-exact recall for the taxonomy, and
containment / over-block for the gate.

## 41. There is no single best sensitivity model — accuracy and containment trade

| option | accuracy | macro F1 | contained (>=CONFID) | over-blocked |
|---|---:|---:|---:|---:|
| se-k1-big-esc (single, escalated) | 0.7808 | 0.7144 | **0.914** | 0.177 |
| 5-model ensemble | **0.8076** | **0.7414** | 0.891 | **0.134** |
| accuracy-calibrated single | 0.8119 | — | 0.853 | 0.102 |

Three independent mechanisms produced the same trade, which makes it a property
of the problem rather than of any one method:

| mechanism | accuracy | containment |
|---|---:|---:|
| post-hoc calibration | +0.031 | **-0.061** |
| ensembling | +0.027 | **-0.023** |
| plain class weighting vs escalation | +0.000 | -0.013 |

Each buys accuracy by shifting predictions toward the majority class, and for an
egress gate that means letting sensitive content through. So the deployment
choice is explicit:

- **gating egress** -> `se-k1-big-esc`, 91.4% contained, at 0.7808 accuracy and
  17.7% over-blocking
- **reporting accuracy** -> the ensemble, 0.8076, at 89.1% contained

Conflating those two is precisely how the original 98.53% became misleading: a
number optimised for one purpose, read as though it answered another.

Round K also settled the mechanism question for sensitivity. On the expanded
corpus, plain soft targets (`se-k3-big`, 0.7836) edge out escalation
(`se-k1-big-esc`, 0.7808) on ACCURACY, while escalation wins on the sensitive
tiers. Class weighting without escalation is worst of the three (0.7680) —
the directional asymmetry, not the re-weighting, is what helps.

Session totals, real traffic:

| signal | shipped | best now | gain |
|---|---:|---:|---:|
| complexity | 0.6411 | 0.9122 | +27.1 |
| cost | 0.3910 | 0.8868 | +49.6 |
| sensitivity | 0.2903 | 0.8076 | **+51.7** |

## 42. Ensemble distillation behaves like extra data, not knowledge transfer

Prediction recorded before the run: the teacher's confidence median was 0.998
with only 2.5% of rows below 0.6, so there was almost no uncertainty structure
to transfer and distillation should act as plain data volume.

| eval | distilled (60k rows) | control (30k rows) | delta |
|---|---:|---:|---:|
| real-gold | **0.8923** | 0.8828 | +0.010 |
| refined-gold | **0.8963** | 0.8936 | +0.003 |
| real-contested | **0.6648** | 0.6477 | +0.017 |
| heldout-v1 | 0.8875 | 0.8875 | 0.000 |

The prediction holds: +0.3 on refined gold is what 30,000 additional rows buys,
not what dark-knowledge distillation typically delivers.

**The control arm is what makes that claim checkable.** Without it, +1.0 on
real-gold reads as distillation succeeding when most of it is volume — the same
attribution trap the `remove_unused_columns` bug set in §28, avoided this time
by design rather than by luck.

Practical outcome regardless of mechanism: `cx-l1-distill` at **0.8963 refined
gold** is the best SINGLE complexity model, ahead of `cx-b4-soft-512-mbert`
(0.8936) — and it is MiniLM, so ~3 ms inference against ~16 ms.

## 43. For ensembles, measure error overlap rather than accuracy

Best complexity ensemble: **0.9149** on refined gold (best single 0.8963,
gain +0.019), per-tier recall SIMPLE 0.92 / MEDIUM 0.92 / COMPLEX 0.84 /
REASONING 0.91 — no weak tier left.

Pairwise error overlap (Jaccard):

| pair | overlap |
|---|---:|
| ModernBERT vs MiniLM arms | **0.34 - 0.36** |
| MiniLM arms vs each other | 0.53 - 0.58 |

The winning five-member set is not the top five by individual accuracy. It spans
ModernBERT and MiniLM, distilled and non-distilled, because those make DIFFERENT
mistakes. Adding a sixth similar member made every ensemble worse.

Architectural diversity is worth more than individual accuracy when selecting
members — the transferable rule is to rank candidates by error overlap with the
incumbent set, not by their own score.

## 44. Specialist binary arbiters make things worse

45 of complexity's 48 real-traffic errors fall on two adjacent pairs, so a
binary classifier trained on one pair should see an easier problem and beat the
4-way head there. Trained both (SIMPLE/MEDIUM on 26,369 rows, MEDIUM/COMPLEX on
21,554) and used them as arbiters: consult a specialist only when the 4-way
model's top two classes are exactly that pair.

| | |
|---|---:|
| base `cx-l1-distill` | 0.8963 |
| with arbiters | **0.8830** (-0.013) |
| arbiter fired on | 286 / 376 rows (76.1%) |
| answers changed | 21 |
| **corrections** | **7** |
| **regressions** | **14** |

Wrong twice as often as right when it intervenes. Two causes, and the second is
the interesting one:

- **The trigger is too permissive.** On a 4-way problem the top two classes are
  an adjacent pair most of the time, so "top two are exactly this pair" fires on
  76% of rows rather than on the genuinely contested minority.
- **The specialist sees an easier problem but a less informative one.** Trained
  only on SIMPLE/MEDIUM rows it never learns what COMPLEX or REASONING look
  like, so it cannot recognise that neither of its two options is right. The
  4-way model's uncertainty between two tiers often encodes information about
  the other two that the specialist has discarded.

That is three architecturally-motivated ideas returning negative or neutral —
ordinal smoothing (§33), disagreement acquisition (§22, half the efficiency of
margin), and specialist arbiters — against data volume and ensembling, which
paid every time. The pattern across this project is that representation-level
changes underperform and distribution-level changes work.

## 45. Distillation gain scales with teacher uncertainty — predicted and confirmed

Prediction recorded before the cost run: complexity's teacher had confidence
median 0.998 (2.5% below 0.6) and distillation behaved as plain extra data;
cost's teacher has median 0.897 with 12.5% below 0.6, so it should gain more.

| signal | teacher median conf | rows < 0.6 | gain on refined gold |
|---|---:|---:|---:|
| complexity | 0.998 | 2.5% | +0.003 |
| **cost** | 0.897 | **12.5%** | **+0.011** |

Confirmed: cost gained roughly 4x what complexity did, from the same technique
and the same 30,000-row pool size.

| cost eval | distilled | control | delta |
|---|---:|---:|---:|
| refined-gold | **0.8841** | 0.8733 | +0.011 |
| real-gold | **0.8618** | 0.8478 | +0.014 |
| refined macro F1 | **0.8025** | 0.7709 | **+0.032** |

Macro F1 moves three times as much as accuracy, which is the signature of dark
knowledge: the teacher's spread carries most information about the minority
tiers, and those are what macro F1 weights.

Practical result: `co-n1-distill` at 0.8841 is a single MiniLM within 0.3 points
of the four-member ensemble (0.8868) — most of the ensemble benefit at a quarter
of the inference cost.

**Operating rule this yields:** before spending on distillation, measure the
teacher's confidence distribution. A teacher above ~0.99 median has little to
transfer and distillation reduces to data augmentation; a teacher with a real
low-confidence tail is worth distilling from.

## 46. Distillation needs an ACCURATE teacher, not just an uncertain one — prediction falsified

§45 derived a rule from cost: distillation gain scales with the teacher's
uncertainty. Applied to sensitivity it predicted the largest gain of the three,
because that teacher has the widest confidence spread (median 0.924, 10.8% of
rows below 0.6). Paired arms, tier-escalation held constant in both:

| arm | entsec acc | macro F1 | PUBLIC | CONFID | REGULATED | NEVER_EGRESS |
|---|---:|---:|---:|---:|---:|---:|
| se-o1-**distill** | 0.7595 | 0.6747 | 0.65 | **0.56** | 0.62 | 0.90 |
| se-o2-**control** | **0.7638** | **0.6842** | 0.63 | **0.64** | 0.61 | 0.91 |

**Distillation HURT** — -0.4 accuracy, -1.0 macro F1, and CONFIDENTIAL recall
fell hardest (0.64 -> 0.56). The prediction was wrong.

The rule was incomplete: it tracked teacher UNCERTAINTY and ignored teacher
ACCURACY. Sensitivity's ensemble scores only ~0.78 on enterprise data, so
roughly 22% of the 9,000 pseudo-labels are simply wrong — and 36% of that pool
is the rare tiers, exactly where the teacher is weakest (REGULATED 0.65). For
complexity and cost the teacher was 0.89-0.91 on the pool it labelled and
injected far less noise.

| signal | teacher acc on pool | rows < 0.6 conf | distillation effect |
|---|---:|---:|---:|
| complexity | ~0.91 | 2.5% | +0.003 (acts as data) |
| cost | ~0.89 | 12.5% | **+0.011** |
| sensitivity | **~0.78** | 10.8% | **-0.004** |

**Corrected rule:** distillation pays when the teacher has a real uncertainty
tail AND is accurate enough on the target pool that its errors do not swamp the
signal. Uncertainty alone predicts the wrong sign, as it did here.

This is the fourth pre-registered prediction in the session; three held and this
one did not. Recording the failure is the point — the earlier rule would have
justified distilling sensitivity again on a larger pool, which would have made
it worse.

`se-k1-big-esc` (0.7808) remains the best sensitivity model; nothing in Round O
displaces it.

## 47. The noise floor is ~1.1 points, and it invalidates most fine-grained claims

Every arm in this project used a single seed, so no run-to-run variance had ever
been measured. Four runs of an IDENTICAL recipe (`cx-l1-distill`, differing only
in seed and data order):

| seed | refined-gold | real-gold |
|---|---:|---:|
| 20260903 | 0.8963 | 0.8923 |
| 11 | 0.8963 | 0.8900 |
| 22 | 0.8910 | 0.8852 |
| 33 | **0.9016** | 0.8923 |
| **spread** | **0.0106** | 0.0072 |
| sd | 0.0043 | 0.0034 |

**Deltas below ~0.011 are not distinguishable from seed variance.** Applying that
to this session's reported results:

| claim | delta | status |
|---|---:|---|
| complexity distillation helps | +0.003 | **NOISE — retracted** |
| ordinal smoothing is neutral on cost | -0.003 | **NOISE — cannot conclude either way** |
| sensitivity distillation hurts | -0.004 | **NOISE — retracted** |
| cost class weighting hurts | -0.008 | **NOISE — retracted** |
| cost distillation helps | +0.011 | **AT the floor — not established** |
| specialist arbiters hurt | -0.013 | marginal, only just above |
| ensembling helps | +0.019 to +0.024 | **holds** |
| real-traffic / enterprise data | +0.25 to +0.50 | **holds, overwhelmingly** |

Two consequences worth stating plainly.

**The distillation rule in §45 and its refutation in §46 are both withdrawn.**
That rule was built from complexity (+0.003) and cost (+0.011) and then
"falsified" by sensitivity (-0.004). All three numbers sit at or inside the
noise floor. I constructed a theory and its refutation out of measurements that
could not support either, and the pre-registration in §45/§46 gave the exercise
a false air of rigour — predicting an outcome does not make an underpowered
measurement conclusive.

**The published complexity model is not the best one.** `cx-p-seed33` reaches
0.9016 on refined gold against the published `cx-l1-distill` at 0.8963 — same
recipe, different seed. Selecting the best of several single-seed runs and
publishing it also means the published number is biased upward relative to what
a fresh run of that recipe would give.

What survives is everything measured in tens of points: the data work, the
evaluation work, ensembling, and the per-tier gate improvements. Those are not
close to the floor. It is the mechanism attributions that do not hold, and they
were the parts stated with the most confidence.

**Process change for anything further: report the median of >=3 seeds, and treat
any single-run delta under 1.1 points as unmeasured.**

## 48. Published figures were inflated by seed selection — corrected to medians

Cost, three runs differing only in seed:

| seed | refined-gold | real-gold |
|---|---:|---:|
| 11 | 0.8760 | 0.8548 |
| 22 | 0.8868 | 0.8618 |
| 33 | 0.8733 | 0.8478 |
| **median** | **0.8760** | **0.8548** |
| spread | **0.0135** | 0.0140 |
| previously published | 0.8841 | 0.8618 |

**Cost's noise floor is 1.35 points, larger than complexity's 1.06** — so each
signal needs its own floor rather than a shared one.

Two corrections follow.

**Selection bias is systematic.** Every model published in this project was
chosen as the best of several single-seed runs, so every published number is
inflated by roughly half a spread. Cost's published 0.8841 sits 0.8 points above
the median of three fresh runs. Corrected by republishing the MEDIAN run with an
explicit variance note on the model card.

**Cost's distillation result is fully retracted.** Its +0.011 gain sits well
inside a 1.35-point floor. §45 built a rule from that number and §46 reported
sensitivity as falsifying it; neither the rule nor its refutation had the
measurement power to stand, and both are withdrawn.

What survives remains everything measured in tens of points — the data work, the
evaluation work, the gate-tier improvements, and ensembling at roughly 2x the
floor.

## 49. The data axis is NOT saturated — 6.5x corpus clears the noise floor

Cost, three seeds each, verdict threshold fixed at 1.35 points before running:

| | refined-gold median | real-gold median |
|---|---:|---:|
| baseline, 6,697 rows | 0.8760 | 0.8548 |
| **6.5x corpus, 43,717 rows** | **0.8976** | **0.8735** |
| **delta** | **+0.0216** | **+0.0187** |

Individual seeds 0.8868 / 0.9003 / 0.8976, all above the baseline median.
**VERDICT: real.** This is the first result in many rounds to exceed its own
measurement noise.

Set against every mechanism tried on the same signal and the same floor:

| lever | effect on cost |
|---|---:|
| **6.5x jury-labelled data** | **+0.0216** (clears) |
| ensemble distillation | +0.011 (at the floor) |
| class weighting | -0.008 |
| vSR regularisation | -0.008 |
| LoRA r=16 | -0.005 |
| ordinal smoothing | -0.003 |
| tier escalation | -0.003 |

Every loss-function and architecture change sits inside the noise. Only data
clears it — the third independent confirmation, after real-traffic training
(+44 points) and the enterprise corpus (+38 on the gate tiers).

Returns are clearly logarithmic: 6.5x the corpus buys ~2 points. But they have
not stopped, and the pool still holds 90,000 harvested prompts against ~30,000
labelled per signal. Extending the same treatment to complexity now.

## 50. ModernBERT loses to MiniLM on sensitivity — the first fully conclusive mechanism verdict

Sensitivity's seed spread is 0.0014, roughly 10x tighter than complexity (0.0106)
or cost (0.0135), so a single run there resolves what needs three seeds
elsewhere. Same corpus, same escalation, same epochs — only the base changes:

| | MiniLM-L6 (23M) | ModernBERT-base (149M) |
|---|---:|---:|
| entsec accuracy | **0.7808** | 0.7694 |
| macro F1 | **0.7144** | 0.6688 |
| PUBLIC recall | **0.71** | 0.57 |
| INTERNAL recall | 0.79 | **0.86** |
| CONFIDENTIAL recall | **0.84** | 0.56 |
| REGULATED recall | **0.65** | 0.62 |
| NEVER_EGRESS recall | **0.90** | 0.80 |
| CPU p50 | **3 ms** | 15.1 ms |
| training time | ~9 min | **49 min** |

The -1.1 accuracy gap is 8x the noise floor, so this is not ambiguous: a model
with 6.5x the parameters is worse on every sensitive tier, 5x slower to serve
and 5x slower to train.

The failure mode is visible in the per-tier split. INTERNAL recall ROSE to 0.86
while CONFIDENTIAL collapsed from 0.84 to 0.56. On a corpus this skewed, the
extra capacity was spent fitting the majority class rather than discriminating
boundaries — precisely what the tier-escalation loss exists to prevent, and the
larger model overwhelmed it.

This also revises §18, which credited ModernBERT with taking the soft-target
gain "without the register penalty" on complexity. That was a single-seed
observation of +0.5 against a 1.06-point floor, so it never established
anything. Where the measurement IS conclusive, bigger is worse.

**Practical consequence: all three published models are MiniLM-L6 at ~3 ms.**
There is no accuracy argument for the larger encoder on any signal, and a clear
throughput argument against it — llm-d-sc serves the classifier on CPU, where
the measured per-replica ceiling is ~480 classifications/sec.

## 51. Data scaling pays only where LABEL NOISE is high — not unconditionally

Two corpus-scaling experiments, three seeds each, thresholds fixed in advance:

| signal | juror agreement | corpus scaled | median delta | floor | verdict |
|---|---:|---|---:|---:|---|
| cost | **73.4%** | 6,697 -> 43,717 (6.5x) | **+0.0216** | 0.0135 | **REAL** |
| complexity | **87.5%** | 15,600 -> 52,068 (3.3x) | +0.0026 | 0.0106 | within noise |

**This corrects a conclusion repeated throughout this report.** I had been
generalising "data wins, mechanisms do not" from cost's result. The truer
statement is that **data wins where label noise is high enough that averaging
still pays.**

Cost's labellers disagree on 27% of rows, so each additional row cancels some of
that error and the corpus had real headroom. Complexity's disagree on 12.5% —
already near its label-quality ceiling — so 3.3x more rows bought nothing
measurable.

That converts an empirical rule of thumb into something predictive: **measure
juror agreement before commissioning more labels.** High disagreement means more
data will help; high agreement means it will not, and effort should go to
cleaning labels or changing the taxonomy instead.

Falsifiable prediction for Round U: sensitivity's enterprise corpus sits at 78%
agreement, between the two, so its scaling gain should fall between +0.0216 and
+0.0026.

### A selection-bias trap avoided

One complexity seed reached 0.9149 refined / 0.9043 real — the best single-model
figures in the project. The median across three seeds is 0.8989. Publishing that
seed would repeat exactly the bias corrected in §48, where the published cost
model sat 0.8 points above a fresh median. The published complexity model stays
at the median.

## 52. The label-noise rule confirmed on a third signal — and it is prospective

Round U tested the prediction from §51 on sensitivity, whose enterprise corpus
sits at 77.9% juror agreement, between cost (73.4%) and complexity (87.5%).
Predicted gain: between +0.0026 and +0.0216.

| signal | juror agreement | corpus scaled | median gain | floor | clears? |
|---|---:|---|---:|---:|---|
| cost | 73.4% | 6.5x | +0.0216 | 0.0135 | **yes** |
| **sensitivity** | **77.9%** | **2.3x** | **+0.0227** | **0.0014** | **yes, 16x** |
| complexity | 87.5% | 3.3x | +0.0026 | 0.0106 | no |

Measured +0.0227 — marginally above the predicted band, but the ORDERING is
exactly right and both low-agreement signals cleared their floors while the
high-agreement one did not.

**This is now a prospective rule rather than a retrospective story: measure
juror agreement before commissioning labels.** Below roughly 80% agreement,
more of the same data still pays. Above roughly 85% it does not, and the budget
belongs in label cleaning or taxonomy revision instead.

Sensitivity's +2.27 is 16x its noise floor — the largest verified single
intervention in the project, and it came from data on the corpus that matters
for that signal rather than from any change to the model.

Two seeds: 0.7793 / 0.7822. Published the median-equivalent run with the spread
stated on the card.

## 53. Complexity is at its TAXONOMY ceiling — both levers exhausted

| lever | median delta | floor | verdict |
|---|---:|---:|---|
| 3.3x more data (15,600 -> 52,068 rows) | +0.0026 | 0.0106 | within noise |
| tie-resolved hard labels, row count held CONSTANT | +0.0027 | 0.0106 | within noise |

The second arm is the clean one: identical rows in both, differing only in
whether 7,218 previously-contested rows train as hard 2-of-3 majorities or as
50/50 soft targets. Label quality isolated from volume, and neither moves it.

Combined with sixteen mechanism-level interventions that all measured inside
their floors, **complexity at 87.5% juror agreement is at its taxonomy ceiling.**

This completes the label-noise rule across three signals and both branches:

| signal | agreement | prescription | tested | result |
|---|---:|---|---|---|
| cost | 73.4% | more data | yes | **+0.0216** |
| sensitivity | 77.9% | more data | yes | **+0.0227** |
| complexity | 87.5% | more data | yes | +0.0026 (no) |
| complexity | 87.5% | clean labels | yes | +0.0027 (no) |

### Where complexity's remaining headroom actually is

Juror disagreement by boundary, normalised for how often each pair ARISES
(tier prior: MEDIUM 35,563 / SIMPLE 12,687 / COMPLEX 2,042 / REASONING 1,776):

| boundary | share of disagreement | expected from frequency | ratio |
|---|---:|---:|---:|
| SIMPLE/MEDIUM | 62.8% | 70.6% | **0.89** |
| **MEDIUM/COMPLEX** | **25.3%** | **11.4%** | **2.22** |
| COMPLEX/REASONING | 1.3% | 0.6% | 2.17 |
| MEDIUM/REASONING | 7.4% | 9.9% | 0.75 |
| SIMPLE/REASONING | 3.2% | 3.5% | 0.91 |
| SIMPLE/COMPLEX | 0.1% | 4.1% | 0.02 |

SIMPLE/MEDIUM dominates raw counts only because those tiers are 93% of traffic;
per opportunity jurors handle it slightly BETTER than chance. **MEDIUM/COMPLEX
draws 2.2x more disagreement than its frequency predicts.**

(This corrects a claim made one round earlier in this session, which read the
absolute counts and concluded SIMPLE/MEDIUM was the defect. That was a base-rate
error.)

MEDIUM/COMPLEX is the boundary the rubric defines most loosely — "one
conventional deliverable" versus "system design with competing tradeoffs" — and
the one that had to be rewritten during rubric calibration (§5) after the first
draft scored 0.8875 against gold.

**Recommendation: complexity's remaining headroom is in the RUBRIC, not the
model, the data, or the labels.** Either sharpen the MEDIUM/COMPLEX rule with
operational tests a labeller can apply mechanically, or merge the two tiers if
the routing decision does not actually differ between them.

## 54. The rubric is the largest lever in the project — +7.0 points of labeller agreement

§53 concluded complexity was at its taxonomy ceiling and pointed at
MEDIUM/COMPLEX, which drew 2.2x more disagreement than its frequency predicted.
The v1 wording asked labellers to judge whether "competent experts would return
materially different good answers" — a counterfactual about people rather than
an inspection of the request.

v2 replaces it with a countable test on features visibly present in the text.
COMPLEX requires **two or more** of:

  T1  names 2+ components/services/stages that must interact
  T2  asks to choose between named alternatives
  T3  competing objectives that trade against each other

A/B on the 600 prompts where v1 labellers split MEDIUM/COMPLEX — same prompts,
same two labellers, only the wording differs:

| wording | agreement |
|---|---:|
| v1 "experts would differ" | 71.3% |
| **v2 countable 2-of-3 test** | **78.3%** |
| **delta** | **+7.0 points** |

MEDIUM/COMPLEX splits fell from 157 to 112, a 29% reduction on the hardest
prompts in the corpus.

**This is the largest single effect measured in the project, and it is not a
model change.** For scale, on complexity:

| lever | effect |
|---|---:|
| **rubric rewrite** | **+7.0 pts of labeller agreement** |
| 16 mechanism/architecture changes | all inside a 1.06 pt floor |
| 3.3x more training data | +0.26 (inside floor) |
| tie-resolved labels | +0.27 (inside floor) |

The rubric sits upstream of everything: eval gold, training labels and the model
all inherit that definition, so raising agreement raises the ceiling for all of
them at once. Every other lever was operating below a ceiling this one sets.

The mechanism is worth stating generally: **a labelling rule that asks an
annotator to simulate other annotators cannot be applied consistently, because
there is no shared referent to check against.** Replacing it with countable
features of the input removes the disagreement at its source. These 600 prompts
were the hardest in the corpus — rows where v1 labellers had already split — so
+7.0 is a lower bound on the corpus-wide effect.

Next: relabel under v2 and retrain, which is the first change in many rounds
with a mechanism large enough to clear complexity's noise floor.

## 55. Sensitivity is model-limited, not rubric-limited — the opposite of complexity

§54's rubric lever was large enough to be worth trying everywhere, so before
spending it on sensitivity (the weakest signal at 0.7793) the question was
whether sensitivity has the same disease. It does not.

**Diagnostic 1 — where jurors disagree, normalised by frequency** (n=949 entsec
rows with 3-juror votes; 25.5% of rows split, vs complexity's concentrated split):

| boundary | splits | over-rep |
|---|---:|---:|
| INTERNAL/CONFIDENTIAL | 58 | 1.68x |
| PUBLIC/INTERNAL | 63 | 1.67x |
| CONFIDENTIAL/REGULATED | 22 | 1.59x |
| INTERNAL/REGULATED | 62 | 1.50x |
| INTERNAL/NEVER_EGRESS | 17 | 0.42x |
| REGULATED/NEVER_EGRESS | 2 | 0.10x |

Unlike complexity — one boundary at 2.2x — sensitivity's disagreement is
**diffuse across four boundaries at 1.5-1.7x, all of them in the bottom half of
the ladder**. Worth stating for the product: the security-critical boundaries are
the CLEAN ones (REGULATED/NEVER_EGRESS 0.10x). The mush is at PUBLIC/INTERNAL/
CONFIDENTIAL, where the egress decision is least consequential.

**Diagnostic 2 — is the model wrong where jurors AGREED?** This is the decisive
one, and no amount of relabelling can move it (`se-u-big-seed11`):

| rows | accuracy |
|---|---:|
| unanimous (n=707) | 77.93% |
| contested (n=242) | 51.24% |
| overall (n=949) | 71.13% |
| **ceiling if every contested row were perfectly labelled** | **83.56%** |

The model misses **156 rows where all three jurors agreed — 16.4% of the set**.
Rubric ambiguity cannot explain a single one of them. A rubric rewrite on
sensitivity would be spending §54's lever on the smaller half of the problem.

**And the error has a direction.** Unanimous-row confusions:

| gold -> pred | n |
|---|---:|
| INTERNAL -> CONFIDENTIAL | 40 |
| INTERNAL -> REGULATED | 22 |
| INTERNAL -> NEVER_EGRESS | 19 |
| REGULATED -> CONFIDENTIAL | 18 |
| PUBLIC -> INTERNAL | 17 |
| REGULATED -> INTERNAL | 12 |

**81 of 156 (52%) are INTERNAL escalated upward.** That is not a mystery: the
recipe carries `--escalate 1.0`, which scales class weight upward through the
tier ladder, so INTERNAL — tier 1 of 5 and the majority class at 486/949 —
carries the smallest weight in the objective. The model was trained to do this.

Escalation was a deliberate trade for gate containment, so Round W sweeps
escalate in {0.0, 0.5, 1.0} at 2 seeds and reports accuracy AND containment
together. Reporting accuracy alone would recommend discarding a safety property;
reporting containment alone hides ~8 points of accuracy.

**Method note worth keeping:** the unanimous/contested split is a cheap, general
test for "is my ceiling the labels or the model?", and it gave opposite answers
on two signals in the same corpus. It should be run before any relabelling
campaign, not after.

## 56. The prior mismatch is real but small — sensitivity's error is not a threshold artifact

§55 named `--escalate 1.0` as the cause of the upward bias. The class priors
offer a second, independent cause that needs no loss function at all:

| tier | train% | eval% | ratio |
|---|---:|---:|---:|
| INTERNAL | 35.1% | 51.2% | 1.46x |
| CONFIDENTIAL | 14.4% | 5.8% | **0.40x** |
| NEVER_EGRESS | 10.9% | 15.0% | 1.38x |
| PUBLIC | 23.3% | 11.0% | 0.47x |

CONFIDENTIAL is **2.5x over-represented in training** relative to the evaluation
distribution — exactly the direction of the dominant confusion. A softmax head
trained on one prior and scored on another is miscalibrated by precisely
log(p_eval/p_train) per class (Menon et al., ICLR 2021). That correction has no
free parameter, so applying it is not fitting the eval set.

`se-u-big-seed11`, gate at CONFIDENTIAL+:

| tau | acc | acc (unanimous) | containment | over-block |
|---:|---:|---:|---:|---:|
| 0.0 (baseline) | 71.13% | 77.93% | 83.29% | 20.34% |
| **1.0 (theoretical)** | **72.08%** | **79.49%** | **81.34%** | **18.47%** |
| 2.0 | 72.39% | 79.49% | 79.11% | 16.10% |

**+0.95 accuracy for -1.95 containment.** Real, reproducible, and a bad trade —
the same roughly 1:2 exchange rate the calibration experiment produced. Two
different threshold-moving techniques hitting the same exchange rate is the
finding: **sensitivity's error is not a decision-boundary artifact.** Moving the
boundary relocates errors rather than removing them. Not adopted.

## 57. The hard pair is a REPRESENTATION limit, and a stronger encoder lifts it — §50 was confounded

If thresholds can't fix INTERNAL/CONFIDENTIAL, is the distinction in the data at
all? A frozen-encoder linear probe answers it in the easiest setting the problem
can be posed in: that pair only, classes balanced 3000/3000, no fine-tuning, no
interference from the other three tiers.

| encoder | CV (train) | eval acc | **eval balanced** | CONFIDENTIAL recall |
|---|---:|---:|---:|---:|
| majority-class predictor | — | 89.83% | 50.00% | 0.00% |
| all-MiniLM-L6-v2 (**shipped**) | 82.88% | 77.63% | 64.17% | 47.27% |
| all-mpnet-base-v2 | 84.60% | 81.52% | 67.14% | 49.09% |
| bge-base-en-v1.5 | 84.60% | 80.78% | **72.37%** | **61.82%** |
| e5-base-v2 | 85.02% | 83.92% | **72.51%** | 58.18% |

Two things fall out.

**(1) Report balanced accuracy on this pair, not accuracy.** Eval is 513:61, so a
model that never predicts CONFIDENTIAL scores 89.8% — higher than every real
probe. Plain accuracy would have ranked "refuse to use the class" first.

**(2) A stronger frozen encoder is worth +8.3 points of balanced accuracy and
+14.5 points of CONFIDENTIAL recall over the shipped MiniLM.** Same data, same
classifier, same balance — only the representation changes, so nothing else can
explain it.

**This directly challenges §50 ("bigger is worse on sensitivity").** §50 measured
FINE-TUNED encoders on the skewed 5-way corpus and observed the larger model
spend its capacity on the majority class: "INTERNAL recall ROSE to 0.86 while
CONFIDENTIAL collapsed from 0.84 to 0.56." That is the *skew* failing to be
handled, not the representation failing to carry the distinction — and the two
were never separated. Frozen and balanced, bigger clearly wins on exactly the
discrimination §50 saw collapse.

Round X tests the implication: fine-tune bge-base 5-way with **balanced
sampling** rather than class weights alone. If §50's result was an artifact of
skew, it should reverse. Inference cost (~13 ms vs ~3 ms) is a real deployment
consideration and gets reported with the accuracy, not after it.

## 58. Extractive prompt compression LOSES at inference time (-5.4 on long prompts)

"98x Faster LLM Routing Without a Dedicated GPU" (arXiv 2603.12646) reports
compression *raising* classifier accuracy rather than only speed — domain
classification 53.1% -> 61.2%, PII detection 78.5% -> 92.4% — crediting "lost in
the middle": dropping irrelevant sentences concentrates signal inside the
encoder's attention span. That is directly relevant here, because the shipped
models truncate at 256 tokens and therefore already compress long prompts by the
crudest possible rule: keep the first 256 tokens, discard the rest.

Implemented their pipeline (`harness/compress.py`) — TextRank over TF-IDF
sentence similarity, U-shaped position weighting, TF-IDF density, inverse-
centrality novelty, weights .20/.40/.35/.05, no neural inference. Both arms see
256 tokens; only WHICH 256 differs.

| subset | head-truncate | compress | delta |
|---|---:|---:|---:|
| long prompts only (n=148) | 77.70% | 72.30% | **-5.41%** |
| full entsec set (n=949) | 71.13% | 70.28% | -0.84% |

**Not adopted at inference.** The likely cause is train/test mismatch — the model
learned on head-truncated text — plus a task difference: for sensitivity the
opening sentences usually carry the framing that fixes the tier ("Here is our
customer export…"), so the head is not the wrong 256 tokens to keep.

**A measurement bug worth recording, because it produced a clean and completely
false null first.** The budget was in whitespace words while the model's limit is
in tokens. These prompts run **3.34 tokens per word** — dense with keys, IDs and
code; median long prompt is 105 words but 346 tokens — so a 256-*word* budget
fired on only 20 of 148 over-length prompts. The first run reported exactly
+0.00% on both subsets. An effect of precisely zero to four decimals is not a
null result, it is a no-op, and the difference is worth checking for every time.

**What it does point at.** 15.6% of entsec prompts exceed 256 tokens — nearly
double complexity's 8.1%. §14 tested maxlen 512 and found it "buys nothing", but
that was measured on COMPLEXITY, where the tier is a property of the whole
request. Sensitivity's tier is a property of the most severe span ANYWHERE in the
prompt, so truncation can remove the only token that determines the label. Same
knob, materially different reason to expect it to matter. Queued as Round Y.

## 59. Truncation is NOT hurting sensitivity — Round Y cancelled before it ran

§58 argued maxlen 512 deserved a retest on sensitivity because 15.6% of entsec
prompts exceed 256 tokens and the tier is a max over spans. A two-minute
inference-only check killed it before any training was spent:

| subset | n | @256 | @512 | delta |
|---|---:|---:|---:|---:|
| short (<=256 tok) | 801 | 69.91% | 69.91% | +0.00% |
| long (>256 tok) | 148 | **77.70%** | 76.35% | **-1.35%** |
| all | 949 | 71.13% | 70.92% | -0.21% |

Long prompts score *higher* than short ones, and handing the model the truncated
tail makes it slightly worse. (The long/short gap is partly composition — long
prompts are 37% NEVER_EGRESS vs 11% for short, and NEVER_EGRESS is the easiest
class. The @512 delta is the composition-free number, and it is negative.)

**§14's null holds on this signal too. Round Y deleted unrun.** The general point:
an inference-time probe often settles a training-time question for a thousandth
of the compute, and is worth reaching for before every round.

## 60. Span-max inference: +2 to +3 containment that turns out to be a ROC slide

The residual NEVER_EGRESS misses are not what the taxonomy's examples suggest.
Of the 25 the classifier fails to contain, **only 2 contain a credential
pattern** — a 16-rule high-precision detector (`harness/secret_rules.py`,
0.25% false-positive rate) confirms it. The rest are attorney-client privileged
emails quoted inside casual framing:

> "just summarise this thanks  \"Anil — quick one, off the back of your
> voicemail. You do not need to answer the customer's…\""

The tier lives in the quoted span; the framing is INTERNAL, comes first, and
dominates the pooled representation. That reads as a composition problem
fixable without retraining: score the parts, take the highest tier.

| gate | containment | delta | over-block | delta |
|---|---:|---:|---:|---:|
| CONFIDENTIAL | 85.79% | +2.51 | 23.56% | +3.22 |
| REGULATED | 78.29% | +3.29 | 13.64% | +2.95 |
| NEVER_EGRESS | 84.51% | +2.11 | 5.58% | +1.24 |

Containment up at every gate — and this is where the result would normally be
published. **It should not be**, because containment can always be bought with a
bias on the high tiers. The only meaningful question is whether span-max beats
the curve a simple bias traces, read at MATCHED containment:

| gate | span-max over-block | plain bias at same containment | verdict |
|---|---:|---:|---|
| NEVER_EGRESS | 5.58% | **5.20%** | bias is cheaper |
| REGULATED | **13.64%** | 15.50% | span-max wins by 1.9 |
| CONFIDENTIAL | 23.56% | **22.03%** | bias is cheaper |

**Two of three gates are beaten by a scalar added to three logits.** Span-max is
a ROC slide, not a gain — and it costs 2.51x the forward passes. Not adopted;
tier-exact accuracy also fell 1.16.

**Methodological rule this establishes.** Any intervention that moves the
containment/over-block pair must be reported against a threshold bias matched on
containment. Without that control this experiment reads as "+2.11 containment at
the security-critical gate" and gets shipped. The same control is what makes
§56's rejection of logit adjustment safe: both are threshold moves wearing
different clothes.

## 61. The encoder is a lever on ALL THREE signals, not just sensitivity

§57's frozen-pair probe, extended to every confusable pair. Balanced accuracy,
since three of the four pairs are skewed enough that a majority-class predictor
outscores every real probe:

| signal / pair | MiniLM-L6 | bge-base | e5-base | best gain |
|---|---:|---:|---:|---:|
| sensitivity INTERNAL/CONFIDENTIAL | 64.17% | **72.37%** | 72.51% | **+8.3** |
| complexity MEDIUM/COMPLEX | 75.61% | 76.82% | **78.71%** | +3.1 |
| cost LOW/MODERATE | 72.81% | 76.09% | **76.26%** | +3.4 |
| cost MODERATE/HIGH | 72.45% | **73.48%** | 73.13% | +1.0 |

The shipped encoder is last on every pair. The gain is largest exactly where the
signal is weakest (sensitivity, +8.3) and smallest where §54 says the ceiling is
the taxonomy rather than the representation (complexity, +3.1) — which is the
ordering the two diagnoses predict, and mild evidence that both are right.

## 62. Frozen 5-way probes are weak everywhere — most of the work is the fine-tune

§61's pair probes are a statement about what a representation *separates*, not
about what it *solves*. The 5-way version, same encoders, same frozen features,
20,000 balanced training rows, linear head:

| encoder (frozen + linear) | acc | balanced |
|---|---:|---:|
| all-MiniLM-L6-v2 | 56.06% | 53.52% |
| bge-base-en-v1.5 | 57.11% | 52.19% |
| e5-base-v2 | **60.80%** | **57.29%** |
| *fine-tuned MiniLM-L6 (`se-u-big-seed11`)* | *71.13%* | — |

The fine-tuned 22M model beats every frozen 110M model by 10-15 points. e5 keeps
its edge (+3.8 balanced over MiniLM), so §61's ranking survives, but the size of
the prize shrinks: **fine-tuning contributes more than the encoder choice does**,
and a frozen-probe gain is a lower bound that fine-tuning may erase by adapting
the weaker encoder to the same place.

This is the honest caveat on §61 and it is worth stating before Round X reports,
not after. The pair probes prove the distinction is representable; they do not
prove a fine-tuned bge will realise it in the 5-way setting.

## 63. Minimal-pair synthetic data FAILS to teach the boundary — and its 98.4% agreement is why

§57 localised sensitivity's error to INTERNAL/CONFIDENTIAL and showed the corpus
does not cleanly encode it (82.9% within-train CV on the balanced pair). Minimal
pairs are the textbook instrument for that: two requests identical in topic,
length and tone, differing only in the feature that moves the tier. Generated 576
across 18 enterprise domains with anti-cues in both directions
(`harness/build_boundary_sdg.py`), blind-checked by a second model that never saw
the proposed label.

**Blind-check agreement: 98.4%. 567 items kept, balanced 284/283.** By every
quality signal this project uses, excellent data.

It does not work. Frozen-probe A/B, eval held fixed, only the training pool
changes:

| arm | n | MiniLM bal | e5 bal |
|---|---:|---:|---:|
| A base corpus only, size-matched | 566 | **59.23%** | **65.35%** |
| B boundary SDG only | 567 | 51.19% | 58.82% |
| C base 3000/class + all boundary | 6,567 | 67.07% | 68.85% |
| *(base 3000/class alone, prior run)* | *6,000* | *68.20%* | *70.88%* |

**Row for row, the synthetic minimal pairs are 8.0 (MiniLM) and 6.5 (e5) points
WORSE than real corpus rows**, and adding them on top does not help.

**The 98.4% agreement is not evidence the data is good — it is the explanation
for why it is useless.** Two independent models agree that easily only when an
item is unambiguous. The real INTERNAL/CONFIDENTIAL cases are the ones jurors
split on 1.68x more often than frequency predicts (§55). Data constructed to be
"unambiguous under the rubric" lands in a region of the space where the model
already has no trouble, and displaces rows that carry the hard cases.

This sharpens §51's rule into a generation-side rule: **a synthetic corpus whose
blind-check agreement is far ABOVE the agreement of the real distribution is
off-distribution by construction.** Agreement should be a target band, not a
maximand. Real entsec agreement on this pair is roughly 75-85%; the generator
produced 98.4%, and that gap was measurable before any training was spent.

Boundary corpus retained at `data/train/sensitivity-boundary.jsonl` but NOT
added to any training arm.

## 64. A third of the sensitivity training corpus is off-distribution from the eval

§63's failure raised the obvious follow-up: if synthetic rows can be worse than
no rows, which of the EXISTING corpora are off-distribution too? Measure it
directly — try to tell each training file apart from the eval text, ignoring
tiers entirely. 50% means indistinguishable; higher means a linear probe can
spot the source.

| training file | rows | separability from eval |
|---|---:|---:|
| sensitivity-enterprise.jsonl | 31,168 | **78.3%** |
| sensitivity-distill.jsonl | 8,995 | 81.7% |
| sensitivity-v2-contested.jsonl | 181 | 90.9% |
| sensitivity-v2.jsonl | 6,783 | **92.8%** |
| sensitivity-real-contested.jsonl | 618 | 96.7% |
| sensitivity-real.jsonl | 8,361 | **98.2%** |
| sensitivity-boundary.jsonl (§63) | 567 | 98.2% |

Nothing is at 50%, so no corpus matches the eval, but the spread is 20 points.
The current recipe trains on v2 + real + enterprise + real-contested: **~15,700
of ~47,000 rows (33%) come from files a linear probe separates from the eval
94-98% of the time.**

By tier, the gap tracks the model's trouble:

| tier | separability | model behaviour |
|---|---:|---|
| CONFIDENTIAL | **82.7%** | the pair it cannot learn (§57) |
| NEVER_EGRESS | 75.0% | 82.4% containment |
| INTERNAL | 72.7% | over-predicted, but recall is fine |

(CONFIDENTIAL's figure rests on 55 eval rows, so treat the ordering as
suggestive rather than measured; the file-level table is the solid part.)

**This is not automatically a defect.** `sensitivity-real.jsonl` is WildChat
traffic and SHOULD look nothing like enterprise text — a deployed router sees
both, and entsec is only one of the evals. The finding is that corpus
composition is currently an accident of which generators were run, not a
decision, and it has never been ablated. Round AA ablates it and reports entsec
AND real-gold together, since any gain from dropping the off-distribution rows
is a trade against the distribution those rows represent.

## 65. The v2 rubric's corpus-wide effect is +0.2, not +7.0 — and a data-loss note

59,582 real prompts relabelled from scratch under the v2 complexity rubric, two
labellers, same protocol as v1:

| | v1 (§51) | v2 |
|---|---:|---:|
| two-labeller agreement, corpus-wide | 87.5% | **87.7%** |
| tier mix | — | MEDIUM 36,011 / SIMPLE 13,059 / REASONING 1,697 / COMPLEX 1,445 |

**Essentially unchanged**, against §54's +7.0 on the hard subset. The arithmetic
that should have been done before the relabel: rows where v1 labellers split are
~12% of the corpus, and v2 resolves 29% of them, which predicts corpus agreement
near 91%. Observed 87.7%.

Two explanations, not yet separated:
1. the A/B's 600 rows are not representative of all v1-split rows;
2. the v2 wording resolves some disagreements while opening others — its
   residual table (§54) shows a MEDIUM/SIMPLE confusion that v1 never produced,
   so the countable test may be sharpening one boundary by blurring another.

**§54 stands as measured but must be read narrowly**: +7.0 points of agreement on
the hardest 12% of prompts, not +7.0 on the corpus. The headline claim that the
rubric is "the largest lever in the project" was extrapolated from a subset and
is not supported at corpus scale. Round AB's prospective prediction is therefore
a gain at or below complexity's 1.06-point floor.

**Data-loss note.** The relabel overwrote `data/train/complexity-real.jsonl` in
place, and `data/train/` is untracked, so the v1 labels for these prompts are
gone. The decision-relevant comparison survives — v1-trained models still exist
and can be scored against the same eval — but the label-level diff (which
specific rows changed, and in which direction) cannot be recovered. The v2 corpus
is now written to `complexity-real-v2rubric.jsonl` and referenced by that name so
this cannot recur silently.

## 66. Tier escalation should be REMOVED — it cost 1.9 accuracy and bought ~0.4 containment

§55 predicted this from the direction of the errors: 52% of the model's
unanimous-row mistakes were INTERNAL escalated upward, and `--escalate 1.0`
scales class weight up the tier ladder, making INTERNAL — tier 1 of 5 and the
majority class — the cheapest class to sacrifice. Round W varied only that flag,
two seeds each, and measured accuracy WITH containment as §55 required.

Medians on entsec-gold (n=707):

| | escalate 1.0 | escalate 0.0 | delta |
|---|---:|---:|---:|
| tier-exact accuracy | 78.08% | **79.99%** | **+1.91** |
| CONFIDENTIAL containment | 92.48% | 92.11% | -0.37 |
| CONFIDENTIAL over-block | 19.05% | **16.67%** | **-2.38** |
| REGULATED containment | 82.78% | 82.99% | +0.21 |
| REGULATED over-block | 9.77% | **8.69%** | **-1.08** |
| NEVER_EGRESS containment | 90.50% | 89.67% | -0.83 |
| NEVER_EGRESS over-block | 3.93% | **3.33%** | **-0.60** |

Real-gold is unchanged (87.50% -> 87.33%, inside noise), so this is not a
distribution trade either.

**This is the first intervention on sensitivity that is not a slide along the
containment curve.** §56's logit adjustment and §60's span-max both paid ~2
points of over-block per point of containment. Removing escalation moves
accuracy up 1.9, over-block DOWN 1-2.4 at every gate, and containment by less
than a point in either direction. Escalation was buying a safety property that,
measured, it barely delivers — while charging for it in both accuracy and
unnecessary blocking.

**Recipe change: drop `--escalate` from the sensitivity recipe.** The completed
sweep is monotone, which is what a real dose-response looks like rather than a
lucky seed:

| escalate | seed 11 | seed 22 | median |
|---:|---:|---:|---:|
| 0.0 | 0.7963 | 0.8034 | **0.7999** |
| 0.5 | 0.7907 | 0.7893 | 0.7900 |
| 1.0 | 0.7793 | 0.7822 | 0.7808 |

Every step of the flag costs about a point, in order, across both seeds — 13x the
0.14-point floor end to end.

Worth naming the general trap: escalation was introduced as a *safety* measure
and was never audited against the safety metric it was justified by. It survived
several rounds on that justification alone.

## 67. bge-base wins on accuracy but only wins the NEVER_EGRESS gate — §50 partially overturned

`se-x1-bge-raw` (bge-base, raw corpus, escalate 1.0, 3ep/3e-5):

| eval | MiniLM incumbent (median) | bge-base | delta |
|---|---:|---:|---:|
| entsec-gold | 78.08% | **81.19%** | **+3.11** |
| enterprise-gold | 71.31% | **77.82%** | **+6.51** |
| real-gold | 87.50% | 88.38% | +0.88 |
| CPU p50 latency | 6.1 ms | 21.9 ms | **3.6x** |

**§50's "bigger is worse on sensitivity" does not survive on this encoder.** It
was measured on ModernBERT; bge-base is a differently-pretrained BERT-base and
gains on all three evals. §57's skew explanation is not needed to explain it,
because this arm used the RAW corpus.

**Caveat I introduced myself:** x1 also changed the schedule (3 epochs at 3e-5,
picked because a larger model overcooks more easily), and no MiniLM control was
run at that schedule — Round X's x3 varies encoder AND corpus together, so it
cannot separate them either. The +3.11 currently confounds encoder with schedule.
Round AC's `se-ac1-mini-raw-3ep` is that missing control.

**And accuracy does not carry over to the gates uniformly.** Over-block at
matched containment (`harness/roc_match.py`, per §60's rule):

| gate | containment | bge over-block | MiniLM over-block | winner |
|---|---:|---:|---:|---|
| NEVER_EGRESS | 85% | **1.19%** | 2.90% | bge (2.4x better) |
| NEVER_EGRESS | 90% | **2.73%** | 4.10% | bge |
| NEVER_EGRESS | 95% | **10.92%** | 11.26% | bge |
| CONFIDENTIAL | 85% | **8.16%** | 9.98% | bge |
| CONFIDENTIAL | 90% | 13.61% | **13.15%** | MiniLM |
| CONFIDENTIAL | 95% | 29.93% | **21.09%** | MiniLM |
| REGULATED | 85% | 15.67% | **10.52%** | MiniLM |
| REGULATED | 90% | 22.75% | **21.89%** | MiniLM |
| REGULATED | 95% | 61.59% | **46.78%** | MiniLM |

**Neither model dominates.** bge owns the NEVER_EGRESS gate outright — the
security-critical one, where it blocks live-secret traffic at the same rate for
less than half the collateral. MiniLM owns REGULATED at every containment level.
So +3.11 tier-exact accuracy is real and does not mean "better router": which
model is better depends on which gate is deployed, and that is a configuration
question, not a leaderboard one.

Their gate strengths being complementary is itself a lead — tested next as an
ensemble.

## 68. The ensemble is never best, and the cheapest model wins — recipe decision

§67 left bge and MiniLM owning different gates, which is the textbook
precondition for averaging to pay. Error overlap confirms the members are
genuinely different: Jaccard 0.56-0.59 between bge and either MiniLM, versus
0.705 between the two MiniLMs.

Raw numbers flatter the ensemble — best containment at every gate. At MATCHED
containment (§60's rule) it flattens out. Over-block, lower is better:

| gate | target | bge | **MiniLM esc0.0** | MiniLM esc1.0 | MEAN ens. |
|---|---:|---:|---:|---:|---:|
| CONFIDENTIAL | 85% | 8.16% | 7.03% | 9.98% | **6.12%** |
| CONFIDENTIAL | 90% | 13.61% | 14.29% | **13.15%** | **13.15%** |
| CONFIDENTIAL | 95% | 29.93% | 24.26% | **21.09%** | 22.00% |
| REGULATED | 85% | 15.67% | **9.44%** | 10.52% | **9.44%** |
| REGULATED | 90% | 22.75% | **18.88%** | 21.89% | 20.82% |
| REGULATED | 95% | 61.59% | **35.41%** | 46.78% | 46.14% |
| NEVER_EGRESS | 85% | **1.19%** | 1.54% | 2.90% | 1.54% |
| NEVER_EGRESS | 90% | **2.73%** | 3.41% | 4.10% | 2.90% |
| NEVER_EGRESS | 95% | 10.92% | **9.22%** | 11.26% | 10.24% |

**The ensemble wins two of nine cells and is never clearly best**, for 3 forward
passes (~34 ms). Not adopted. Its raw-number advantage was entirely an operating
point, exactly as §60 predicted.

**`se-w-esc0.0` — MiniLM with escalation removed — is best or joint-best in 4 of
9 cells and never worst, at 6.1 ms.** bge's wins are confined to the
NEVER_EGRESS gate at 85-90% containment and cost 3.6x latency; at 95% containment
on that same gate, MiniLM esc0.0 beats it (9.22% vs 10.92%).

**Recipe decision for sensitivity: drop `--escalate`, keep MiniLM-L6.** The
cheapest available change — deleting one flag — is also the best one, and the
encoder swap that looked like the headline (+3.11 tier-exact, §67) does not
survive contact with the metric the router actually runs on.

Worth stating because it recurs: three separate interventions this session
(logit adjustment, span-max, this ensemble) produced impressive raw containment
tables that dissolved under the matched-containment control. The control is
doing more work than any of the techniques it tested.

## 69. Headroom for all three signals — and the arithmetic on "high 90s"

The §55 diagnostic run on every signal, against its best model:

| signal | unanimous rows | contested rows | contested share | ceiling from perfect labels | unanimous-row error |
|---|---:|---:|---:|---:|---:|
| complexity | **89.47%** | 63.64% | 29.6% | 92.59% | 7.41% |
| cost | **88.06%** | 58.33% | 28.2% | 91.43% | 8.57% |
| sensitivity | **77.93%** | 51.24% | 25.5% | 83.56% | 16.44% |

Two conclusions, and they point in opposite directions.

**(1) High 90s is not reachable on these evals as constructed.** Even resolving
every contested row perfectly — an upper bound on the entire label-quality axis,
and one that also assumes the model then gets all of them right — caps complexity
at 92.6%, cost at 91.4%, sensitivity at 83.6%. Dropping contested rows from the
eval entirely does not help: that is just the unanimous column, 78-89%.

**(2) But 10-22 points of the remaining error is genuine model error on rows
three independent jurors agreed on.** That is real headroom, it is not label
noise, and it is where effort belongs. The framing "we are stuck at the label
ceiling" is wrong for all three signals.

The unanimous-row confusions are **adjacent-tier almost without exception**:

| signal | top unanimous confusions |
|---|---|
| cost | LOW->MODERATE 13, MODERATE->HIGH 11, MINIMAL->LOW 9, HIGH->MODERATE 7 |
| complexity | MEDIUM->COMPLEX 12, MEDIUM->SIMPLE 8, SIMPLE->MEDIUM 6 |
| sensitivity | INTERNAL->CONFIDENTIAL 40, INTERNAL->REGULATED 22 |

For cost this is pointed, because cost's boundaries are the most MECHANICAL of
the three. Its rubric decides LOW vs MODERATE on brevity cues ("brief", "short",
"concise") and MODERATE vs HIGH on quantity words ("these thirty", "entire",
"every", "all of our"). Those are lexical triggers, not judgement calls — and the
model is missing 24 of them on rows where every juror agreed. Tested next.

## 70. The cost rubric describes cues that are nearly ABSENT from real traffic

§69 pointed at cost because two of its three boundaries are supposedly
MECHANICAL: LOW vs MODERATE decided by brevity cues ("brief", "short",
"concise"), MODERATE vs HIGH by quantity words ("these thirty", "entire",
"every"). Those are lexical triggers, so a regex should reproduce the rubric's
own decision procedure. Measured on the 427 unanimous real rows:

| cue | MINIMAL | LOW | MODERATE | HIGH |
|---|---:|---:|---:|---:|
| brevity ("brief", "short"…) | 0.0% | **8.7%** | 6.4% | 4.3% |
| bulk ("entire", "these N"…) | 0.0% | 0.9% | 2.5% | **4.3%** |
| single-artifact ("comprehensive"…) | 2.3% | 4.3% | **22.7%** | 26.1% |

**The brevity cue fires on 8.7% of LOW rows and 6.4% of MODERATE rows — it does
not discriminate at all. The bulk cue fires on 4.3% of HIGH rows.** A rule-only
classifier built from the rubric's stated decision order fires on 19.7% of rows
and is 65.5% accurate when it does; it says HIGH seven times and is right once.

**The rubric's stated decision procedure is not the procedure anyone can
actually apply to this traffic.** Jurors reading it must be inferring the tier
from something the rubric does not name — which is a plausible cause of cost's
73.4% agreement, the lowest of the three signals.

What DOES separate the tiers, from an interpretable bag-of-words fit on the same
unanimous rows (5-fold CV 67.5%):

| tier | strongest features | median words |
|---|---|---:|
| MINIMAL | *is, what is, what, does* — question forms | 12 |
| LOW | *suggest, describe, tell, how to* | 20 |
| MODERATE | *write, script, essay*, narrative pronouns | 82 |
| HIGH | *write me, 100, 000* — magnitudes | **67** |

Note HIGH prompts are SHORTER than MODERATE ones (67 vs 82 median words), which
is why prompt length alone scores 43.1% — below the majority-class baseline.
Inspection of the actual HIGH rows explains it: in real traffic HIGH is almost
never a large supplied corpus, it is a **huge requested output** — "compose a 50
pages master thesis", "use 15,000 words", "Write me a C compiler". The v1 rubric
leads with the supplied-corpus case, which barely occurs.

**v2 cost rubric** (`rubrics/cost.md`, v1 preserved) restructures around what is
present: an explicit size spec decided first via a numeric table, then artifact
type; HIGH re-centred on huge requested output; the false claim that LOW
"almost always carries an explicit brevity cue" removed.

A/B running on BOTH a random 500-prompt sample and 400 contested prompts —
§65's lesson, so the population effect is measured directly instead of
extrapolated from the hard subset.

**Caveat carried forward:** the fine-tuned model scores 88.06% on these same
unanimous rows against bag-of-words' 67.5%, so the model is NOT merely missing
keywords. This finding is about rubric validity and juror agreement, not a
diagnosis of the model's residual error.

## 71. Cost rubric v2: +22.2 on contested rows, +0.8 on the population — the paired design pays

§70 rewrote the cost rubric around what is actually present in the traffic: an
explicit size spec decided first from a numeric table, then artifact type, with
HIGH re-centred on huge REQUESTED OUTPUT rather than huge supplied input.

§65's lesson applied — both samples measured, not one extrapolated:

| sample | v1 | v2 | delta |
|---|---:|---:|---:|
| **random 500 prompts (population)** | 75.0% | 75.8% | **+0.8** |
| contested 400 prompts (hard subset) | 37.2% | **59.5%** | **+22.2** |

**+22.2 points on the rows that were previously coin-flips is the largest rubric
effect measured in this project — and it moves the corpus by 0.8.** Had only the
contested sample been run, this would have been reported as a breakthrough, for
the second time, in exactly the way §65 retracted. The paired design cost one
extra labelling run and prevented a repeat.

The residual tables say precisely what v2 fixed and what it broke:

| boundary | v1 random | v2 random | v1 contested | v2 contested |
|---|---:|---:|---:|---:|
| LOW/MINIMAL | 46 | **31** | 86 | **41** |
| MODERATE/LOW | 49 | **62** | 115 | 94 |
| HIGH/MODERATE | 5 | 13 | — | 7 |

**v2's size-spec table fixed MINIMAL/LOW** (−33% on the population, −52% on
contested) — that was the change naming multiple-choice, recall and image prompts
as MINIMAL. **v2's artifact-type ladder made LOW/MODERATE worse on the
population** (+27%), and HIGH/MODERATE worse in both samples: re-centring HIGH on
requested output pulled some MODERATE rows up with it.

So the net +0.8 is two real effects cancelling, not a null. The next iteration is
narrow and specified by this table: **keep the size-spec table and the MINIMAL
clarification, revert the artifact-type ladder toward v1's wording, and tighten
the HIGH threshold so long articles stay MODERATE.**

Generalisation worth keeping: a rubric rewrite is not one intervention. Report
per-boundary residuals, because a headline delta can be two changes of opposite
sign, and only the per-boundary view tells you which half to keep.

## 72. Cost rubric v3 is WORSE than v2 — and rubric rewrites move the subset, never the population

§71 diagnosed the v2 cost rubric as two effects cancelling and specified v3
narrowly: keep the size-spec table and the MINIMAL clarification, revert the
artifact-type ladder toward v1's wording, tighten HIGH. Same seeded rows as §71,
so all three versions are directly comparable.

| rubric | random 500 (population) | contested 400 |
|---|---:|---:|
| v1 | 75.0% | 37.2% |
| **v2** | **75.8%** | **59.5%** |
| v3 | 73.4% | 57.2% |

**v3 lost on both samples. The prescribed fix made things worse.** Per-boundary,
MODERATE/LOW disagreement on the population went 49 (v1) -> 62 (v2) -> **73
(v3)** — the boundary v3 existed to repair got monotonically worse as v1's
MODERATE cue words were restored. §70 already measured why and I did not join it
up: the "single-artifact" cue (*detailed, thorough, comprehensive*) fires on
22.7% of MODERATE rows but also 26.1% of HIGH and 4.3% of LOW. It is not a
MODERATE cue at all. Restoring it re-imported the confusion.

`rubrics/cost.md` reverted to v2, the best measured. v1 and v3 preserved.

**The pattern across two signals and three rewrites is now hard to dismiss:**

| signal | rewrite | hard-subset delta | population delta |
|---|---|---:|---:|
| complexity | v1 -> v2 | **+7.0** | +0.2 |
| cost | v1 -> v2 | **+22.2** | +0.8 |
| cost | v1 -> v3 | **+20.0** | -1.6 |

Every rewrite produces a large, real gain on the rows that were previously
coin-flips, and moves the corpus by less than a point in either direction. The
mechanism is arithmetic rather than mysterious — contested rows are ~12-28% of a
corpus, so even resolving a third of them cannot move the whole by much — but the
consequence is worth stating as a rule: **rubric rewriting is not a corpus-level
lever.** It is worth doing to make a boundary decidable and to shrink the
contested pool, and it is not worth doing in the hope of a headline number.

## 73. Corpus-wide agreement PREDICTED the training outcome — the cheap proxy works

§65 recorded the prospective prediction before Round AB ran: corpus agreement
moved 87.5% -> 87.7% under the v2 complexity rubric, so the retrain should land
"at or below complexity's 1.06-point floor". First seed:

| | incumbent (median of 3 seeds) | v2-rubric relabel, seed 11 |
|---|---:|---:|
| refined gold | 0.9016 | 0.9016 |
| real gold | 0.8923 | 0.8971 |

Dead level on refined gold, +0.48 on real gold — both inside the floor. Two seeds
still running, but the prediction is holding.

Read together with §54's +7.0 on the hard subset, this validates a cheap
protocol: **relabel two samples and check agreement on a RANDOM one before
spending a relabelling campaign.** The subset number predicted a gain that did not
materialise; the population number predicted no gain, and there was none. A
59,582-prompt relabel and a 3-seed retrain — roughly a day of wall clock — were
settled in advance by 500 labelled rows.

## 74. Model accuracy tracks JURY AGREEMENT within 2-3 points at every fold

Before proposing any taxonomy change, measure what each candidate fold buys.
`harness/merge_analysis.py` reports, for each, the **jury agreement** it creates
(the ceiling) beside the **accuracy the existing model scores under it with no
retraining** (a lower bound, since a model trained on the fold should beat it).

| taxonomy | jury agreement | model accuracy | gap |
|---|---:|---:|---:|
| complexity, 4 tiers | 70.4% | 81.82% | — |
| complexity merge MEDIUM+COMPLEX | 83.0% | 89.23% | — |
| cost, 4 tiers | 71.8% | 79.66% | — |
| cost merge LOW+MODERATE | 83.4% | 86.89% | — |
| sensitivity, 5 tiers | 74.5% | 72.92% | -1.6 |
| sensitivity merge INTERNAL+CONFIDENTIAL | 80.6% | 79.35% | -1.3 |
| sensitivity binary @ CONFIDENTIAL | 84.8% | 82.30% | -2.5 |
| sensitivity binary @ REGULATED | 87.5% | 84.93% | -2.6 |
| **sensitivity binary @ NEVER_EGRESS** | **96.4%** | **94.42%** | **-2.0** |

On sensitivity, where both are measured on identical rows, **model accuracy sits
within 1.3-2.6 points of jury agreement at every one of five folds.** The model
tracks the decidability of the question almost exactly.

(Complexity and cost sit ABOVE their agreement figures because those are
three-juror unanimity over four tiers while the model is scored against the
adjudicated majority — a laxer target. The *ordering* is what transfers.)

**This reframes the whole project.** Eighteen mechanisms, three encoders, six
corpora, two rubric rewrites, and the model has been pinned to how decidable the
question is the entire time. The remaining lever is not the model. It is which
question gets asked.

## 75. The router's ACTUAL decisions score 92-96% — tier-exact was the wrong metric

llm-d-sc does not consume four complexity tiers. The deployed Praxis table maps
SIMPLE/MEDIUM to the small model and COMPLEX/REASONING to the large one. A
classifier that says MEDIUM where the jury said SIMPLE routes to the same backend
and has made **no routing error**; tier-exact accuracy counts it as one.

Scored on the decisions that exist in the deployment, folding existing models'
outputs with no retraining (`harness/route_analysis.py`):

| signal | decision | jury agr | **accuracy** | majority baseline |
|---|---|---:|---:|---:|
| complexity | *tier-exact (as reported)* | 70.4% | 81.82% | — |
| complexity | route: small vs large | 82.0% | 88.22% | 75.42% |
| complexity | **is reasoning needed** | 94.6% | **95.79%** | 86.53% |
| cost | *tier-exact (as reported)* | 71.8% | 79.66% | — |
| cost | **short vs long generation** | 88.1% | **92.44%** | 50.42% |
| cost | reserve a big budget | 93.8% | 93.78% | **93.45%** |
| sensitivity | *tier-exact (as reported)* | 74.5% | 72.92% | — |
| sensitivity | **block at NEVER_EGRESS** | 96.4% | **94.42%** | 85.04% |
| sensitivity | block at REGULATED | 87.5% | 84.93% | 67.97% |

**The majority-baseline column is not decoration.** Cost's "reserve a big budget"
scores 93.78% — a high-90s-adjacent number for a decision that is 93.45%
achievable by always answering no. It is worthless, and without that column it
would be the most impressive row in the table. The three bolded decisions clear
their baselines by 9.3, 42.0 and 9.4 points.

So: **three of the router's real decisions already score 92-96% with no
retraining**, and every one of them is a fold of a model trained for a different
task. Rounds AD and AE train them directly. The one decision that does NOT reach
the 90s is complexity's small-vs-large route at 88.22% — the honest gap, and the
one worth reporting as unfinished.

**What this does and does not claim.** It does not claim the tier taxonomies are
fine; §69's ceilings stand and tier-exact accuracy on them will not reach the high
90s. It claims the taxonomy is finer than the decision, so tier-exact accuracy
has been charging the classifier for distinctions the router discards.

## 76. My own §63 hypothesis is FALSIFIED — agreement was confounded with separability

§63 explained the minimal-pair corpus's failure by its 98.4% blind-check
agreement: "a synthetic corpus whose agreement sits far above the real
distribution's is off-distribution by construction", and prescribed treating
agreement as a target band rather than a maximand.

`harness/build_train_greyzone.py` was built to test exactly that, and it hit the
target: unconditioned enterprise scenes steered at situations that can land
either side of INTERNAL/CONFIDENTIAL, tier never named, **blind agreement 81.1%,
squarely inside the 75-85% band**, 2,075 rows, 1,424 CONFIDENTIAL / 507 INTERNAL.

Same frozen-probe A/B that rejected the minimal pairs:

| arm | n | MiniLM bal | e5 bal | CONF recall (MiniLM) |
|---|---:|---:|---:|---:|
| A base corpus only, size-matched | 1,014 | **63.66%** | **64.03%** | 47.27% |
| B minimal-pair SDG (§63) | 567 | 51.19% | 58.82% | 49.09% |
| **C grey-zone SDG (on-band)** | 1,014 | 56.07% | 60.04% | **21.82%** |
| D base 3000/class | 6,000 | **69.51%** | **70.69%** | 60.00% |
| E base 3000/class + all grey-zone | 7,931 | 64.31% | 69.20% | 43.64% |

**It fails too — 7.6 points worse than real rows at matched size, and adding it
to the base corpus costs 5.2 points.** Landing in the agreement band bought
nothing. The hypothesis predicted a specific outcome and got the opposite one.

The competing explanation was already in §64 and I did not test it against mine.
Separability from the eval, same probe:

| corpus | blind agreement | separability | does it help? |
|---|---:|---:|---|
| enterprise (31,168 rows) | — | **78.1%** | yes — historically the largest data gain on this signal |
| grey-zone (2,075) | **81.1%** on band | 94.8% | no, -5.2 |
| minimal pairs (567) | 98.4% off band | 98.2% | no, -8.0 |
| v2 synthetic (6,783) | — | 93.2% | marginal |

**Agreement and separability were confounded in §63, and the grey-zone corpus
decouples them: right agreement, wrong distribution, still useless.**
Separability is the operative variable. §63's rule is withdrawn and replaced:

> A synthetic corpus helps in proportion to how INDISTINGUISHABLE it is from the
> evaluation distribution. Measure separability before training on it; agreement
> is not a substitute and, on this evidence, not predictive at all.

**What actually differs between the generator that works and the one that does
not** is not the prompt template — both use unconditioned scenes and never name
the tier. It is pool breadth: `sensitivity-enterprise.jsonl` accumulated 31,168
rows across many runs over 39 roles x 10 settings x 16 moments, while the
grey-zone run drew 2,075 rows from 22 x 14 x 18 in a single pass with an added
"vary the stakes" instruction. A narrower, more homogeneous pool is more
separable. That is the natural next hypothesis and it is **untested** — recorded
as a hypothesis, not a finding, which is the mistake §63 made.

Corpus retained at `data/train/sensitivity-greyzone.jsonl`, NOT added to any
training arm.

## 77. Pool breadth is a small real effect (3.4 pts) — and the separability rule needed a circularity check

**The controlled test §76 called for.** Same generator, same prompt template, same
160 scenes and ~1,280 rows, single pass each. Only the pool differs:

| arm | pool | distinct combos in 160 scenes | separability vs entsec |
|---|---|---:|---:|
| NARROW | 6x3x4 = 72 | 66 | 92.8% |
| WIDE | 1200x160x60 = 11.5M | 160 | **89.4%** |
| *grey-zone (§76), for scale* | 22x14x18 | — | 95.3% |
| *enterprise, capped to same n* | — | — | **79.3%** |

Breadth is real and **small: 3.4 points** from a 160,000-fold increase in
available combinations. It explains a fifth of the ~13-point gap to the corpus
that works. The pool-breadth hypothesis is confirmed in sign and rejected as the
explanation. Note the enterprise figure is capped to the same n=949, so corpus
size is not the confound either.

**Which forced a check I should have run before trusting the metric at all.**
`entsec` is itself SYNTHETIC — an unconditioned enterprise generator very like
the one that produced `sensitivity-enterprise.jsonl`. If those share a lineage,
"separability from the eval" could be measuring shared provenance rather than
realism. The non-circular reference is WildChat: real assistant traffic nobody
here generated.

| corpus | vs entsec (synthetic) | vs REAL traffic |
|---|---:|---:|
| sensitivity-enterprise.jsonl | **74.8%** | 96.8% |
| breadth-wide.jsonl | 86.9% | 97.9% |
| sensitivity-greyzone.jsonl | 93.9% | 99.0% |
| sensitivity-boundary.jsonl | 97.8% | 99.0% |
| sensitivity-real.jsonl (WildChat) | 97.0% | **66.2%** |
| **entsec eval itself** | — | **95.8%** |

**The two evaluation sets are 95.8% distinguishable from each other.** Every
corpus that scores well against entsec scores badly against real traffic, and the
one WildChat-derived corpus inverts perfectly (97.0% / 66.2%).

The rule survives, sharpened and no longer circular: **separability predicts
helpfulness ON THE DISTRIBUTION IT IS MEASURED AGAINST.** It is a domain-match
metric, correctly behaved. §76's ranking was right about entsec and says nothing
about production.

**But the caveat this exposes is larger than the finding.** Sensitivity has been
optimised end to end against a synthetic eval that is 95.8% distinguishable from
real assistant traffic. That was a deliberate trade — WildChat is ~93% PUBLIC and
literally cannot measure the tiers that gate egress, which is why entsec was built
(§10) — and it is recorded in the model cards. It is still worth stating plainly
in one place:

> **The sensitivity numbers in this report describe enterprise-*like* synthetic
> traffic. There is no real enterprise corpus here to validate them against, and
> the one real corpus available is a different distribution. Complexity and cost
> are measured on real traffic; sensitivity is not.**

That is the largest open risk in this project, and no amount of further
optimisation against entsec reduces it. Reducing it needs real enterprise
prompts, which is an access problem rather than a modelling one.

## 78. Uniform class balancing LOSES 3.1 points — §57's prediction was wrong

§57 argued §50's "bigger is worse on sensitivity" was an artefact of class skew,
and predicted a big encoder on a balanced corpus would reverse it. Round X tested
that with the encoder held fixed:

| arm | corpus | entsec-gold | real-gold |
|---|---|---:|---:|
| se-x1-bge-raw | natural | **0.8119** | 0.8838 |
| se-x2-bge-bal | uniform balance | 0.7808 | **0.8944** |

**Uniform balancing costs 3.11 points on entsec** — 22x the noise floor — while
gaining 1.06 on real traffic. §57's prediction is wrong in sign on the eval it
was made about.

§56 already contained the explanation. Uniform balancing cut INTERNAL from 16,450
to 7,694 rows, and INTERNAL is **51.2% of the entsec eval**; the resample moved
the training prior away from the evaluation prior, and §56 measured that mismatch
as worth about a point through logit adjustment alone. Resampling applies the
same distortion harder.

**Uniform balance is not the neutral choice it looks like.** It is a specific
prior — the one that maximises entropy — chosen by default rather than measured.
Round AG varies the prior directly (natural / sqrt-interpolated / exact eval
match) with encoder and schedule fixed.

The two candidates are in genuine tension and both are run rather than argued:
the exact eval prior matches best but starves CONFIDENTIAL to 1,861 rows, and
CONFIDENTIAL is the class §57 showed the model already cannot learn. Whichever
wins says which pressure dominates.

**Scope caveat.** "Match the eval prior" tunes to entsec, which §77 measured as
95.8% distinguishable from real traffic. real-gold is reported on every arm so
the cost of that tuning is visible. Notably x2 already shows the trade running
the other way — uniform balancing HELPED real-gold by 1.06 while hurting entsec.

## 79. The egress gate reaches ~95.5% — and the dedicated model only wins where it matters

Round AD trained the binary egress decision directly. Headline on entsec-gold
(n=707, BLOCK=121, **majority baseline 82.89%**):

| model | argmax accuracy | containment | over-block |
|---|---:|---:|---:|
| binary egress (`eg-ad1-cw-seed11`) | 95.33% | 85.12% | 2.56% |
| **5-way esc0.0, folded** | **95.62%** | 89.26% | 3.07% |
| 5-way v2 (shipped), folded | 95.05% | **90.08%** | 3.92% |

**All three sit at ~95.5%, clearing the majority baseline by 12+ points.** Training
the decision directly bought nothing at argmax — the existing sensitivity model,
folded, is fractionally the best of the three.

**Accuracy is the wrong headline here and the binary model shows why.** It has the
HIGHEST accuracy of the trio at argmax while containing the LEAST (85.12% vs
90.08%): it is more willing to let sensitive content through, which on a 17%-BLOCK
eval buys accuracy. Over-block at matched containment, per §60:

| model | @85% | @90% | @95% | @99% |
|---|---:|---:|---:|---:|
| binary egress | 1.88% | 3.58% | **8.87%** | **48.98%** |
| 5-way folded | **1.54%** | **3.07%** | 10.41% | 79.01% |
| 5-way v2 shipped | 2.73% | 3.92% | 10.75% | 85.32% |

**The curves cross.** Below 90% containment the folded 5-way model is cheaper;
at 95% and above the dedicated binary model is, and at 99% containment it
over-blocks **49% against 79%** — a 30-point difference in how much legitimate
traffic gets stopped to catch the last few percent of secrets.

A security gate operates at high containment. **So the binary model is the right
one to deploy, and the reason is the opposite of its headline number.**

**A trap worth recording, because the numbers were sitting right there.** The
binary model scores **99.65% on real-gold and 99.46% on enterprise-gold**. Both
are worthless: those evals contain 2 and 5 BLOCK rows respectively, majority
baselines are 99.30% and 99.33%, and BLOCK recall is 1/2 and 1/5. Reported without
the baseline column, "99.65% on real traffic" is the most impressive figure
produced anywhere in this project and it means the model almost never fires.
entsec is the only eval with enough BLOCK mass (121 rows) to say anything.

**Against the standing goal, stated precisely.** This is ~95.5% — mid 90s, not
high 90s — on one real deployed decision, measured against a synthetic eval that
§77 found 95.8% distinguishable from real traffic. It is the best-supported
number in this project and it is not "high 90s across all domains".

## 80. Round X complete: the encoder effect is real (+4.5) and §50 is overturned

The full factorial, entsec-gold:

| encoder | corpus | schedule | entsec | real-gold |
|---|---|---|---:|---:|
| bge-base | natural | 3ep/3e-5 | **0.8119** | 0.8838 |
| bge-base | uniform balance | 3ep/3e-5 | 0.7808 | **0.8944** |
| MiniLM-L6 | uniform balance | 3ep/3e-5 | 0.7355 | 0.8732 |
| MiniLM-L6 | natural | 4ep/5e-5 | 0.7808 (median) | 0.8750 (median) |

**§67's confound is resolved.** The bge-vs-MiniLM comparison on the balanced
corpus holds encoder schedule AND corpus exactly constant: **+4.53 points for
bge**, matching the +3.11 seen on the natural corpus. The gain is the encoder,
not the 3ep/3e-5 schedule I introduced alongside it.

**§50's "bigger is worse on sensitivity" is overturned for this encoder.** It was
measured on ModernBERT; bge-base gains 3-4.5 points wherever it is compared
fairly. Round AC's remaining cell (MiniLM/natural/3ep) would close the 2x2
exactly but is no longer load-bearing.

**Balancing hurts both encoders on entsec and helps both on real-gold** — bge
-3.11/+1.06, MiniLM -4.53/-0.18 — which is the signature of a prior effect rather
than a capacity effect, and is what Round AG is now measuring directly.

**This does not change the deployment recommendation.** §68's matched-containment
comparison still has MiniLM+escalate-0.0 winning 2 of 3 gates at a fifth of bge's
latency (6.1 ms vs 21.9 ms). Tier-exact accuracy and gate behaviour disagree here,
and the gate is what the router runs on. bge is the better *classifier*; MiniLM is
the better *component*.

## 81. A harness defect hid three failed runs — round scripts discarded their own tracebacks

Round Z (hierarchy-aware SupCon) printed four arm headers and no results. The
`|| echo "  FAILED"` guard never fired, and `models/se-z-supcon*/` contained only
an empty `_ck` directory.

The pattern every `round_*.sh` used:

```
python train.py ... 2>&1 | grep -E "^\[|entsec-gold|trained in" || echo "  FAILED"
```

It loses failures twice. stderr is merged into the pipe and then thrown away by a
grep that does not match tracebacks — and the `|| echo FAILED` cannot fire,
because grep DOES match the `[transformers] LOAD REPORT` banner on `^\[`, so the
pipeline always exits 0. **Three arms died silently and the log looked like they
had simply produced nothing.**

Fixed with `harness/runarm.sh`, which writes full output to `/tmp/arm-<tag>.log`,
prints the summary lines, and on non-zero exit prints the tail of the real log.

**What actually killed the arms is NOT established, and I am not going to guess.**
The evidence rules out the obvious answer: no jetsam or `memorystatus` entries in
the system log, ~31 GB free, total RSS across all trainers ~10.6 GB of 128 GB.
Re-running arm 1 in isolation works — it is 5% through a normal 75-minute run.
And Round Z's FOURTH arm never died at all; it was still training when the round
script was killed. Arms 1-3 failing while arm 4 survived is not consistent with a
deterministic bug in the SupCon loss, and not obviously consistent with resource
exhaustion either. Recorded as unexplained.

**The operational lesson is separate and is confirmed.** Each `queue_*.sh` waiter
watches the round before it, so killing one round cascaded the next two into life
simultaneously and produced eight concurrent trainers on a 16-core machine. A
chain of waiters is a chain of triggers, and interrupting it anywhere fires
everything downstream. Waiters are now disarmed and rounds are launched
deliberately.

## 83. The ceiling is a property of the PANEL, not the task — self-agreement beats inter-agreement by 27.7 points

Every ceiling in this report rests on inter-juror agreement, and all three jurors
are Claude models. The question never asked: **when the same juror labels the
same prompt twice, does it agree with itself?**

400 complexity prompts, stratified 200 unanimous / 176 contested, one model
(`claude-sonnet-5`) relabelling three times with distinct cache keys:

| measurement | value |
|---|---:|
| inter-juror agreement over the sample | 53.2% |
| **self-agreement across 3 passes** | **80.9%** |
| **delta** | **+27.7** |
| self-agreement on unanimous rows (n=200) | 90.0% |
| self-agreement on contested rows (n=176) | 70.5% |

**Each model is far more consistent with itself than the models are with each
other.** That is the "self >> inter" branch, and it changes the reading of §69:
the disagreement driving every ceiling in this report is dominated by
**systematic between-model difference**, not by irreducible ambiguity in the
prompts.

Two consequences, both actionable:

1. **Intra-juror sampling noise is real and is currently baked into the gold.**
   Each juror labelled once; rows where all three happened to agree became gold.
   Re-sampling shows **10% of "unanimous" rows are unanimous by luck of the draw**
   (90.0% self-agreement on rows the panel agreed on 100% of the time). The fix is
   cheap and does not need a new panel: **majority-of-3-samples per juror before
   cross-juror adjudication**, which costs 3x labelling on one model.

2. **Modelling the raters is the right response, not averaging them.** If the
   models are individually coherent but mutually different, soft targets over
   their votes throw away the structure. That is exactly the case per-annotator
   heads are built for, and it is now motivated by measurement rather than by the
   literature alone.

**The honest alternative reading**, which this experiment cannot separate: each
model may hold a stable but genuinely different interpretation, in which case the
ambiguity is real and lives *between* raters rather than within them. That is the
perspectivist position, and it points at the same response — model the raters —
while meaning something different about the task. Distinguishing them needs a
non-Claude juror, which this project does not have.

**What it does settle:** "high 90s is unreachable" was justified by inter-juror
agreement, and inter-juror agreement is not a measurement of task difficulty. It
is a measurement of this particular three-model panel. §69's ceilings stand as
descriptions of the current evaluation and should NOT be read as facts about the
problem.

## 84. Egress gate, second seed

`eg-ad1-cw-seed22`: entsec-gold 0.9434 against seed 11's 0.9533 — median **0.9484**,
majority baseline 82.89%. The gate result is stable across seeds; publication
waits on the containment comparison in §79 being re-run on the median model
rather than the better one.

## 85. Two independent labelling processes agree on 81% of rows — and the model is 6.4 points better there

§83 said a single juror re-sampled is more coherent than the panel. The obvious
follow-up: build a gold that way and see what changes. One juror
(`claude-sonnet-5`), three samples, majority vote, over all 594 complexity eval
rows (`harness/selfconsistency_gold.py`). A majority formed for 594/594; only 2
rows were three-way splits.

| | |
|---|---:|
| self-consistency gold agrees with PANEL gold | **81.1%** (482/594) |
| top disagreements | MEDIUM->COMPLEX 36, MEDIUM->SIMPLE 17, COMPLEX->MEDIUM 16 |

Model `cx-v-resolved-seed11` scored against each:

| target | n | accuracy |
|---|---:|---:|
| PANEL gold (3 models, 1 sample each) | 594 | 0.8182 |
| SELF-CONSISTENCY gold (1 model, 3 samples) | 594 | 0.7912 |
| **rows where BOTH golds agree** | **482** | **0.8817** |

**Switching to the more coherent rater does NOT raise measured accuracy — it
lowers it by 2.7 points.** That is unsurprising in hindsight: the model was
trained on panel-derived labels, so it reproduces the panel. It is worth stating
because §83 could easily be over-read as "use one juror and the numbers go up".
They go down.

**What agreement BETWEEN the two processes buys is a noise detector.** On the 482
rows where a 3-model panel and a 3-sample self-consistency vote independently
concur, the model scores **0.8817 against 0.8182 overall — +6.35 points**. The
112 rows (18.9%) where the processes disagree carry all of that gap.

This is a better-grounded ceiling estimate than §69's, because it is built from
two labelling procedures that differ in *kind* — across-model versus
within-model — rather than from unanimity inside one panel, which §83 showed is
partly sampling luck.

**And it does not rescue the high-90s target.** With the cleanest labels this
project can construct, complexity's model sits at 88.2%. The gap to the high 90s
is not all label noise.

**Not pursued: retraining on the both-agree subset.** Label cleaning has been
tried on complexity twice already — tie-resolved labels at +0.27 (§53) and the
v2-rubric relabel at +0.13 (§65/AB) — both inside the 1.06-point floor. A third
attempt with a better noise detector is a reasonable idea with a poor prior, and
the compute is better spent on the per-annotator heads, which attack the
structure §83 identified rather than the noise.

## 86. Every per-run latency figure in this report was contaminated by concurrent load

`bge-small` reported p50 **24.76 ms** and `bge-base` reported **21.9 ms** — a 33M
model slower than a 109M model at the same depth, which is not possible. The two
numbers came from different training runs, at different times, with different
numbers of other trainers competing on the same machine.

**That invalidates every per-run latency number quoted so far**, including the
6.1 ms vs 21.9 ms comparison §68's deployment recommendation leans on.

`harness/latency_bench.py` measures all models in ONE process, single-threaded,
in interleaved rounds so any drift in machine load hits every model equally
instead of whichever ran last:

| model | params | p50 | p90 | p99 | vs MiniLM |
|---|---:|---:|---:|---:|---:|
| MiniLM-L6 (`se-w-esc0.0-seed22`) | 22.7M | **8.38 ms** | 16.75 | 19.86 | 1.00x |
| bge-small (`se-ah-bgesmall-seed11`) | 33.4M | 13.94 ms | 28.40 | 33.07 | 1.66x |
| bge-base (`se-x1-bge-raw`) | 109.5M | 31.20 ms | 63.77 | 75.50 | 3.73x |

The ordering now matches parameter count, and the ratios are stable. §68's
qualitative conclusion survives — bge-base really is ~3.7x MiniLM, not the 3.6x
claimed, so the recommendation does not change — but it survived by luck rather
than by measurement.

**Rule: latency comparisons across separately-timed runs are not evidence.** Any
number that has to be compared must be measured in the same process, in the same
pass.

## 87. bge-small matches bge-base's accuracy at 45% of its cost

`se-ah-bgesmall-seed11`, natural corpus, escalate 0.0 (§66's recipe):

| model | recipe | entsec-gold | real-gold | p50 | latency vs MiniLM |
|---|---|---:|---:|---:|---:|
| MiniLM-L6 | esc 0.0 | 0.7999 (median) | 0.8750 | 8.38 ms | 1.00x |
| **bge-small** | esc 0.0 | **0.8119** | 0.8803 | 13.94 ms | **1.66x** |
| bge-base | esc 1.0 | 0.8119 | 0.8838 | 31.20 ms | 3.73x |

**bge-small reaches bge-base's entsec score exactly, at 45% of its latency**, and
beats MiniLM by 1.20 points for a 66% latency increase. That is the cell §80's
split implied and §57's frozen probe predicted: the probe had bge-small
recovering 52% of bge-base's balanced-accuracy gain over MiniLM, and the trained
model recovers all of it.

It also vindicates probing all three small encoders rather than assuming: e5-base
beat bge-base on the frozen pair probe (§61), but e5-small was WORSE than
bge-small (§87 probe table). Family rank at one size does not predict rank at
another.

**bge-small is now the recommended sensitivity encoder** unless §68's
matched-containment comparison reverses on it — that comparison is what decided
against bge-base and has not yet been run here. Pending, and the recommendation
is provisional until it is.

## 88. All three domains now have a deployed decision at 95%+, trained directly

Rounds AD and AE trained the router's real decisions rather than the tier
taxonomies. Every figure below carries its majority baseline and per-class
recall, because §79 showed a 99.65% result on this project's own data that meant
the model almost never fired.

| domain | decision | eval | n | **accuracy** | majority | minority-class recall |
|---|---|---|---:|---:|---:|---:|
| complexity | is reasoning needed | refined gold | 376 | **96.54%** | 85.64% | YES 98.15% |
| cost | short vs long generation | refined gold | 371 | **95.69%** | 53.10% | SHORT 91.95% |
| sensitivity | block at NEVER_EGRESS | entsec gold | 707 | **95.05%** (median of 3) | 82.89% | BLOCK 85.12% |

**The cost result is the strongest of the three and the least flattering-looking.**
Its majority baseline is 53.10%, so no majority-class shortcut exists: 95.69% is
+42.6 points over chance, with LONG recall 98.98% and SHORT recall 91.95%. It
also replicates on a second independent eval — `genlen-volume-gold`, 727 rows,
92.30% against a 59.28% baseline — which none of the other decisions has.

The reasoning decision clears its baseline by 10.9 points with 98.15% recall on a
class that is 3.2% of the training corpus. Egress clears by 12.2 points.

**Where they are weakest, stated:** on rows the jury split, reasoning scores
89.77% against a 90.91% baseline (below it) and genlen 84.52% against 55.95%
(well above). Contested rows are 30% of the complexity eval, so the reasoning
figure is a real limitation rather than a rounding note.

**Relation to the tier numbers.** These do not replace tier-exact accuracy; they
measure something different and narrower. Tier-exact on the shipped taxonomies is
0.8963 / 0.8976 / 0.8034 and §69's ceilings still apply to it. What §75 argued and
this confirms is that the taxonomy is finer than the branch the router takes, so
tier-exact has been charging the classifier for distinctions the deployment
discards.

## 89. Round AB complete: the v2-rubric relabel moved complexity by exactly zero

Three seeds, against the five-run incumbent median of 0.9016:

| | refined gold |
|---|---:|
| cx-ab-v2rubric-seed11 | 0.9016 |
| cx-ab-v2rubric-seed22 | 0.9043 |
| cx-ab-v2rubric-seed33 | 0.9016 |
| **median** | **0.9016** |
| incumbent median | 0.9016 |
| **delta** | **0.0000** |

§65 predicted "at or below the 1.06-point floor" from corpus-wide labeller
agreement moving 87.5% -> 87.7%, recorded before the retrain. The outcome is
zero to four decimal places.

**A 59,582-prompt relabel and a three-seed retrain — roughly a day of wall clock —
were correctly called in advance by 500 labelled rows.** That protocol is the
transferable result: A/B a rubric on a RANDOM sample, and if corpus agreement
does not move, do not spend the relabel.

## 90. The two sensitivity wins stack — 0.8204, and a clean accuracy/latency ladder

§66 (remove tier escalation) and §80 (bge-base instead of MiniLM) act on
different parts of the system, one on the loss and one on the representation, so
Round AI tested whether they compose. They do:

| model | encoder | escalate | entsec-gold | real-gold | p50 |
|---|---|---:|---:|---:|---:|
| shipped v2 | MiniLM-L6 | 1.0 | 0.7793 | 0.8873 | 8.38 ms |
| v2.1 (published) | MiniLM-L6 | 0.0 | 0.8034 | 0.8627 | 8.38 ms |
| bge-small | bge-small | 0.0 | 0.8119 | 0.8803 | 13.94 ms |
| bge-base | bge-base | 1.0 | 0.8119 | 0.8838 | 31.20 ms |
| **bge-base** | **bge-base** | **0.0** | **0.8204** | **0.8908** | **31.20 ms** |

**0.8204 is the best sensitivity number in the project, and it is best on real
traffic too (0.8908) — the first time one model has led both evals.** Total gain
over the shipped v2: **+4.11 points**, from two independent one-line changes.

Encoder size gives a clean monotone ladder at escalate 0.0 — 0.7999 / 0.8119 /
0.8204 for 8.38 / 13.94 / 31.20 ms — so this is a deployment menu rather than a
single answer.

**The gate comparison does NOT follow the accuracy ranking, again.** Over-block at
matched containment (§60's rule), bge+esc0 versus MiniLM+esc0:

| gate | 85% | 90% | 95% |
|---|---|---|---|
| CONFIDENTIAL | MiniLM 7.03% | **bge 12.70%** | MiniLM 24.26% |
| REGULATED | MiniLM 9.44% | MiniLM 18.88% | MiniLM 35.41% |
| NEVER_EGRESS | MiniLM 1.54% | **bge 2.39%** | **bge 8.19%** |

**MiniLM wins 6 of 9 cells despite being 1.70 points worse on tier-exact
accuracy.** But bge wins the two cells that matter most for a security gate — the
NEVER_EGRESS boundary at 90% and 95% containment, where it over-blocks 2.39% and
8.19% against MiniLM's 3.41% and 9.22%.

So the recommendation is conditional and should be stated that way:

- **egress gate run at high containment** -> bge-base + escalate 0.0
- **REGULATED gating, or latency-bound CPU serving** -> MiniLM + escalate 0.0
- **middle ground** -> bge-small, which matches bge-base's accuracy at 45% of its
  cost (§87) and has not yet had its matched-containment comparison run

This is the third time tier-exact accuracy and gate behaviour have disagreed
(§68, §79, here). The pattern is consistent enough to state as a rule: **on an
ordered taxonomy with a threshold gate, accuracy ranks models differently from
the metric the deployment runs on, and the accuracy ranking is the wrong one to
ship against.**

## 91. Per-annotator heads do NOT clear the floor — the gain was my training loop

§83 motivated per-juror heads as the one mechanism with a route past the
agreement ceiling, and the first run looked like it delivered: 0.9149 on refined
gold against the five-run `train.py` incumbent median of 0.9016, **+1.33 against
a 1.06-point floor**.

The control says otherwise. `perjuror.py` in `soft` mode is the identical script,
corpus, loader, optimiser, schedule and pooling, with a single shared head over
the vote distribution instead of one head per juror:

| arm | refined gold | real gold |
|---|---:|---:|
| per-juror heads | 0.9149 | 0.8947 |
| **control: shared soft head, same loop** | **0.9096** | 0.9019 |
| **delta attributable to per-juror heads** | **+0.53** | **-0.72** |
| `train.py` incumbent (median of 5 runs) | 0.9016 | 0.8923 |

**+0.53 on refined gold and -0.72 on real gold: the mechanism is inside the noise
floor and does not have a consistent sign.** Per-annotator modelling is not
confirmed on this signal.

**What the control does show is that most of the apparent gain was the training
loop**, which beats the `train.py` incumbent by +0.80 on refined gold with a
plain shared head. `perjuror.py` differs from `train.py` in several ways at once
— masked-mean pooling instead of BERT's pooler, OneCycleLR instead of linear
warmup, and a larger row count (161,617 vs the round-V corpus) — so that +0.80 is
**confounded across three changes and is not a finding**. It is a lead: if it
survives isolation it is worth more than anything the head structure did.

**This is the correct outcome to record and the one I would have got wrong.**
Without the control, "per-annotator heads clear complexity's floor" was a clean
story with literature support behind it (§83, arXiv 2409.17577), and it would have
been wrong.

## 92. Self-consistency training labels: +0.80, inside the floor

Round AJ, the untested cell from §83/§85. Same 20,000 prompts in both arms; only
the labelling process differs. Two seeds each:

| arm | labels | seed 11 | seed 22 | median |
|---|---|---:|---:|---:|
| aj1 | panel (2 jurors, 1 sample each) | 0.8670 | 0.8511 | 0.8591 |
| **aj2** | **self-consistency (1 juror, 3 samples)** | **0.8697** | **0.8644** | **0.8671** |
| | | +0.27 | +1.33 | **+0.80** |

**Positive in both seeds but +0.80 median against a 1.06-point floor: suggestive,
not established.** Two seeds cannot resolve it, and the honest label is
"unresolved" rather than either "works" or "does not".

What it does close is the pre-registered question. Four label-quality
interventions have now been measured on complexity — tie-resolved labels (+0.27),
the v2-rubric relabel (0.00 over three seeds), §85's clean-subset analysis, and
now denoised training labels (+0.80) — and **not one has cleared the floor.**
Label noise is not the binding constraint on this signal. The label axis is
closed unless someone wants to spend many seeds chasing a sub-floor effect.

Note the absolute numbers here are low (0.86-0.87) because both arms train on
20,000 rows rather than the full corpus; only the aj1-vs-aj2 difference is
meaningful.

## 93. The small-vs-large route closes the gap: 88.22% -> 93.35%

§75 flagged the route decision as the one deployed branch not in the 90s, and the
one that fires on EVERY request. Trained directly rather than folded from the
4-way model (`rt-af1-seed11`):

| eval | n | accuracy | majority | LARGE recall | SMALL recall |
|---|---:|---:|---:|---:|---:|
| refined gold | 376 | **93.35%** | 80.59% | 91.78% | 93.73% |
| real gold | 418 | 93.30% | 76.79% | 91.75% | 93.77% |
| real contested | 176 | 77.84% | 72.16% | 83.67% | 75.59% |

**+5.13 over the folded 88.22%, and +12.8 over the majority baseline**, with the
minority class (LARGE, 19.4% of the eval) at 91.78% recall — this is not a
majority-class artifact. LARGE precision is 77.91%, so it over-routes about 6% of
small requests to the large model: a cost trade, and the direction most
deployments would choose over the reverse.

**Every deployed decision is now above 93%**, trained directly:

| domain | decision | folded | **trained** | majority |
|---|---|---:|---:|---:|
| complexity | is reasoning needed | 95.79% | **96.54%** | 85.64% |
| cost | short vs long generation | 92.44% | **95.69%** | 53.10% |
| sensitivity | block at NEVER_EGRESS | 94.42% | **95.05%** | 82.89% |
| complexity | route: small vs large | 88.22% | **93.35%** | 80.59% |

Training the decision directly beats folding on all four, by +0.75 to +5.13. The
gain is largest exactly where the fold was worst, which is what you would expect
if folding loses information the binary task needs.

## 94. The §91 loop lead is NOT the schedule

Round AK's second arm changed only the learning-rate schedule — cosine instead of
linear — against the round-V corpus:

| arm | refined gold | vs incumbent 0.9016 |
|---|---:|---:|
| cx-ak2-cosine | 0.9069 | +0.53 |

Inside the 1.06-point floor, single seed. **The schedule does not explain the
+0.80 gap** between `perjuror.py`'s control and `train.py`.

That leaves two candidates: corpus coverage and masked-mean pooling. The corpus
arm is running, and it is now a genuine like-for-like — the corrected file glob
totals **exactly 161,617 rows**, matching what `perjuror.py` trained on. Two of
those files (`complexity-pair-MEDIUM-COMPLEX`, `complexity-pair-SIMPLE-MEDIUM`,
~48k rows of synthetic minimal pairs for the two hardest boundaries) plus the
`disagree` corpora have never appeared in any `train.py` complexity recipe.

If corpus coverage is the answer, the finding is unglamorous: **every complexity
run in this project trained on a subset, by accident of which files were listed
in a shell variable.**

## 95. The §91 loop lead dissolves under isolation — no component, and no total, clears the floor

`perjuror.py`'s control beat `train.py` by +0.80 on complexity refined gold, and
§91 recorded that as the largest unexplained gap on the signal. Round AK varied
each difference alone:

| arm | change from the incumbent recipe | refined gold | delta |
|---|---|---:|---:|
| incumbent (`train.py`, median of 5) | — | 0.9016 | — |
| cx-ak2-cosine | cosine schedule only | 0.9069 | +0.53 |
| cx-ak1-allrows | full 161,617-row corpus only | 0.9069 | +0.53 |
| perjuror control | all three changes together | 0.9096 | +0.80 |

**Neither component clears the 1.06-point floor, and neither does their total.**
The two isolated arms land on the same number to four decimals, which is a
coincidence of single seeds rather than a result, and the combined +0.80 was
already inside the floor when §91 called it a lead.

**Masked-mean pooling is left untested and will stay that way.** With the total
effect inside the floor, isolating the third component is chasing a sub-floor
difference across single seeds; the compute buys more elsewhere. Recorded as
closed, not as pending.

Worth noting what the corpus arm rules out: `complexity-pair-MEDIUM-COMPLEX`,
`complexity-pair-SIMPLE-MEDIUM` and the `disagree` corpora — about 110k rows no
`train.py` complexity recipe has ever included — are worth +0.53, i.e. nothing.
**The incumbent recipe was not leaving anything on the floor.**

## 96. Minimal pairs fail on the route decision too (-0.53)

Round AF's second arm added `route-pair-MEDIUM-COMPLEX` to the route corpus. Those
pairs were generated specifically to sharpen the MEDIUM/COMPLEX boundary, which
is exactly the boundary this decision cuts on, so this was the most favourable
test they could get:

| arm | refined gold |
|---|---:|
| rt-af1-seed11 (no pairs) | **0.9335** |
| rt-af2-pairs | 0.9282 |

**-0.53.** That is the third independent failure of synthetic minimal pairs in
this project (§63 on sensitivity's INTERNAL/CONFIDENTIAL, §76 on the grey-zone
variant, and now the boundary they were literally built for). §77's separability
explanation covers all three: the pair corpora sit 97-98% distinguishable from
the eval distribution, and distance from the target distribution predicts
uselessness regardless of how well-aimed the content is.

## 97. The SupCon crashes were an eval-time OOM in my own integration

§81 recorded three Round Z arms dying silently and left the cause unexplained,
noting the evidence ruled out system-level OOM (no jetsam entries, 31 GB free).
The isolated re-run died too — further along, partway through the eval loop, again
leaving only an empty `_ck` — which is what identified it.

The bug is one line:

```python
outputs = model(**inputs, output_hidden_states=(supcon > 0))
```

`compute_loss` runs during EVALUATION as well as training, so this requested 13
layers of hidden states per eval batch. HF Trainer accumulates eval outputs on the
accelerator, and hidden states are batch x seq x hidden x layers — orders of
magnitude larger than logits. The accumulation grows until the process dies. It
is not visible in system memory statistics because it is accelerator memory, which
is why §81's check came back clean and the cause looked mysterious.

It also explains the pattern §81 could not: **arms died at different points**
(some early, the isolated re-run at 74% of an eval loop) because the failure
depends on accumulated eval batches, not on a deterministic code path. And Round
Z's fourth arm survived because it was still TRAINING when the round was killed —
it had not reached evaluation yet.

Fixed with `output_hidden_states=(supcon > 0 and model.training)` — **and that
fix was half of one.** The next arm died 33% into training at the first epoch-end
evaluation with `TypeError: 'NoneType' object is not subscriptable`: hidden states
were correctly no longer requested, but the loss block still indexed
`outputs.hidden_states[-1]`. `Trainer.prediction_step` calls `compute_loss` during
evaluation, so **a training-only loss term needs the guard in both places**, not
just at the point where its inputs are produced. Complete fix:

```python
outputs = model(**inputs, output_hidden_states=(supcon > 0 and model.training))
...
if supcon > 0 and model.training:
    loss = loss + supcon * _hier_supcon(...)
```

Verified by a 1-epoch smoke run that reaches evaluation and completes. The
hierarchy-aware contrastive loss has still never produced a measurement.

**§81's operational lesson stands and its diagnosis was wrong.** "Unexplained" was
the honest label at the time, and keeping the arms' failure visible is what made
the re-run diagnosable — but the cause was mine, in the mechanism under test,
not in the environment.

## 98. Jury agreement predicts model accuracy across every taxonomy in the project

Seven taxonomies, four signals, folds ranging from binary to five-tier, all scored
on the same underlying traffic:

| taxonomy | n | jury agreement | minority share | model accuracy |
|---|---:|---:|---:|---:|
| sensitivity, 5 tiers | 949 | 74.5% | 5.8% | 0.8176 |
| complexity, 4 tiers | 594 | 70.4% | 11.1% | 0.8963 |
| cost, 4 tiers | 595 | 71.8% | 6.6% | 0.8976 |
| **cx2 SIMPLE/REASONING** | 594 | 82.0% | 24.6% | **0.9269** |
| egress ALLOW/BLOCK | 949 | 96.4% | 15.0% | 0.9505 |
| genlen SHORT/LONG | 595 | 88.1% | **49.6%** | 0.9569 |
| reasoning YES/NO | 594 | 94.6% | 13.5% | 0.9654 |

**Pearson r = 0.760**, fit `model = 0.36 x agreement + 61.7`. The single best
predictor of how well a classifier will score is not its architecture, its
encoder, its corpus size or its loss — it is **how often two labellers agree on
the taxonomy it is being graded against.**

*(This paragraph originally read r = 0.90. That number was written before it was
computed and was wrong; corrected here rather than silently. r = 0.760 on n=7 is
suggestive, not strong — the sample is seven points and they are not independent,
since several folds are derived from the same underlying labels.)*

Residuals from the fit: +2.4, +1.3, +0.6, +2.1, +2.1, **-6.9**, -1.6. Six of the
seven sit within 2.5 points of the line. The outlier is **sensitivity 5-tier at
-6.9**: it scores far below what its agreement predicts. That is consistent with
§55's separate diagnosis that sensitivity is the one MODEL-limited signal here —
it misses 156 rows where all three jurors agreed — so it is the only taxonomy
whose ceiling is not the binding constraint. The residual and the diagnosis were
arrived at independently and agree.

This is the summary of the whole project. Twenty mechanisms, five encoders, six
corpora, three rubric rewrites and four label-quality interventions were measured
against these signals; the largest effect any of them produced was +4.53 (encoder,
on sensitivity). Moving from a 4-tier taxonomy to the 2-tier decision the router
actually makes is worth **+3.1 to +6.9** and costs nothing operationally.

**The practical rule:** measure jury agreement on a candidate taxonomy BEFORE
building a classifier for it. It is a few hundred labelled rows and it bounds
everything downstream. A taxonomy whose labellers agree 70% of the time will not
produce a high-90s classifier no matter what is done to the model.

**Read the minority-share column alongside it.** `genlen` has the highest accuracy
per point of agreement in the table because its classes are nearly balanced
(49.6%), so its 0.9569 is earned against a 53.10% baseline. `egress` reaches 96.4%
agreement partly because 85% of its rows are ALLOW. Agreement and balance both
matter, and agreement alone can be inflated by a degenerate split — which is
exactly what the native-rubric A/B ran into (§99).

## 99. A native 2-tier rubric is WORSE than folding a 4-tier one — twice

Collapsing the taxonomy helps (§98). The obvious next step is to write the rubric
natively for two tiers rather than folding labels assigned under four. It loses:

| split | folded 4-tier rubric | native 2-tier rubric | delta |
|---|---:|---:|---:|
| reasoning YES/NO | 98.8% | 92.2% | **-6.5** |
| cx2 SIMPLE/REASONING | 93.8% | 87.8% | **-6.0** |

Same direction, near-identical magnitude, two independent splits, same prompts and
same two labellers in each pair.

**Likely mechanism: the intermediate tiers are scaffolding.** Asking "SIMPLE,
MEDIUM, COMPLEX or REASONING?" decomposes one hard judgement into three easier
ones, and the fold discards the hardest boundary AFTER it has done its work of
anchoring the other two. A 2-tier rubric poses the whole question at once with
nothing to brace against.

**Practical consequence, and it inverts the obvious design:** keep the fine-grained
taxonomy for ANNOTATION, ship the collapsed one. The 4-tier rubric is a better
labelling instrument than the 2-tier rubric even when the 2-tier taxonomy is the
better product.

**Correction recorded here rather than quietly:** the route split was described in
conversation as "80/20 on real traffic". That is the ENRICHED eval. A random
400-prompt sample of real traffic folds 376/24 — about **94/6**. The gold sets
over-sample rare tiers roughly 4x, which does not affect any accuracy figure
(baselines are computed on the eval) but does matter for capacity planning: the
large model is selected on roughly 6% of live traffic, not 20%.

## 100. Two-tier complexity settles at 0.9269 — and it is the model already published

Round AM trained the collapsed taxonomy directly:

| arm | encoder | refined gold |
|---|---|---:|
| cx2-am1-mini-seed11 | MiniLM-L6 | 0.9335 |
| cx2-am1-mini-seed22 | MiniLM-L6 | 0.9202 |
| **median** | | **0.9269** |
| *(4-tier complexity, for comparison)* | | *0.8963* |

**Those numbers are identical to `route-gate`'s, and the ensemble diagnostic
confirms why: pairwise error overlap between `cx2-am1-mini-seed11` and
`rt-af1-seed11` is Jaccard = 1.000.** They are the same model. `cx2` and `route`
are the same fold under different label vocabularies — SIMPLE/REASONING versus
SMALL/LARGE — trained on the same rows with the same seed. Retraining the
collapsed taxonomy reproduced the published model exactly, which is a
reproducibility check passed rather than a new result.

So the answer to "collapse complexity to two tiers": **+3.06 points (0.8963 ->
0.9269), already shipped as
[`llm-d-sc-route-gate`](https://huggingface.co/cnuland/llm-d-sc-route-gate).**

## 101. Ensembling fails on the 2-tier taxonomy, and the diagnostic said so first

Five members — two seeds of MiniLM, two of the identically-recipe'd route model,
one bge-small:

| ensemble | accuracy |
|---|---:|
| best single member | **93.35%** |
| mean of 3 | 92.29% |
| mean of 5 | 92.29% |

**The ensemble is worse than its best member.** Pairwise error overlap explains it
before the accuracy does: Jaccard runs **0.667 to 1.000**, so the members fail on
the same rows and averaging has nothing to correct. bge-small is the most
complementary at 0.667 and it is still far from independent.

This is the opposite of the earlier complexity result that gained +1.9 from
ensembling, and the difference is the taxonomy. On four tiers, members disagreed
about WHICH boundary a row sat on, and those disagreements partly cancelled. On
two tiers there is one boundary, every member is confused by the same rows near
it, and there is no diversity left to exploit. **Collapsing the taxonomy removes
the ensemble's headroom along with the labels' headroom.**

Reporting error overlap before accuracy is what made this cheap: the Jaccard
table alone was enough to predict the null.

## 102. Denoised labels are WORSE on the collapsed taxonomy (-1.06)

Round AN, with the prediction recorded in the script before it ran. Round AJ's two
20k corpora, folded to two tiers, no new labelling:

| arm | labels | seed 11 |
|---|---|---:|
| an1 | panel consensus | **0.9122** |
| an2 | self-consistency (3 samples, majority) | 0.9016 |
| | | **-1.06** |

The prediction was that the gap would SHRINK because the fold already merges away
MEDIUM/COMPLEX, the boundary where self-consistency was least stable. It did more
than shrink — it reversed. Direct evidence for the mechanism: **the two labelling
processes agree on 93.2% of rows at four tiers and 97.8% once folded.** Two thirds
of the disagreement between them was about a boundary the fold deletes.

**Collapsing the taxonomy and denoising the labels attack the same disagreement,
and you do not get to bank both.** Collapsing is far cheaper — it is a fold, not
three labelling passes over 20,000 prompts.

That is the fifth label-quality intervention on complexity to fail, now on both
taxonomies. The label axis is closed on this signal.
