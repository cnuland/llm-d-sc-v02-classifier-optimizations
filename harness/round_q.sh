#!/usr/bin/env bash
# Round Q: seed replicates for cost and sensitivity.
#
# §47 measured a ~1.1 point noise floor and exposed a selection problem: every
# published model was chosen as the best of several single-seed runs, so its
# reported number is biased upward relative to a fresh run of the same recipe.
#
# Three seeds each gives a MEDIAN to publish instead of a maximum, and a
# per-signal variance estimate so future deltas can be judged against the right
# floor rather than complexity's.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
CO="$D/cost-v2.jsonl,$D/cost-real.jsonl,$D/cost-real-contested.jsonl,$D/cost-distill.jsonl"
SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"

for S in 11 22 33; do
  echo "########## co-q-seed$S ##########"
  ./.venv/bin/python harness/train.py cost --arch head --base "$MINILM" --train "$CO" \
      --epochs 4 --lr 5e-5 --maxlen 256 --soft --seed $S --tag "co-q-seed$S" 2>&1 \
    | grep -E "^\[|refined-gold|real-gold|trained in" || echo "  FAILED"
done
for S in 11 22 33; do
  echo "########## se-q-seed$S ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" --train "$SE" \
      --epochs 5 --lr 5e-5 --maxlen 256 --soft --escalate 1.0 --seed $S --tag "se-q-seed$S" 2>&1 \
    | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
done
echo "ROUND Q COMPLETE"
