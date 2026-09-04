#!/usr/bin/env bash
# Round U: sensitivity on the doubled enterprise corpus.
#
# Enterprise data is the lever with a track record on THIS signal: 1,604 ->
# 13,733 rows took CONFIDENTIAL recall 0.33 -> 0.84 and REGULATED 0.51 -> 0.65.
# It has now doubled again to 31,168, with REGULATED 6,226 / CONFIDENTIAL 5,315
# / NEVER_EGRESS 3,873 -- the tiers that gate egress finally have thousands of
# examples each rather than hundreds.
#
# Only two seeds needed: sensitivity's spread is 0.0014, so a difference this
# lever might plausibly produce is resolvable almost immediately.
#
# Recipe held at the measured best: MiniLM + soft targets + tier escalation.
# Round S established that ModernBERT (-1.1), LoRA (-1.1) and the vLLM SR
# regularisation recipe (-1.7) all lose to it by 8-12x the noise floor.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
for S in 11 22; do
  echo "########## se-u-big-seed$S ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" --train "$SE" \
      --epochs 4 --lr 5e-5 --maxlen 256 --soft --escalate 1.0 --seed $S \
      --tag "se-u-big-seed$S" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
done
echo "ROUND U COMPLETE"
