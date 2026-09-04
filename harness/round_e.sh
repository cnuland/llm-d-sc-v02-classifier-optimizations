#!/usr/bin/env bash
# Round E: the two winning ingredients together.
#
# Round B showed ModernBERT takes the soft-target gain WITHOUT the register
# penalty MiniLM pays (heldout-v1 0.875 vs 0.825) -- capacity, not the loss.
# Round D adds 8,986 actively-selected rows on which the model was wrong 55.4%
# of the time. Neither has been tried with the other.
#
# Also runs the cost/sensitivity ordinal arms, which are still pending, so a
# single queue covers all three signals.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MBERT=answerdotai/ModernBERT-base

echo "########## cx-e1-mbert-active ##########"
./.venv/bin/python harness/train.py complexity --arch head --base "$MBERT" \
  --train "$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl" \
  --epochs 5 --lr 3e-5 --maxlen 512 --soft --tag cx-e1-mbert-active 2>&1 \
  | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"

exec ./harness/round_c.sh 6 5e-5 512 --soft
