#!/usr/bin/env bash
# Round K: the two things that actually worked, combined.
#
#   corpus     enterprise training data 5,582 -> 13,733 rows, with the sensitive
#              tiers finally well represented (REGULATED 2,718, CONFIDENTIAL
#              2,264, NEVER_EGRESS 1,680 versus a few hundred before)
#   mechanism  tier-escalated class weighting, the best single mechanism tried:
#              CONFIDENTIAL 0.33->0.72, REGULATED 0.51->0.62, NEVER_EGRESS
#              0.75->0.91
#
# Neither has been tried with the other. REGULATED at 0.62 is the weakest
# sensitive tier and the one with the most new training support, so it is the
# metric to watch rather than aggregate accuracy.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"

run () {
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" \
      --train "$SE" --epochs 6 --lr 5e-5 --maxlen 256 --soft $2 --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
}

run "se-k1-big-esc"  "--escalate 1.0"
run "se-k2-big-cw"   "--classweight"
run "se-k3-big"      ""
echo "ROUND K COMPLETE"
