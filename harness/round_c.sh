#!/usr/bin/env bash
# Round C: carry Round B's winning recipe to sensitivity and cost.
#
# Sensitivity is the one with unexploited data: the enterprise corpus grew
# 1,604 -> 5,582 and NEVER_EGRESS went from 5 rows to 675, which is the tier
# that actually gates egress and was previously unmeasurable.
# Cost has the lowest juror agreement of the three (74.6%), so its 2,279
# contested rows carry proportionally the most information.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
MBERT=answerdotai/ModernBERT-base
EP=${1:-6}; LR=${2:-5e-5}; ML=${3:-512}; SOFT=${4:---soft}

run () {   # run <tag> <signal> <base> <corpora> <extra-flags>
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py "$2" --arch head --base "$3" --train "$4" \
      --epochs "$EP" --lr "$LR" --maxlen "$ML" $SOFT ${5:-} --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency|NOTE" || echo "  FAILED"
}

SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
CO="$D/cost-v2.jsonl,$D/cost-real.jsonl,$D/cost-real-contested.jsonl"

# Both signals get an ordinal arm: their tiers form a genuine total order and
# 78% of observed errors are adjacent-tier, which plain cross-entropy cannot see.
run "se-c1-minilm"      sensitivity "$MINILM" "$SE" ""
run "se-c2-minilm-ord"  sensitivity "$MINILM" "$SE" "--ordinal"
run "se-c3-mbert-ord"   sensitivity "$MBERT"  "$SE" "--ordinal"
run "co-c1-minilm"      cost        "$MINILM" "$CO" ""
run "co-c2-minilm-ord"  cost        "$MINILM" "$CO" "--ordinal"
run "co-c3-mbert-ord"   cost        "$MBERT"  "$CO" "--ordinal"
echo "ROUND C COMPLETE"
