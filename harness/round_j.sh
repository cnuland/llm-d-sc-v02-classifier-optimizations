#!/usr/bin/env bash
# Round J: transfer tier-escalated class weighting to cost and complexity.
#
# It was the single most effective mechanism tried on sensitivity: every
# sensitive tier improved (CONFIDENTIAL 0.33->0.72, REGULATED 0.51->0.62,
# NEVER_EGRESS 0.75->0.91) and it produced the best macro F1 on both evals.
# It is also the CORRECTED form of the ordinal idea -- symmetric smoothing
# cancelled because it leaked mass in both directions; escalation leaks upward
# only.
#
# For cost the asymmetry is different but still real: under-estimating cost
# under-provisions capacity, which is worse than over-provisioning it. For
# complexity there is no such asymmetry, so that arm gets plain inverse-sqrt
# class weighting with no escalation -- the imbalance is real (MEDIUM dominates)
# even where the direction is not.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2

run () {  # run <tag> <signal> <corpora> <flags>
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py "$2" --arch head --base "$MINILM" --train "$3" \
      --epochs 6 --lr 5e-5 --maxlen 256 --soft $4 --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency|NOTE" || echo "  FAILED"
}

CO="$D/cost-v2.jsonl,$D/cost-real.jsonl,$D/cost-real-contested.jsonl"
CX="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl"

run "co-j1-cw"      cost       "$CO" "--classweight"
run "co-j2-cw-esc"  cost       "$CO" "--escalate 1.0"
run "cx-j1-cw"      complexity "$CX" "--classweight"
echo "ROUND J COMPLETE"
