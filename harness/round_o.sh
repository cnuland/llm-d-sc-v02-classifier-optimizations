#!/usr/bin/env bash
# Round O: sensitivity distillation onto the enterprise pool, with control.
#
# By the rule established on cost (§45), distillation gain scales with the
# teacher's uncertainty. Sensitivity's ensemble is the least confident of the
# three signals, so it should gain the most -- and unlike the WildChat pool
# (85% PUBLIC), the 8,995-row enterprise pool actually covers the tiers that
# gate egress.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
BASE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"

run () {
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" \
      --train "$2" --epochs 5 --lr 5e-5 --maxlen 256 --soft $3 --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
}
run "se-o1-distill"     "$BASE,$D/sensitivity-distill.jsonl" "--escalate 1.0"
run "se-o2-control"     "$BASE"                              "--escalate 1.0"
echo "ROUND O COMPLETE"
