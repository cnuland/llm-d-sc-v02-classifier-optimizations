#!/usr/bin/env bash
# Round N: cost distillation, where the teacher actually has uncertainty to transfer.
#
# complexity's teacher had confidence median 0.998 (2.5% below 0.6) and
# distillation behaved as plain extra data: +0.3 on refined gold.
# cost's teacher has median 0.897 with 12.5% below 0.6 -- real uncertainty
# structure, which is the "dark knowledge" distillation is meant to move.
# Prediction recorded before the run: a LARGER gain than complexity's.
#
# Control arm again, identical corpus minus the distilled rows, so a gain cannot
# be attributed to volume by default.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
BASE="$D/cost-v2.jsonl,$D/cost-real.jsonl,$D/cost-real-contested.jsonl"

run () {
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py cost --arch head --base "$MINILM" \
      --train "$2" --epochs 4 --lr 5e-5 --maxlen 256 --soft --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
}
run "co-n1-distill" "$BASE,$D/cost-distill.jsonl"
run "co-n2-control" "$BASE"
echo "ROUND N COMPLETE"
