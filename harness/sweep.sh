#!/usr/bin/env bash
# Hyperparameter sweep for the winning architecture on one signal.
#
# The matrix answered "which corpus, which architecture, which base". This
# answers "how hard can that configuration be pushed" -- epochs, learning rate
# and base size -- selecting on real-gold, which is the only eval that predicts
# production behaviour.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n')
export TOKENIZERS_PARALLELISM=false
SIG=${1:-complexity}
D=data/train
MIX="$D/$SIG-v2.jsonl,$D/$SIG-real.jsonl"
[ -f "$D/$SIG-enterprise.jsonl" ] && MIX="$MIX,$D/$SIG-enterprise.jsonl"

run () {  # run <tag> <base> <epochs> <lr>
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py "$SIG" --arch head --base "$2" \
      --train "$MIX" --epochs "$3" --lr "$4" --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
}

MINILM=sentence-transformers/all-MiniLM-L6-v2
MBERT=answerdotai/ModernBERT-base

run "$SIG-sw-minilm-e6"    "$MINILM" 6 2e-5
run "$SIG-sw-minilm-e6lr5" "$MINILM" 6 5e-5
run "$SIG-sw-mbert-e5"     "$MBERT"  5 2e-5
run "$SIG-sw-mbert-e5lr3"  "$MBERT"  5 3e-5
run "$SIG-sw-mbert-e8"     "$MBERT"  8 2e-5
