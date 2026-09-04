#!/usr/bin/env bash
# Round I: vary the BASE ENCODER, which this project has barely explored.
#
# Every arm so far is MiniLM-L6 (23M) or ModernBERT-base (149M). Meanwhile the
# data axis has saturated -- 5.5k -> 21k rows helped, 21k -> 34.5k hurt, because
# the 2,059 disagreement-selected rows came from a screen with only 24.8%
# precision and carry more label noise than the rest.
#
# So use the 21k corpus that actually performed best (dropping the disagree
# rows) and vary the encoder instead. bge and gte are strong retrieval encoders
# never tried here with a classification head; ModernBERT-large tests whether
# the ModernBERT-base gain continues with size.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
# best-performing corpus: v2 + real + real-contested + active, WITHOUT disagree
CX="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl"

run () {  # run <tag> <base> <epochs> <lr>
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py complexity --arch head --base "$2" \
      --train "$CX" --epochs "$3" --lr "$4" --maxlen 256 --soft --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency|soft targets requested" || echo "  FAILED"
}

run "cx-i1-bge"          BAAI/bge-base-en-v1.5              5 3e-5
run "cx-i2-e5"           intfloat/e5-base-v2                5 3e-5
run "cx-i3-mpnet"        sentence-transformers/all-mpnet-base-v2  5 3e-5
# ModernBERT-large (395M) dropped: on this hardware it is hours per arm, and
# the base-size gain measured so far is +0.5 pts for 5x latency.
echo "ROUND I COMPLETE"
