#!/usr/bin/env bash
# Round B: with the corpora doubled, isolate the three remaining levers on the
# priority signal, then carry the winner to the others.
#
#   soft targets   include the 2,358 contested rows as vote distributions
#                  instead of discarding them
#   maxlen 512     8.1% of real prompts exceed 256 tokens and are truncated
#   base model     MiniLM 23M vs ModernBERT 149M
#
# Baseline to beat: complexity-sw-minilm-e6lr5 at 0.8589 real-gold, trained on
# the previous, half-size corpus.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
MBERT=answerdotai/ModernBERT-base

run () {  # run <tag> <signal> <base> <corpora> <epochs> <lr> <maxlen> [--soft]
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py "$2" --arch head --base "$3" --train "$4" \
      --epochs "$5" --lr "$6" --maxlen "$7" ${8:-} --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
}

CX_HARD="$D/complexity-v2.jsonl,$D/complexity-real.jsonl"
CX_SOFT="$CX_HARD,$D/complexity-real-contested.jsonl"

run "cx-b1-hard-256"      complexity "$MINILM" "$CX_HARD" 6 5e-5 256
run "cx-b2-soft-256"      complexity "$MINILM" "$CX_SOFT" 6 5e-5 256 --soft
run "cx-b3-soft-512"      complexity "$MINILM" "$CX_SOFT" 6 5e-5 512 --soft
run "cx-b4-soft-512-mbert" complexity "$MBERT" "$CX_SOFT" 5 3e-5 512 --soft
echo "ROUND B COMPLETE"
