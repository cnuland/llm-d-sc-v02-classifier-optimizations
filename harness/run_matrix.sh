#!/usr/bin/env bash
# Training matrix for one signal.
#
# Arms are chosen to separate the three variables that could each explain the
# real-traffic gap, so a win can be attributed:
#   corpus       synthetic only | real only | mixed
#   architecture embed+anchors (drop-in) | softmax head (runtime change)
#   base model   MiniLM 23M (current) | ModernBERT-base 149M | bge-base 109M
#
# Every arm reports against heldout-v1, real-gold, real-contested and (for
# sensitivity) enterprise-gold, plus CPU latency -- llm-d-sc serves on CPU and
# the measured per-replica ceiling scales with model size, so an accuracy win
# that costs 8x latency is not automatically a win.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n')
export TOKENIZERS_PARALLELISM=false
SIG=${1:-complexity}
EP=${2:-3}
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
MBERT=answerdotai/ModernBERT-base
BGE=BAAI/bge-base-en-v1.5

run () {  # run <tag> <arch> <base> <corpora>
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py "$SIG" --arch "$2" --base "$3" \
      --train "$4" --epochs "$EP" --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
}

SYN=$D/$SIG-v2.jsonl
REAL=$D/$SIG-real.jsonl
ENT=$D/$SIG-enterprise.jsonl
# sensitivity gets a third corpus: unconditioned enterprise scenes. WildChat is
# 93% PUBLIC, so real traffic alone cannot teach the tiers that gate egress.
MIX="$SYN,$REAL"
[ -f "$ENT" ] && MIX="$SYN,$REAL,$ENT"

[ -f "$SYN"  ] && run "$SIG-embed-minilm-syn"   embed "$MINILM" "$SYN"
[ -f "$REAL" ] && run "$SIG-embed-minilm-real"  embed "$MINILM" "$REAL"
[ -f "$REAL" ] && run "$SIG-embed-minilm-mix"   embed "$MINILM" "$MIX"
[ -f "$REAL" ] && run "$SIG-embed-bge-mix"      embed "$BGE"    "$MIX"
[ -f "$REAL" ] && run "$SIG-head-minilm-mix"    head  "$MINILM" "$MIX"
[ -f "$REAL" ] && run "$SIG-head-mbert-mix"     head  "$MBERT"  "$MIX"
[ -f "$ENT"  ] && run "$SIG-head-minilm-ent"    head  "$MINILM" "$ENT"
