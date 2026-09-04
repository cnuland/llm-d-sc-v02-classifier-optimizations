#!/usr/bin/env bash
# Round S: mechanisms on sensitivity, where they can actually be measured.
#
# Sensitivity's seed spread is 0.0014 — roughly 10x tighter than cost (0.0135)
# and complexity (0.0106), because its larger corpus and skewed prior damp
# run-to-run drift. A single run there resolves differences that would need
# three seeds on cost.
#
# So the mechanisms that were untestable elsewhere get their clean test here:
# ModernBERT on the expanded corpus (se-k1-big-esc was MiniLM), LoRA (only ever
# run on cost), and the vLLM SR regularisation recipe.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MBERT=answerdotai/ModernBERT-base
MINILM=sentence-transformers/all-MiniLM-L6-v2
SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"

run () {  # run <tag> <base> <extra>
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$2" --train "$SE" \
      --epochs 5 --lr "${4:-5e-5}" --maxlen 256 --soft --escalate 1.0 $3 --tag "$1" 2>&1 \
    | grep -E "^\[|entsec-gold|real-gold|trained in|latency|trainable" || echo "  FAILED"
}
run "se-s1-mbert-big"  "$MBERT"  ""                                        3e-5
run "se-s2-lora16"     "$MINILM" "--lora 16 --wd 0.1 --sched cosine"       2e-4
run "se-s3-vsr-reg"    "$MINILM" "--wd 0.1 --sched cosine --warmup 0.06"   5e-5
echo "ROUND S COMPLETE"
