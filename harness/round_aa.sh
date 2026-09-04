#!/usr/bin/env bash
# Round AA: corpus composition ablation for sensitivity.
#
# §64 measured each training file's separability from the entsec eval: enterprise
# 78.3%, v2 92.8%, real 98.2%. A third of the corpus is drawn from distributions
# a linear probe tells apart from the eval almost perfectly, and §63 showed
# off-distribution rows can be actively worse than no rows.
#
# BOTH evals are reported on every arm. Dropping WildChat rows should help entsec
# and hurt real-gold; the question is the exchange rate, and an arm that only
# reported entsec would look like a free win.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
ALL="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
ENT="$D/sensitivity-enterprise.jsonl"
ENTD="$D/sensitivity-enterprise.jsonl,$D/sensitivity-distill.jsonl"
NOV2="$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"

run () {
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" --train "$2" \
      --epochs 4 --lr 5e-5 --maxlen 256 --soft --escalate 0.0 --seed 11 \
      --tag "$1" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
}
run se-aa1-all      "$ALL"
run se-aa2-ent      "$ENT"
run se-aa3-ent-dist "$ENTD"
run se-aa4-no-v2    "$NOV2"
echo "ROUND AA COMPLETE"
