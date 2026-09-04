#!/usr/bin/env bash
# Round F: add the disagreement-selected rows on top of everything.
#
# These target the population margin-based selection structurally cannot reach:
# prompts the model is CONFIDENT about and still gets wrong (32 of 48 residual
# errors carry p>0.90). Enrichment is real but weaker than the margin route --
# 27.3% jury-confirmed model errors against a ~13% base rate, versus 55.4% for
# low-margin rows -- so this is additive, not a replacement.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
ALL="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl,$D/complexity-disagree.jsonl,$D/complexity-disagree-contested.jsonl"

echo "########## cx-f1-minilm-all ##########"
./.venv/bin/python harness/train.py complexity --arch head \
  --base sentence-transformers/all-MiniLM-L6-v2 --train "$ALL" \
  --epochs 6 --lr 5e-5 --maxlen 256 --soft --tag cx-f1-minilm-all 2>&1 \
  | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"

echo "########## cx-f2-mbert-all ##########"
./.venv/bin/python harness/train.py complexity --arch head \
  --base answerdotai/ModernBERT-base --train "$ALL" \
  --epochs 5 --lr 3e-5 --maxlen 512 --soft --tag cx-f2-mbert-all 2>&1 \
  | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
echo "ROUND F COMPLETE"
