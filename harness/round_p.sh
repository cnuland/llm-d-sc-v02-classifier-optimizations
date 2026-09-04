#!/usr/bin/env bash
# Round P: multi-seed members for the ensemble.
#
# Ensembling is the only lever that has paid every time it was tried (+1.9 to
# +2.4 on all three signals), and every ensemble so far has used
# ARCHITECTURALLY different members. Seed diversity is a separate, additive
# source: same recipe, different initialisation and data order, so the members
# land in different local optima and make different mistakes.
#
# Cheap to test -- MiniLM arms at ~18 min against ~90 for an encoder arm, which
# is why this runs before the encoder sweep despite being queued after it.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
CX="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl,$D/complexity-distill.jsonl"

for S in 11 22 33; do
  echo "########## cx-p-seed$S ##########"
  ./.venv/bin/python harness/train.py complexity --arch head --base "$MINILM" \
      --train "$CX" --epochs 4 --lr 5e-5 --maxlen 256 --soft --seed $S \
      --tag "cx-p-seed$S" 2>&1 | grep -E "^\[|acc=|trained in" || echo "  FAILED"
done
echo "ROUND P COMPLETE"
