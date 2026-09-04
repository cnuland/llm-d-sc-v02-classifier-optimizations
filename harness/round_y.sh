#!/usr/bin/env bash
# Round Y: does maxlen 512 matter for SENSITIVITY, where §14 found it did not
# for complexity?
#
# §14's null was measured on complexity, where the tier describes the request as
# a whole and the opening usually settles it. Sensitivity's tier is the maximum
# over spans: one API key in paragraph six sets NEVER_EGRESS no matter how benign
# the first 256 tokens read. Truncation there does not blur the label, it deletes
# the evidence for it.
#
# 15.6% of entsec prompts exceed 256 tokens vs complexity's 8.1%, and at 3.34
# tokens/word (§58) these prompts are far denser than prose, so the truncated
# tail is larger than the token count suggests.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
for S in 11 22; do
  echo "########## se-y-512-seed$S ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" --train "$SE" \
      --epochs 4 --lr 5e-5 --maxlen 512 --soft --escalate 1.0 --seed $S \
      --tag "se-y-512-seed$S" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
done
echo "ROUND Y COMPLETE"
