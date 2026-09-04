#!/usr/bin/env bash
# Round AE: dedicated classifiers for the two remaining DEPLOYED decisions.
#
# §75 measured what the router actually asks, and folded the existing 4-way
# models' outputs onto it with no retraining:
#
#   decision                              jury agr   folded acc   majority
#   complexity "is reasoning needed"         94.6%       95.79%     86.53%
#   cost "short vs long generation"          88.1%       92.44%     50.42%
#   sensitivity "block at NEVER_EGRESS"      96.4%       94.42%     85.04%   (Round AD)
#
# Those are lower bounds -- models trained for a 4- or 5-way task, folded. Each is
# trained directly here.
#
# Majority baselines are in the table for a reason. §75 also tested cost's
# "reserve a big budget" fold, which scores 93.78% against a 93.45% majority
# baseline: an impressive-looking number for a decision no better than always
# answering no. The three folds trained here clear their baselines by 9, 42 and
# 9 points respectively, which is why they are worth training and that one is not.
#
# YES is 3.2% of the reasoning corpus, so accuracy alone would flatter a model
# that never fires; minority recall is what to read.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
RS="$D/reasoning-v2.jsonl,$D/reasoning-real.jsonl,$D/reasoning-active.jsonl,$D/reasoning-distill.jsonl,$D/reasoning-real-contested.jsonl"
GL="$D/genlen-v2.jsonl,$D/genlen-real.jsonl,$D/genlen-real-contested.jsonl,$D/genlen-distill.jsonl"
for S in 11 22; do
  echo "########## rs-ae1-seed$S ##########"
  ./.venv/bin/python harness/train.py reasoning --arch head --base "$MINILM" --train "$RS" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed $S \
      --tag "rs-ae1-seed$S" 2>&1 | grep -E "^\[|real-gold|refined-gold|trained in" || echo "  FAILED"
  echo "########## gl-ae2-seed$S ##########"
  ./.venv/bin/python harness/train.py genlen --arch head --base "$MINILM" --train "$GL" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --seed $S \
      --tag "gl-ae2-seed$S" 2>&1 | grep -E "^\[|real-gold|refined-gold|trained in" || echo "  FAILED"
done
echo "ROUND AE COMPLETE"
