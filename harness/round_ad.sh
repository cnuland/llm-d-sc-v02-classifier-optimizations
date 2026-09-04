#!/usr/bin/env bash
# Round AD: a dedicated BINARY egress classifier.
#
# §74 measured every candidate taxonomy fold. Exactly one reaches the high 90s on
# the number that bounds everything else -- jury agreement:
#
#   sensitivity as shipped (5 tiers)   agreement 74.5%   model 72.9%
#   merge INTERNAL+CONFIDENTIAL        agreement 80.6%   model 79.4%
#   BINARY gate at CONFIDENTIAL        agreement 84.8%   model 82.3%
#   BINARY gate at NEVER_EGRESS        agreement 96.4%   model 94.4%
#
# 94.4% is what the EXISTING 5-way model scores when its output is folded, with
# no retraining -- a lower bound. A model trained directly on the binary decision
# should do better, and 96.4% agreement means high 90s is a real target here
# rather than an artefact of grading against noisy labels.
#
# This is also the decision with the most at stake: live credentials and
# privileged material leaving the network.
#
# BLOCK is 12.5% of training and 17.1% of the eval, and the majority-class
# baseline on entsec-gold is 82.9% -- so accuracy alone is a weak metric here and
# BLOCK recall is reported alongside it. --classweight (inverse-sqrt, NOT the
# tier escalation §66 removed) handles the imbalance.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
EG="$D/egress-v2.jsonl,$D/egress-real.jsonl,$D/egress-enterprise.jsonl,$D/egress-real-contested.jsonl"
for S in 11 22; do
  echo "########## eg-ad1-cw-seed$S ##########"
  ./.venv/bin/python harness/train.py egress --arch head --base "$MINILM" --train "$EG" \
      --epochs 4 --lr 5e-5 --maxlen 256 --soft --classweight --seed $S \
      --tag "eg-ad1-cw-seed$S" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
done
echo "########## eg-ad2-nocw ##########"
./.venv/bin/python harness/train.py egress --arch head --base "$MINILM" --train "$EG" \
    --epochs 4 --lr 5e-5 --maxlen 256 --soft --seed 11 \
    --tag "eg-ad2-nocw" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
echo "ROUND AD COMPLETE"
