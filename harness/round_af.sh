#!/usr/bin/env bash
# Round AF: the small-vs-large route -- the one deployed decision NOT in the 90s.
#
# §75's table, folding existing models with no retraining:
#   complexity "is reasoning needed"      95.79%   (majority 86.53%)
#   cost       "short vs long"            92.44%   (majority 50.42%)
#   sensitivity"block at NEVER_EGRESS"    94.42%   (majority 85.04%)
#   complexity "small vs large route"     88.22%   (majority 75.42%)  <- this one
#
# It is also the decision that fires on EVERY request, so it is the one whose
# error rate actually costs money. LARGE is 23.2% of the eval and 6.0% of the
# training corpus -- a much steeper skew than the fold suggests -- so class
# weighting is on and LARGE recall is the number to read, not accuracy.
#
# Two arms. af1 is the plain fold. af2 adds the two synthetic PAIR corpora, which
# exist precisely to sharpen MEDIUM/COMPLEX -- the boundary this fold cuts on.
# They were built for the 4-way task and have never been tried against the
# binary decision they are actually about.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
BASE="$D/route-v2.jsonl,$D/route-real.jsonl,$D/route-active.jsonl,$D/route-distill.jsonl,$D/route-real-contested.jsonl"
PAIRS="$D/route-pair-MEDIUM-COMPLEX.jsonl"
for S in 11 22; do
  echo "########## rt-af1-seed$S ##########"
  ./.venv/bin/python harness/train.py route --arch head --base "$MINILM" --train "$BASE" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed $S \
      --tag "rt-af1-seed$S" 2>&1 | grep -E "^\[|real-gold|refined-gold|trained in" || echo "  FAILED"
done
echo "########## rt-af2-pairs ##########"
./.venv/bin/python harness/train.py route --arch head --base "$MINILM" --train "$BASE,$PAIRS" \
    --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed 11 \
    --tag "rt-af2-pairs" 2>&1 | grep -E "^\[|real-gold|refined-gold|trained in" || echo "  FAILED"
echo "ROUND AF COMPLETE"
