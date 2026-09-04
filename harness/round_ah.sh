#!/usr/bin/env bash
# Round AH: bge-SMALL -- the deployment sweet spot §80 implies.
#
# §80 left an awkward split: bge-base is the better classifier (+4.5 at fixed
# schedule and corpus) but MiniLM is the better component (6.1 ms vs 21.9 ms,
# and §68's matched-containment comparison favours it). llm-d-sc serves this on
# CPU, so latency is not a footnote.
#
# The frozen pair probe, all five encoders on identical rows, says a small strong
# encoder recovers most of the gain:
#
#   encoder                params   eval-bal   CONFIDENTIAL recall
#   all-MiniLM-L6-v2         22M     73.02%        71.82%     <- shipped
#   bge-small-en-v1.5        33M     75.05%        75.60%
#   e5-small-v2              33M     73.94%        68.38%
#   gte-small                33M     72.21%        64.26%
#   bge-base-en-v1.5        109M     76.90%        77.32%     <- reference
#
# bge-small captures 52% of bge-base's gain at ~1.5x MiniLM's cost rather than
# 3.6x, and has the best minority recall of the small encoders. e5 and gte do
# NOT transfer down -- e5-base beat bge-base on this pair earlier, e5-small is
# worse than bge-small, so "the small version of the best base model" is not a
# safe inference and is why all three were probed.
#
# Recipe is the current best: natural corpus (§78), escalate 0.0 (§66).
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
NAT="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
for S in 11 22; do
  echo "########## se-ah-bgesmall-seed$S ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base BAAI/bge-small-en-v1.5 \
      --train "$NAT" --epochs 4 --lr 5e-5 --maxlen 256 --soft --escalate 0.0 --seed $S \
      --tag "se-ah-bgesmall-seed$S" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
done
echo "ROUND AH COMPLETE"
