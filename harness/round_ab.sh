#!/usr/bin/env bash
# Round AB: complexity trained on labels produced under the v2 RUBRIC.
#
# §54's payoff. The v2 wording -- COMPLEX requires 2 of 3 countable tests rather
# than "competent experts would differ" -- raised two-labeller agreement from
# 71.3% to 78.3% on the 600 prompts where v1 labellers split. 59,582 real prompts
# have now been relabelled from scratch under it.
#
# TEMPER THIS BEFORE READING THE RESULT. Corpus-wide agreement under v2 came out
# at 87.7%, against 87.5% for v1 (§51). Essentially unchanged. §54's +7.0 was
# measured on the hardest ~12% of the corpus, and fixing 29% of those predicts
# roughly 91% corpus-wide; the observed 87.7% is well short of that. So either
# the A/B's 600 rows are not representative of all v1-split rows, or the v2
# wording resolves some disagreements while opening others -- its residual table
# did show a MEDIUM/SIMPLE confusion v1 never produced.
#
# The prediction on record is therefore: a gain at or below complexity's
# 1.06-point floor. Recording that BEFORE the run is the only thing that makes
# the number afterwards worth anything.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
CORE="$D/complexity-v2.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl,$D/complexity-distill.jsonl"
for S in 11 22 33; do
  echo "########## cx-ab-v2rubric-seed$S ##########"
  ./.venv/bin/python harness/train.py complexity --arch head --base "$MINILM" \
      --train "$CORE,$D/complexity-real-v2rubric.jsonl" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --seed $S --tag "cx-ab-v2rubric-seed$S" 2>&1 \
    | grep -E "^\[|refined-gold|real-gold|trained in" || echo "  FAILED"
done
echo "ROUND AB COMPLETE"
