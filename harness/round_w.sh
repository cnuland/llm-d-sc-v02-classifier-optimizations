#!/usr/bin/env bash
# Round W: is sensitivity's error caused by its own class weighting?
#
# sens_headroom.py split the error decisively. The model misses 156 rows where
# all three jurors AGREED -- 16.4% of the set. Rubric ambiguity cannot explain
# those, and even perfect labels on every contested row cap the score at 83.6%.
# So sensitivity is MODEL-limited, unlike complexity which is rubric-limited.
#
# And the error has a direction. 81 of those 156 (52%) are INTERNAL escalated
# upward: INTERNAL->CONFIDENTIAL 40, ->REGULATED 22, ->NEVER_EGRESS 19.
#
# --escalate 1.0 multiplies class weight upward through the tier ladder. INTERNAL
# is tier 1 of 5, so it carries the SMALLEST weight while being the majority
# class (486/949). The model was trained to do exactly what it is doing.
#
# That was a deliberate trade -- escalation buys gate containment. So sweep it
# and report BOTH numbers, because accuracy alone would recommend dropping a
# safety property, and containment alone would hide 8 points of accuracy.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
for E in 0.0 0.5; do
  for S in 11 22; do
    T="se-w-esc${E}-seed$S"
    echo "########## $T ##########"
    ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" --train "$SE" \
        --epochs 4 --lr 5e-5 --maxlen 256 --soft --escalate "$E" --seed $S \
        --tag "$T" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
  done
done
echo "ROUND W COMPLETE"
