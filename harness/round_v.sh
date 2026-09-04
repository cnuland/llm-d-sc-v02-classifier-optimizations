#!/usr/bin/env bash
# Round V: complexity with tie-resolved labels — the label-QUALITY lever.
#
# §51/§52: data volume pays only below ~80% juror agreement. Complexity is at
# 87.5%, and its 3.3x corpus bought +0.26 against a 1.06 floor — nothing. The
# rule prescribes label quality instead, so the 7,435 contested rows have been
# put to a third independent juror; those resolving to a 2-of-3 majority now
# train as HARD labels rather than 50/50 soft targets.
#
# Control arm uses the identical corpus with those rows still as soft targets,
# so the comparison isolates label quality from row count. Three seeds, floor
# 1.06 points.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
CORE="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl,$D/complexity-distill.jsonl"
for S in 11 22 33; do
  echo "########## cx-v-resolved-seed$S ##########"
  ./.venv/bin/python harness/train.py complexity --arch head --base "$MINILM" \
      --train "$CORE,$D/complexity-resolved.jsonl" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --seed $S --tag "cx-v-resolved-seed$S" 2>&1 \
    | grep -E "^\[|refined-gold|real-gold|trained in" || echo "  FAILED"
done
echo "ROUND V COMPLETE"
