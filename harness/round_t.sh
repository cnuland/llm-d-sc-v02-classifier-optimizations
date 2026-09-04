#!/usr/bin/env bash
# Round T: complexity on the doubled corpus — the treatment that gave cost +2.16.
#
# Cost went 6,697 -> 43,717 agreed rows and gained a verified +2.16 against its
# 1.35 pt floor, the only change in this project to clear a noise floor by more
# than 2x. Complexity's real corpus has now doubled the same way.
#
# Three seeds, because complexity's floor is 1.06 points and a single run cannot
# resolve anything smaller. Baseline to beat: median 0.8963 refined gold.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
CX="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl,$D/complexity-distill.jsonl"
for S in 11 22 33; do
  echo "########## cx-t-big-seed$S ##########"
  ./.venv/bin/python harness/train.py complexity --arch head --base "$MINILM" --train "$CX" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --seed $S --tag "cx-t-big-seed$S" 2>&1 \
    | grep -E "^\[|refined-gold|real-gold|trained in" || echo "  FAILED"
done
echo "ROUND T COMPLETE"
