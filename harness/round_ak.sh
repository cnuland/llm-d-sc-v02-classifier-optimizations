#!/usr/bin/env bash
# Round AK: isolate the training-loop lead from §91.
#
# perjuror.py's CONTROL arm -- a plain shared head -- beat train.py's incumbent by
# +0.80 on complexity refined gold (0.9096 vs 0.9016). That is the largest
# unexplained gap left on this signal, but perjuror.py differs from train.py in
# THREE ways at once, so §91 recorded it as a lead rather than a finding:
#
#   1. masked-mean pooling over the last hidden state, instead of BERT's [CLS]
#      pooler followed by tanh
#   2. OneCycleLR, instead of linear warmup + linear decay
#   3. 161,617 training rows (every complexity-*.jsonl) instead of round V's
#      five-corpus subset
#
# (3) is the cheapest to test and the most likely explanation, so it goes first:
# train.py on perjuror.py's full row set, everything else unchanged. If that
# recovers the +0.80, the "loop" was never the point and the answer is corpus
# coverage. If it does not, the remaining gap is pooling or schedule and ak2
# isolates the schedule.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
ALL=$(ls $D/complexity-*.jsonl | grep -vE "panel20k|selfconsistency|v2rubric|-pool" | tr '\n' ',' | sed 's/,$//')
CORE="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl,$D/complexity-distill.jsonl,$D/complexity-resolved.jsonl"
for S in 11 22; do
  T="cx-ak1-allrows-seed$S"
  echo "########## $T  (corpus = every complexity file) ##########"
  ./harness/runarm.sh "$T" ./.venv/bin/python harness/train.py complexity --arch head \
      --base "$MINILM" --train "$ALL" --epochs 3 --lr 5e-5 --maxlen 256 --soft --seed $S --tag "$T"
done
echo "########## cx-ak2-cosine  (schedule only) ##########"
./harness/runarm.sh cx-ak2-cosine ./.venv/bin/python harness/train.py complexity --arch head \
    --base "$MINILM" --train "$CORE" --epochs 3 --lr 5e-5 --maxlen 256 --soft \
    --sched cosine --warmup 0.10 --seed 11 --tag cx-ak2-cosine
echo "ROUND AK COMPLETE"
