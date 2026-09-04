#!/usr/bin/env bash
# Round AC: attribute Round X's bge gain, and combine it with Round W's.
#
# se-x1-bge-raw scored 0.8119 on entsec vs the MiniLM incumbent's 0.7808 median
# -- +3.11, 22x the 0.14 floor. But x1 also changed the SCHEDULE (3 epochs at
# 3e-5, chosen because the bigger model is easier to overcook) and I did not run
# a MiniLM control at that schedule. As written, the +3.11 confounds encoder with
# schedule, and none of Round X's arms separate them: x3 is MiniLM on the
# BALANCED corpus, so it varies two things at once.
#
# ac1 is that missing control. ac2 stacks the two independent wins -- bge from
# Round X, escalate 0.0 from Round W (§66) -- which is only worth running because
# they act on different parts of the system, one on the representation and one on
# the loss.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
BGE=BAAI/bge-base-en-v1.5
RAW="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
run () {  # tag base escalate
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$2" --train "$RAW" \
      --epochs 3 --lr 3e-5 --maxlen 256 --soft --escalate "$3" --seed 11 \
      --tag "$1" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
}
run se-ac1-mini-raw-3ep  "$MINILM" 1.0    # schedule control for x1
run se-ac2-bge-esc0      "$BGE"    0.0    # stack the two wins
echo "########## se-ac3-bge-esc0-s22 ##########"
./.venv/bin/python harness/train.py sensitivity --arch head --base "$BGE" --train "$RAW" \
    --epochs 3 --lr 3e-5 --maxlen 256 --soft --escalate 0.0 --seed 22 \
    --tag se-ac3-bge-esc0-s22 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
echo "ROUND AC COMPLETE"
