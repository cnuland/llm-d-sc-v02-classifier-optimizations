#!/usr/bin/env bash
# Round D: fold in the actively-selected rows.
#
# 8,986 new rows chosen because the model could not separate them (median margin
# 0.44 against a pool-wide 0.996), then jury-labelled. On those rows the model
# disagreed with the jury 55.4% of the time, so each one carries far more signal
# than a random prompt where it is already right ~87% of the time.
#
# d1 isolates the active data alone; d2 adds it on top of everything else. The
# comparison matters: if d1 (smaller, harder corpus) beats d2 (everything), the
# random rows are diluting rather than helping.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2

run () {
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py complexity --arch head --base "$MINILM" \
      --train "$2" --epochs 6 --lr 5e-5 --maxlen 256 --soft --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
}

BASE="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl"
ACT="$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl"

run "cx-d1-active-only"  "$D/complexity-v2.jsonl,$ACT"
run "cx-d2-everything"   "$BASE,$ACT"
echo "ROUND D COMPLETE"
