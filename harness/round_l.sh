#!/usr/bin/env bash
# Round L: ensemble distillation for complexity.
#
# l1 mixes 30,000 ensemble-pseudo-labelled rows with the real corpus. The
# pseudo-labels carry the teacher's full distribution, so boundary cases train
# toward a spread rather than a coin-flip.
#
# l2 is the control: the identical corpus WITHOUT the distilled rows. Without it
# any gain could be data volume rather than distillation, and this project has
# already been caught attributing a gain to the wrong cause once (§28).
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
BASE="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl"

run () {
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py complexity --arch head --base "$MINILM" \
      --train "$2" --epochs 4 --lr 5e-5 --maxlen 256 --soft --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency" || echo "  FAILED"
}

run "cx-l1-distill" "$BASE,$D/complexity-distill.jsonl"
run "cx-l2-control" "$BASE"
echo "ROUND L COMPLETE"
