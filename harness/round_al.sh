#!/usr/bin/env bash
# Round AL: apply the encoder finding to the four deployed gates.
#
# All four gates were trained on all-MiniLM-L6-v2 because that is what every
# recipe in this project defaulted to. But §80 measured bge-base at +4.53 over
# MiniLM on sensitivity at fixed schedule and corpus, and §87 measured bge-small
# recovering ALL of that gain at 45% of the latency. That axis has never been
# tried on the gates -- they were built before the encoder result existed.
#
# bge-small rather than bge-base: §87's ladder makes it the best accuracy per ms,
# and these gates run on the CPU serving path where MiniLM is 8.38 ms and
# bge-base is 31.20 ms (§86, measured in one interleaved process -- the per-run
# latency figures elsewhere in this report were contaminated by concurrent load).
#
# Ordered weakest-first, so if the queue is cut short the most valuable arm has
# already run: route is the lowest-scoring gate at 93.35% AND the one that fires
# on every request.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
BGES=BAAI/bge-small-en-v1.5
run () {  # signal tag train-files extra-flags
  echo "########## $2 ##########"
  ./harness/runarm.sh "$2" ./.venv/bin/python harness/train.py "$1" --arch head \
      --base "$BGES" --train "$3" --epochs 3 --lr 5e-5 --maxlen 256 --soft \
      --classweight --seed 11 --tag "$2"
}
run route     rt-al-bgesmall  "$D/route-v2.jsonl,$D/route-real.jsonl,$D/route-active.jsonl,$D/route-distill.jsonl,$D/route-real-contested.jsonl"
run reasoning rs-al-bgesmall  "$D/reasoning-v2.jsonl,$D/reasoning-real.jsonl,$D/reasoning-active.jsonl,$D/reasoning-distill.jsonl,$D/reasoning-real-contested.jsonl"
run genlen    gl-al-bgesmall  "$D/genlen-v2.jsonl,$D/genlen-real.jsonl,$D/genlen-real-contested.jsonl,$D/genlen-distill.jsonl"
run egress    eg-al-bgesmall  "$D/egress-v2.jsonl,$D/egress-real.jsonl,$D/egress-enterprise.jsonl,$D/egress-real-contested.jsonl"
echo "ROUND AL COMPLETE"
