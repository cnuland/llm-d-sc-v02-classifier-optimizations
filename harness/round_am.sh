#!/usr/bin/env bash
# Round AM: complexity as a TWO-TIER taxonomy, retrained.
#
# The 4-tier ladder has four boundaries and every one is a place two labellers
# can disagree; §74 measured model accuracy tracking jury agreement within a
# couple of points at every fold. Two tiers means one boundary, and it is the one
# the deployed Praxis table actually consumes (SIMPLE/MEDIUM -> small model,
# COMPLEX/REASONING -> large).
#
#   4-tier complexity     jury agreement 70.4%   model 0.8963
#   2-tier (this split)    jury agreement 82.0%   model 0.9335 folded, seed 11
#
# Three arms, best-known recipe from the rest of the project:
#   am1  MiniLM, 2 seeds        the deployable baseline at 8.4 ms
#   am2  bge-small, 1 seed      §87's encoder result, +1.2 on sensitivity at 1.66x
#
# The 4-tier model scored on the SAME folded eval is the comparison that matters;
# route-gate already provides it at 0.9335 / 0.9202. Anything here has to beat
# that to justify a second taxonomy rather than folding the one we have.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
CX2="$D/cx2-v2.jsonl,$D/cx2-real.jsonl,$D/cx2-active.jsonl,$D/cx2-distill.jsonl,$D/cx2-real-contested.jsonl"
for S in 11 22; do
  T="cx2-am1-mini-seed$S"
  echo "########## $T ##########"
  ./harness/runarm.sh "$T" ./.venv/bin/python harness/train.py cx2 --arch head \
      --base sentence-transformers/all-MiniLM-L6-v2 --train "$CX2" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed $S --tag "$T"
done
echo "########## cx2-am2-bgesmall ##########"
./harness/runarm.sh cx2-am2-bgesmall ./.venv/bin/python harness/train.py cx2 --arch head \
    --base BAAI/bge-small-en-v1.5 --train "$CX2" \
    --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed 11 --tag cx2-am2-bgesmall
echo "ROUND AM COMPLETE"
