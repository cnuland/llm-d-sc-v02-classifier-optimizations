#!/usr/bin/env bash
# Round G: techniques taken from the vLLM Semantic Router training code that
# this project had not tried, applied to COST -- the signal with the most
# headroom (+10.4 points to its 0.9350 ceiling).
#
#   wd 0.1 + cosine + warmup 0.06   vSR's regularisation recipe, explicitly
#                                   commented there as "higher weight decay for
#                                   better regularisation"
#   LoRA r=16                       a different finetuning mechanism entirely:
#                                   only the adapter trains, which is a far
#                                   stronger regulariser than full fine-tuning.
#                                   Cost's gold carries 6.5% label noise, and
#                                   full fine-tuning has the capacity to
#                                   memorise noise that LoRA does not.
#   ordinal                         cost's tiers are genuinely monotone
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
CO="$D/cost-v2.jsonl,$D/cost-real.jsonl,$D/cost-real-contested.jsonl"

run () {  # run <tag> <extra flags>
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py cost --arch head --base "$MINILM" --train "$CO" \
      --epochs 6 --lr 5e-5 --maxlen 256 --soft $2 --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency|trainable|NOTE" || echo "  FAILED"
}

run "co-g1-vsr-reg"      "--wd 0.1 --sched cosine --warmup 0.06"
run "co-g2-vsr-reg-ord"  "--wd 0.1 --sched cosine --warmup 0.06 --ordinal"
run "co-g3-lora16"       "--lora 16 --lr 2e-4 --wd 0.1 --sched cosine --warmup 0.06"
run "co-g4-lora16-ord"   "--lora 16 --lr 2e-4 --wd 0.1 --sched cosine --warmup 0.06 --ordinal"
echo "ROUND G COMPLETE"
