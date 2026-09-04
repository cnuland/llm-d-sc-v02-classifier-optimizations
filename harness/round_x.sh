#!/usr/bin/env bash
# Round X: was §50's "bigger is worse on sensitivity" an artifact of class skew?
#
# §57 (frozen probe, that pair only, balanced, no fine-tuning) found bge-base and
# e5-base carry +8.3 points of balanced accuracy and +14.5 points of CONFIDENTIAL
# recall over the shipped MiniLM. Same data, same classifier, same balance --
# only the representation differs, so the distinction IS representable and MiniLM
# is what cannot hold it.
#
# §50 concluded the opposite from FINE-TUNED big encoders on the raw 5-way corpus,
# where it observed "INTERNAL recall ROSE to 0.86 while CONFIDENTIAL collapsed
# from 0.84 to 0.56" -- capacity spent on the majority class. Encoder size and
# class skew were varied together, so neither was isolated.
#
# Three arms separate them. x1 reproduces §50's condition with the current recipe;
# x2 changes ONLY the corpus balance; x3 is the MiniLM control on the same balanced
# corpus, so any x2 gain can be attributed to the encoder rather than to balancing.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
BGE=BAAI/bge-base-en-v1.5
RAW="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
BAL="$D/sensitivity-balanced.jsonl"

run () {  # tag base train
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$2" --train "$3" \
      --epochs 3 --lr 3e-5 --maxlen 256 --soft --escalate 1.0 --seed 11 \
      --tag "$1" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
}
run se-x1-bge-raw  "$BGE"    "$RAW"
run se-x2-bge-bal  "$BGE"    "$BAL"
run se-x3-mini-bal "$MINILM" "$BAL"
echo "ROUND X COMPLETE"
