#!/usr/bin/env bash
# Round AP: the last untested data-side lever on the one signal with headroom.
#
# §64 measured every sensitivity training file's separability from the entsec
# eval, and §77 confirmed the metric is a domain-match measure rather than a
# circular one:
#
#   sensitivity-enterprise.jsonl   31,168 rows   78.1% separable
#   sensitivity-distill.jsonl       8,995 rows   81.7%
#   sensitivity-v2.jsonl            6,783 rows   92.8%
#   sensitivity-real.jsonl          8,361 rows   98.2%   <- WildChat
#
# A THIRD of the corpus comes from files a linear probe tells apart from the eval
# 93-98% of the time, and §63/§76 showed off-distribution rows can be actively
# worse than no rows. Corpus composition here is an accident of which generators
# happened to get run; it has never been ablated.
#
# BOTH evals on every arm. Dropping WildChat should help entsec and hurt
# real-gold, and an arm reporting only entsec would look like a free win. The
# question is the exchange rate, not the entsec number.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
BGE=BAAI/bge-base-en-v1.5
run () {
  echo "########## $1 ##########"
  ./harness/runarm.sh "$1" ./.venv/bin/python harness/train.py sensitivity --arch head \
      --base "$BGE" --train "$2" --epochs 3 --lr 3e-5 --maxlen 256 --soft \
      --escalate 0.0 --seed 11 --tag "$1"
}
run se-ap1-noreal   "$D/sensitivity-v2.jsonl,$D/sensitivity-enterprise.jsonl"
run se-ap2-entonly  "$D/sensitivity-enterprise.jsonl"
run se-ap3-entdist  "$D/sensitivity-enterprise.jsonl,$D/sensitivity-distill.jsonl"
echo "ROUND AP COMPLETE"
