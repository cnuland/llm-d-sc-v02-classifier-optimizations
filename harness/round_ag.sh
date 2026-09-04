#!/usr/bin/env bash
# Round AG: which PRIOR, not whether to balance.
#
# Round X's second arm settled the question X was asked to settle, in the
# opposite direction to §57's prediction:
#   bge, raw corpus        entsec 0.8119   real 0.8838
#   bge, uniform balance   entsec 0.7808   real 0.8944    -3.11 on entsec
#
# §56 explains it. Uniform balancing cut INTERNAL from 16,450 to 7,694 while
# INTERNAL is 51.2% of the entsec eval, so it moved the training prior AWAY from
# the evaluation prior -- and §56 had already measured that mismatch as worth
# about a point through logit adjustment alone. Uniform is not neutral; it is a
# specific and, here, wrong choice of prior.
#
# Three arms on the SAME encoder and schedule, varying only the prior:
#   ag1  natural corpus            (control, = x1's condition on MiniLM)
#   ag2  sqrt interpolation        CONFIDENTIAL 14.4% -> 9.4%
#   ag3  exact eval prior          CONFIDENTIAL 14.4% -> 5.8%
#
# The tension is deliberate. ag3 matches best but starves CONFIDENTIAL -- the
# class §57 showed the model already cannot learn -- down to 1,861 rows. If ag3
# beats ag2 anyway, prior match dominates class starvation; if it loses, the
# minority class needs protecting and sqrt is the compromise that does it.
#
# NOTE this is entsec-specific by construction. §77 measured entsec as 95.8%
# distinguishable from real traffic, so "matching the eval prior" tunes to a
# synthetic enterprise distribution. real-gold is reported on every arm to show
# what that costs.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
NAT="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
run () {
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" --train "$2" \
      --epochs 4 --lr 5e-5 --maxlen 256 --soft --escalate 0.0 --seed 11 \
      --tag "$1" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
}
run se-ag1-natural     "$NAT"
run se-ag2-prior-sqrt  "$D/sensitivity-prior-sqrt.jsonl"
run se-ag3-prior-eval  "$D/sensitivity-prior-eval.jsonl"
echo "ROUND AG COMPLETE"
