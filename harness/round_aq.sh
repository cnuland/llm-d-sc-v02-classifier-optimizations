#!/usr/bin/env bash
# Round AQ: push the prior-matching result further, and transfer it.
#
# §111: interpolating the training prior toward the eval prior gave +0.70 on
# entsec AND won all nine matched-containment cells -- the only intervention this
# control has confirmed rather than overturned. Two immediate questions it raises.
#
# aq1  Does the EXACT eval prior beat the sqrt interpolation? This is the tension
#      §78 wrote down and never resolved: exact matching aligns best but starves
#      CONFIDENTIAL from 14.4% to 5.8% of the corpus, and CONFIDENTIAL is the
#      class §57 showed the model already cannot learn. Whichever wins says which
#      pressure dominates.
#
# aq2  Does it transfer to the EGRESS gate? That is the published security model,
#      and its prior is mismatched in the same direction -- BLOCK is 12.5% of
#      training against 17.1% of the eval. If prior matching is a general property
#      of this eval family rather than a quirk of the 5-tier task, it should help
#      here too.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
echo "########## se-aq1-prior-exact ##########"
./harness/runarm.sh se-aq1-prior-exact ./.venv/bin/python harness/train.py sensitivity --arch head \
    --base BAAI/bge-base-en-v1.5 --train "$D/sensitivity-prior-eval.jsonl" \
    --epochs 3 --lr 3e-5 --maxlen 256 --soft --escalate 0.0 --seed 11 --tag se-aq1-prior-exact
echo "########## eg-aq2-prior-sqrt ##########"
./harness/runarm.sh eg-aq2-prior-sqrt ./.venv/bin/python harness/train.py egress --arch head \
    --base sentence-transformers/all-MiniLM-L6-v2 --train "$D/egress-prior-sqrt.jsonl" \
    --epochs 4 --lr 5e-5 --maxlen 256 --soft --classweight --seed 11 --tag eg-aq2-prior-sqrt
echo "ROUND AQ COMPLETE"
