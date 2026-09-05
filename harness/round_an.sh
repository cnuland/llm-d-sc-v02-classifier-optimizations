#!/usr/bin/env bash
# Round AN: does label denoising pay MORE on the collapsed taxonomy?
#
# Round AJ tested self-consistency labels against panel labels on the 4-tier
# taxonomy and got +0.80 -- inside the 1.06-point floor. The decomposition above
# says why it might differ here: on the 2-tier fold, 9.8 of the 18.0 points of
# residual jury disagreement is a single juror disagreeing with ITSELF, which is
# exactly what majority-of-3 voting removes.
#
# PREDICTION ON RECORD, and it is the opposite of the hopeful one: the fold
# already merges away the boundary where self-consistency is least stable
# (MEDIUM/COMPLEX), so the two labellings should agree MORE once folded and the
# gap between them should SHRINK, not grow. If that holds, the fold has already
# captured most of the denoising benefit and there is nothing left to buy.
#
# No new labelling -- AJ's two 20k corpora, folded. Only the taxonomy changes.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
for S in 11 22; do
  for arm in "an1-panel:$D/cx2-aj-panel20k.jsonl" "an2-selfcons:$D/cx2-aj-selfcons.jsonl"; do
    T="cx2-${arm%%:*}-seed$S"; F="${arm#*:}"
    echo "########## $T ##########"
    ./harness/runarm.sh "$T" ./.venv/bin/python harness/train.py cx2 --arch head \
        --base sentence-transformers/all-MiniLM-L6-v2 --train "$F" \
        --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed $S --tag "$T"
  done
done
echo "ROUND AN COMPLETE"
