#!/usr/bin/env bash
# Round AV: a middle tier for triage, because abstention cannot save it.
#
# §121 found triage's confidence is BLIND to jury disagreement -- its least
# confident 5% contains contested rows at exactly the 31.9% base rate, against
# 1.7-1.9x enrichment for every other gate. It is confidently wrong on hard rows,
# not uncertain about them, which is why §118 measured it with the worst
# abstention curve of any gate despite the second-highest accuracy.
#
# §119 hit this exact failure shape on the binary egress gate and the fix was not
# a threshold -- it was a middle tier, giving uncertain rows somewhere to GO. That
# took secrets-released from 14.88% to 0.00% for 0.99 points of accuracy.
#
# Same fix, same signal family: TRIVIAL / STANDARD / HARD instead of TRIVIAL /
# WORK. The prediction to check is not accuracy -- it is whether the three-way
# version stops being confidently wrong, i.e. whether ITS abstention curve
# enriches for contested rows.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
for S in 11 22; do
  echo "########## t3cx-av-seed$S ##########"
  ./harness/runarm.sh "t3cx-av-seed$S" ./.venv/bin/python harness/train.py triage3cx --arch head \
      --base sentence-transformers/all-MiniLM-L6-v2 \
      --train "$D/triage3cx-v2.jsonl,$D/triage3cx-real.jsonl,$D/triage3cx-active.jsonl,$D/triage3cx-distill.jsonl,$D/triage3cx-real-contested.jsonl" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed $S --tag "t3cx-av-seed$S"
done
echo "ROUND AV COMPLETE"
