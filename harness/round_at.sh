#!/usr/bin/env bash
# Round AT: the best folds nobody picked.
#
# §116 enumerated every contiguous fold of each ordered taxonomy and ranked them
# by jury agreement. Two beat the hand-picked gates and were never built:
#
#   complexity  SIMPLE | MEDIUM+COMPLEX+REASONING   86.9% agreement, 19.0% minority
#               against the route fold's             82.0%          , 24.6%
#   cost        MINIMAL | LOW+MODERATE+HIGH          89.6%          , 18.7%
#               against genlen's                     88.1%          , 49.6%
#
# Both are real routing decisions rather than metric conveniences: "is this
# trivially simple, or does it need actual work?" triages a cheap or cached path
# against the main model, and "one-liner or composed?" is the same question for
# generation budget.
#
# §98's fit (model = 0.36 x agreement + 61.7) predicts about +1.7 for triage over
# route. Recorded before running.
#
# CAVEAT ON genlen: its 49.6% minority share is the highest of any fold in this
# project, which is why §88 called it the best-evidenced gate. `oneliner` trades
# that away for +1.5 agreement, so even a win on accuracy is a weaker result.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
M=sentence-transformers/all-MiniLM-L6-v2
for S in 11 22; do
  echo "########## tri-at1-seed$S ##########"
  ./harness/runarm.sh "tri-at1-seed$S" ./.venv/bin/python harness/train.py triage --arch head \
      --base "$M" --train "$D/triage-v2.jsonl,$D/triage-real.jsonl,$D/triage-active.jsonl,$D/triage-distill.jsonl,$D/triage-real-contested.jsonl" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed $S --tag "tri-at1-seed$S"
done
echo "########## onl-at2-seed11 ##########"
./harness/runarm.sh onl-at2-seed11 ./.venv/bin/python harness/train.py oneliner --arch head \
    --base "$M" --train "$D/oneliner-v2.jsonl,$D/oneliner-real.jsonl,$D/oneliner-real-contested.jsonl,$D/oneliner-distill.jsonl" \
    --epochs 3 --lr 5e-5 --maxlen 256 --soft --classweight --seed 11 --tag onl-at2-seed11
echo "ROUND AT COMPLETE"
