#!/usr/bin/env bash
# Round AI: stack the two independent sensitivity wins.
#
# Two effects were established separately and never combined:
#   §66  removing tier escalation      MiniLM 0.7808 -> 0.7999 median  (+1.91)
#   §80  bge-base instead of MiniLM    +4.53 at fixed schedule and corpus
#
# They act on different parts of the system -- one on the loss, one on the
# representation -- which is the only reason stacking is worth a run rather than
# an assumption. bge at escalate 1.0 already scored 0.8119; if the escalation
# effect carries over, this lands near 0.83.
#
# Round AC's first arm (MiniLM/natural/3ep) is dropped: Round X's factorial
# already attributed the encoder gain at fixed schedule and corpus, so it is no
# longer load-bearing.
#
# Uses runarm.sh, so a crash prints its real traceback instead of vanishing (§81).
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
NAT="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
for S in 11 22; do
  T="se-ai-bge-esc0-seed$S"
  echo "########## $T ##########"
  ./harness/runarm.sh "$T" ./.venv/bin/python harness/train.py sensitivity --arch head \
      --base BAAI/bge-base-en-v1.5 --train "$NAT" \
      --epochs 3 --lr 3e-5 --maxlen 256 --soft --escalate 0.0 --seed $S --tag "$T"
done
echo "ROUND AI COMPLETE"
