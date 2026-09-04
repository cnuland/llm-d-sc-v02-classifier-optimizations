#!/usr/bin/env bash
# Round R: the decisive data test — cost corpus 6.5x larger.
#
# Every mechanism, architecture, acquisition strategy and calibration approach
# tried in this project sits at or inside the measured noise floor (cost: 1.35
# pts). Jury-labelled real data is the ONLY lever that has ever cleared it by
# more than 2x. The corpus went 6,697 -> 43,717 agreed rows (plus 15,805
# contested, usable as soft targets), so this is that lever at full stretch.
#
# Three seeds, because a single run cannot resolve anything under 1.35 points
# and the whole point is to know whether the gain is real.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
CO="$D/cost-v2.jsonl,$D/cost-real.jsonl,$D/cost-real-contested.jsonl,$D/cost-distill.jsonl"
for S in 11 22 33; do
  echo "########## co-r-big-seed$S ##########"
  ./.venv/bin/python harness/train.py cost --arch head --base "$MINILM" --train "$CO" \
      --epochs 3 --lr 5e-5 --maxlen 256 --soft --seed $S --tag "co-r-big-seed$S" 2>&1 \
    | grep -E "^\[|refined-gold|real-gold|trained in" || echo "  FAILED"
done
echo "ROUND R COMPLETE"
