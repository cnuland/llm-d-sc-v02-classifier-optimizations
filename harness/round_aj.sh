#!/usr/bin/env bash
# Round AJ: does a DENOISED, COHERENT training target beat panel consensus?
#
# The untested cell from §83/§85. §83 found a juror is 27.7 points more
# consistent with itself than the panel is with itself, so the gold carries
# intra-juror sampling noise. §85 found that swapping only the EVAL to a
# self-consistency gold LOWERS accuracy, because the model reproduces whatever it
# was trained on. So the experiment has to change the TRAINING target.
#
# Both arms use the SAME 20,000 prompts. Only the labelling process differs:
#   aj1  panel labels        two jurors, one sample each, agreed rows
#   aj2  self-consistency    one juror, three samples, majority (92.9% stable)
#
# Both are scored against BOTH golds, because "reproduces this rater" and
# "reproduces the consensus" are different claims and each arm is favoured by one
# of them. The interesting number is aj2-on-sc-gold versus aj1-on-panel-gold:
# each model measured against the target it was trained for.
#
# If denoising the labels does not move that pair, then label noise was never the
# binding constraint on complexity, and the label axis is closed for good --
# which is the more useful outcome, because three attempts have now failed inside
# the noise floor (§53 +0.27, §65/AB +0.13, §85's clean-subset analysis).
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
for S in 11 22; do
  for arm in "aj1-panel:$D/complexity-panel20k.jsonl" "aj2-selfcons:$D/complexity-selfconsistency.jsonl"; do
    T="cx-${arm%%:*}-seed$S"; F="${arm#*:}"
    echo "########## $T ##########"
    ./harness/runarm.sh "$T" ./.venv/bin/python harness/train.py complexity --arch head \
        --base "$MINILM" --train "$F" --epochs 3 --lr 5e-5 --maxlen 256 --soft \
        --seed $S --tag "$T"
  done
done
echo "ROUND AJ COMPLETE"
