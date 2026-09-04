#!/usr/bin/env bash
# Round H: the FIRST run in which soft targets and ordinal smoothing actually
# reach the loss function.
#
# Every prior --soft / --ordinal arm silently trained on hard labels: HF Trainer
# defaults remove_unused_columns=True and dropped the `soft` column before
# collation. So the +1.4 previously credited to soft targets was really the
# 2,358 contested ROWS being added to the corpus, and the "ordinal had no
# effect" result was the flag never applying.
#
# Paired arms so the two can finally be separated: same corpus, same seed, one
# with soft targets and one without.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2

run () {  # run <tag> <signal> <corpora> <flags>
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py "$2" --arch head --base "$MINILM" --train "$3" \
      --epochs 6 --lr 5e-5 --maxlen 256 $4 --tag "$1" 2>&1 \
    | grep -E "^\[|acc=|trained in|latency|NOTE|soft targets requested" || echo "  FAILED"
}

CX="$D/complexity-v2.jsonl,$D/complexity-real.jsonl,$D/complexity-real-contested.jsonl,$D/complexity-active.jsonl,$D/complexity-active-contested.jsonl,$D/complexity-disagree.jsonl,$D/complexity-disagree-contested.jsonl"
CO="$D/cost-v2.jsonl,$D/cost-real.jsonl,$D/cost-real-contested.jsonl"
SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"

run "cx-h1-hard"      complexity  "$CX" ""
run "cx-h2-soft"      complexity  "$CX" "--soft"
run "co-h1-hard"      cost        "$CO" ""
run "co-h2-soft"      cost        "$CO" "--soft"
run "co-h3-soft-ord"  cost        "$CO" "--soft --ordinal"
run "se-h1-soft"      sensitivity "$SE" "--soft"
run "se-h2-soft-ord"  sensitivity "$SE" "--soft --ordinal"
# Target the measured failure directly: REGULATED recall 0.46-0.51 and
# NEVER_EGRESS 0.69-0.85 on the enterprise-secrets eval, with 0.78 accuracy
# carried by INTERNAL at 0.94. Symmetric cross-entropy cannot express that
# letting a credential through is worse than over-flagging a public prompt.
run "se-h3-cw"        sensitivity "$SE" "--soft --ordinal --classweight"
run "se-h4-cw-esc"    sensitivity "$SE" "--soft --ordinal --escalate 1.0"
echo "ROUND H COMPLETE"
