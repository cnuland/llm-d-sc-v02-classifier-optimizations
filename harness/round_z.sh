#!/usr/bin/env bash
# Round Z: hierarchy-aware supervised contrastive loss.
#
# Every intervention tried on sensitivity so far acts on the DECISION RULE --
# class weights (§55), logit adjustment (§56), span-max (§60), thresholds. Two of
# them landed on the same 1:2 accuracy/containment exchange rate, which is the
# signature of sliding along one fixed ROC curve rather than moving it. §57 said
# why: the failure is where INTERNAL and CONFIDENTIAL SIT in representation
# space, and no rule over those coordinates can fix the coordinates.
#
# SupCon (Khosla et al. 2020) acts on the coordinates. The variant here is
# hierarchy-aware, after Learning Label Hierarchy with Supervised Contrastive
# Learning (Findings of EACL 2024): pair weight decays with tier distance,
# w = 1 - |rank_i - rank_j|/(K-1), so the ladder's ordering is in the objective
# instead of being flattened into same/different.
#
# Two lambdas, and BOTH escalate settings, because Round W is concurrently
# showing escalate 0.0 ahead of the incumbent 1.0 and confounding the two would
# make either result unreadable. Seed 11 first; seeds are added only to arms
# that clear sensitivity's 0.14-point floor.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2
SE="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
for E in 0.0 1.0; do
  for LAM in 0.1 0.3; do
    T="se-z-supcon${LAM}-esc${E}"
    echo "########## $T ##########"
    ./.venv/bin/python harness/train.py sensitivity --arch head --base "$MINILM" --train "$SE" \
        --epochs 4 --lr 5e-5 --maxlen 256 --soft --escalate "$E" --supcon "$LAM" \
        --seed 11 --tag "$T" 2>&1 | grep -E "^\[|entsec-gold|real-gold|trained in" || echo "  FAILED"
  done
done
echo "ROUND Z COMPLETE"
