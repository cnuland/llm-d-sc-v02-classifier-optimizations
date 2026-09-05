#!/usr/bin/env bash
# Round AU: sensitivity as ALLOW / REVIEW / BLOCK.
#
# §116's fold search ranked PUBLIC | INTERNAL+CONFIDENTIAL+REGULATED |
# NEVER_EGRESS third on sensitivity at 89.5% agreement -- and it is the only
# high-agreement fold on any signal that is NOT binary. It is also the shape a
# DLP deployment actually wants: send it, hold it for review, or stop it.
#
# The binary egress gate (96.4% agreement) is higher, but it collapses everything
# below NEVER_EGRESS into ALLOW, which means CONFIDENTIAL and REGULATED content
# passes with no signal at all. This fold keeps that distinction at a cost of 6.9
# points of agreement.
#
# §117's rule applies: agreement generates the candidate, balance decides whether
# it counts. Checked above before training.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
for S in 11 22; do
  echo "########## t3-au-seed$S ##########"
  ./harness/runarm.sh "t3-au-seed$S" ./.venv/bin/python harness/train.py triage3 --arch head \
      --base BAAI/bge-base-en-v1.5 \
      --train "$D/triage3-v2.jsonl,$D/triage3-real.jsonl,$D/triage3-enterprise.jsonl,$D/triage3-real-contested.jsonl" \
      --epochs 3 --lr 3e-5 --maxlen 256 --soft --escalate 0.0 --seed $S --tag "t3-au-seed$S"
done
echo "ROUND AU COMPLETE"
