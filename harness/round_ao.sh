#!/usr/bin/env bash
# Round AO: the two untested mechanisms on the ONE signal with headroom left.
#
# §98's residual analysis is the reason this round is sensitivity-only. Six of
# seven taxonomies score within 2.5 points of what their jury agreement predicts;
# sensitivity is the single outlier at -6.9, scoring BELOW its ceiling. That
# matches §55's independent diagnosis that it is the only model-limited signal
# here -- it misses 156 rows where all three jurors agreed. Everywhere else the
# labels bind and mechanism work cannot help. Here it can.
#
# ao1/ao2  hierarchy-aware supervised contrastive loss, never once measured. Every
#          previous arm died to an eval-time OOM in my own integration
#          (output_hidden_states was requested during evaluation, so Trainer
#          accumulated 13 layers per eval batch on the accelerator). Fixed.
#          SupCon acts on representation GEOMETRY, which is precisely what a
#          model-limited signal needs and what reweighting the loss cannot touch.
#
# ao3      prior matching. §78 showed uniform class balancing COSTS 3.11 points
#          because it moves the training prior away from the eval prior. The
#          sqrt-interpolated corpus was built and never run.
#
# All on the current best recipe: bge-base + escalate 0.0 (§90, entsec 0.8204).
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
BGE=BAAI/bge-base-en-v1.5
NAT="$D/sensitivity-v2.jsonl,$D/sensitivity-real.jsonl,$D/sensitivity-enterprise.jsonl,$D/sensitivity-real-contested.jsonl"
run () {  # tag corpus supcon
  echo "########## $1 ##########"
  ./harness/runarm.sh "$1" ./.venv/bin/python harness/train.py sensitivity --arch head \
      --base "$BGE" --train "$2" --epochs 3 --lr 3e-5 --maxlen 256 --soft \
      --escalate 0.0 --supcon "$3" --seed 11 --tag "$1"
}
run se-ao1-supcon0.1  "$NAT" 0.1
run se-ao2-supcon0.3  "$NAT" 0.3
run se-ao3-priorsqrt  "$D/sensitivity-prior-sqrt.jsonl" 0.0
echo "ROUND AO COMPLETE"
