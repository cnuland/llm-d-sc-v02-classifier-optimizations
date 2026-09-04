#!/usr/bin/env bash
# Remaining work, serialised so nothing contends for the GPU.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
# wait out whatever is training now
while pgrep -f 'harness/[t]rain.py' >/dev/null; do sleep 30; done

# 1. Re-run the embed arms now that anchors are re-selected in the fine-tuned
#    space. Those arms scored against stale hand-written anchors, which is why
#    sensitivity embed/real landed at 0.2958 on a set where always answering
#    PUBLIC scores 0.85.
for SIG in sensitivity complexity; do
  D=data/train; MIX="$D/$SIG-v2.jsonl,$D/$SIG-real.jsonl"
  [ -f "$D/$SIG-enterprise.jsonl" ] && MIX="$MIX,$D/$SIG-enterprise.jsonl"
  echo "########## $SIG-embed-minilm-mix-reanchor ##########"
  ./.venv/bin/python harness/train.py "$SIG" --arch embed \
    --base sentence-transformers/all-MiniLM-L6-v2 --train "$MIX" --epochs 3 \
    --tag "$SIG-embed-minilm-mix-reanchor" 2>&1 | grep -E "^\[|acc=|trained|latency"
done

# 2. cost, which now has its real-traffic corpus
echo "########## cost matrix ##########"
./harness/run_matrix.sh cost 3

# 3. push the winning architecture on the two priority signals
./harness/sweep.sh complexity
./harness/sweep.sh sensitivity
echo "QUEUE COMPLETE"
