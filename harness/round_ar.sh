#!/usr/bin/env bash
# Round AR: is prior matching a real gain, or is it fitting the eval's SAMPLING?
#
# §111's prior matching won 9/9 matched-containment cells on sensitivity, so the
# obvious move is to apply it to the four gates -- all of which are mismatched in
# the same direction, by 1.26x to 2.81x.
#
# But the gates' training corpora come from the SAME real traffic their evals were
# drawn from, so train% is approximately traffic%. That makes the EVALS the
# enriched thing: build_eval_real.py screens and samples to get enough minority
# rows to measure. §99 caught this once already -- the route split is ~94/6 in
# live traffic and 80/20 in its gold. "Matching the eval prior" on these gates
# means fitting the eval's sampling design, not the deployment.
#
# §111 already showed the shape of that failure: +0.70 on the eval it was matched
# to, -2.81 on the other one.
#
# genlen is the only gate with TWO independent evals, so it is the only place the
# failure is detectable. Match the prior to real-gold-v2 and score BOTH. A gain on
# the matched eval with a loss on volume-gold means the technique fits eval
# construction; a gain on both means it is real.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
echo "########## gl-ar-prior-sqrt ##########"
./harness/runarm.sh gl-ar-prior-sqrt ./.venv/bin/python harness/train.py genlen --arch head \
    --base sentence-transformers/all-MiniLM-L6-v2 --train "$D/genlen-prior-sqrt.jsonl" \
    --epochs 3 --lr 5e-5 --maxlen 256 --soft --seed 11 --tag gl-ar-prior-sqrt
echo "ROUND AR COMPLETE"
