#!/usr/bin/env bash
# Round M: specialist binary arbiters for the two boundaries that carry the errors.
#
# 78% of complexity errors are adjacent-tier, and they concentrate on two pairs:
# SIMPLE<->MEDIUM (26 of 48) and MEDIUM<->COMPLEX (19 of 48). A single 4-way
# softmax has to carve all three boundaries with one representation. A binary
# classifier trained ONLY on one contested pair sees a much easier problem and
# can spend its whole capacity on it.
#
# Used as an ARBITER, not a replacement: the 4-way model decides, and when it
# lands on one side of a contested pair the specialist re-decides between just
# those two. That keeps the 4-way model's coverage and only intervenes where it
# is known to be weak.
set -u
cd /Users/cnuland/llm-d-sc-accuracy
export HF_TOKEN=$(cat ~/hf_token.txt | tr -d '\n'); export TOKENIZERS_PARALLELISM=false
D=data/train
MINILM=sentence-transformers/all-MiniLM-L6-v2

# Build two-class corpora from the existing labelled data.
./.venv/bin/python - <<'PY'
import json, sys; sys.path.insert(0,"harness")
from evalkit import ROOT, load_jsonl
srcs = ["complexity-v2","complexity-real","complexity-real-contested",
        "complexity-active","complexity-active-contested"]
rows = []
for s in srcs:
    p = ROOT/f"data/train/{s}.jsonl"
    if p.exists(): rows += load_jsonl(p)
for a, b in [("SIMPLE","MEDIUM"), ("MEDIUM","COMPLEX")]:
    sel = [r for r in rows if r["tier"] in (a, b)]
    out = ROOT/f"data/train/complexity-pair-{a}-{b}.jsonl"
    with open(out,"w") as fh:
        for r in sel: fh.write(json.dumps(r)+"\n")
    print(f"  pair {a}/{b}: {len(sel)} rows -> {out.name}")
PY

run () {
  echo "########## $1 ##########"
  ./.venv/bin/python harness/train.py complexity --arch head --base "$MINILM" \
      --train "$2" --epochs 5 --lr 5e-5 --maxlen 256 --soft --tag "$1" 2>&1 \
    | grep -E "^\[|trained in|Traceback|Error" | head -4 || echo "  FAILED"
}
run "cx-m1-pair-SIMPLE-MEDIUM"  "$D/complexity-pair-SIMPLE-MEDIUM.jsonl"
run "cx-m2-pair-MEDIUM-COMPLEX" "$D/complexity-pair-MEDIUM-COMPLEX.jsonl"
echo "ROUND M COMPLETE"
