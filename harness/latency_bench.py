"""Comparative CPU latency, all models in ONE process, back to back.

Why this exists: `bge-small` reported p50 24.76 ms and `bge-base` reported
21.9 ms, which is impossible on parameter count -- 33M against 109M at the same
depth. The two numbers were measured by different training runs at different
times, under different amounts of concurrent load from other trainers on the same
machine. **Every per-run latency figure in this report is contaminated the same
way**, including the 6.1 ms vs 21.9 ms comparison §68's deployment
recommendation leans on.

Measuring them together fixes the comparison even if the absolute numbers still
carry whatever load is present: all models see the same conditions, in
interleaved rounds so a drift in load hits every model equally rather than
whichever one ran last.
"""
import sys, time, statistics
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import torch
from evalkit import ROOT, load_jsonl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

TAGS = sys.argv[1:] or ["se-w-esc0.0-seed22", "se-ah-bgesmall-seed11", "se-x1-bge-raw"]
texts = [r["text"] for r in load_jsonl(ROOT/"data/eval/sensitivity-entsec-gold.jsonl")][:120]
torch.set_num_threads(1)          # per-request latency, not batch throughput

M = {}
for t in TAGS:
    d = ROOT/"models"/t
    tok = AutoTokenizer.from_pretrained(str(d))
    m = AutoModelForSequenceClassification.from_pretrained(str(d)).eval().to("cpu")
    M[t] = (tok, m, sum(p.numel() for p in m.parameters()))
    with torch.no_grad():                                   # warm up
        for x in texts[:5]: m(**tok(x, truncation=True, max_length=256, return_tensors="pt"))

times = {t: [] for t in TAGS}
for i, x in enumerate(texts):                               # interleaved rounds
    for t in TAGS:
        tok, m, _ = M[t]
        e = tok(x, truncation=True, max_length=256, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad(): m(**e)
        times[t].append((time.perf_counter()-t0)*1000)

print(f"{'model':<26}{'params':>10}{'p50 ms':>9}{'p90 ms':>9}{'p99 ms':>9}{'vs first':>10}")
base = None
for t in TAGS:
    v = sorted(times[t]); n = len(v)
    p50, p90, p99 = v[n//2], v[int(n*0.9)], v[int(n*0.99)]
    if base is None: base = p50
    print(f"{t:<26}{M[t][2]/1e6:9.1f}M{p50:9.2f}{p90:9.2f}{p99:9.2f}{p50/base:9.2f}x")
