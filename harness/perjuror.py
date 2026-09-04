"""Per-annotator heads: model each JUROR, not the average of them.

§74 is the motivation. Model accuracy tracked jury agreement within 1.3-2.6
points at every fold of the sensitivity eval, which says the models have been
pinned to the decidability of the question. Soft targets (the current recipe)
already use disagreement, but they discard WHO disagreed: a 1-1 split becomes
0.5/0.5 whichever juror took which side.

The perspectivist literature argues that is the wrong reduction -- majority
voting and its soft-label cousin "obscure nuanced viewpoints" and lose
systematic per-annotator signal (Leveraging Annotator Disagreement for Text
Classification, arXiv 2409.17577; Beyond Consensus, arXiv 2601.09065). With a
shared encoder and one head per juror, the encoder is pushed to represent the
ITEM while each head absorbs its juror's systematic bias, and the biases cancel
at aggregation instead of being baked into the target.

This is the only mechanism tried in this project that has a route to beating the
agreement ceiling rather than tracking it, so it is worth the extra head.

Implemented standalone: train.py's eval path reads model.config.id2label, and a
J*L output layer would silently break it. If this wins, THEN fold it in.
"""
import sys, json, glob, collections, math, time
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from evalkit import ROOT, load_jsonl, load_taxonomy
from transformers import AutoTokenizer, AutoModel

SIG   = sys.argv[1] if len(sys.argv) > 1 else "complexity"
J     = 2                                   # juror slots present in training rows
EPOCHS= int(sys.argv[2]) if len(sys.argv) > 2 else 3
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 11
MODE  = sys.argv[4] if len(sys.argv) > 4 else "perjuror"   # or "soft" (control)
BASE  = "sentence-transformers/all-MiniLM-L6-v2"
LABELS= load_taxonomy(SIG)["labels"]
L     = len(LABELS)
idx   = {l: i for i, l in enumerate(LABELS)}
dev   = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

rows = []
for p in glob.glob(str(ROOT/"data/train"/f"{SIG}-*.jsonl")):
    if any(x in p for x in ("balanced","boundary","greyzone","pool","v2rubric")): continue
    for l in open(p):
        r = json.loads(l)
        t = r.get("tier") or r.get("label")
        if t not in idx: continue
        v = r.get("votes")
        v = [x for x in v if x in idx] if isinstance(v, list) else []
        # rows without a full vote list fall back to the adjudicated label for
        # every head, which is exactly what soft targets do for them anyway
        v = (v + [t]*J)[:J]
        rows.append((r["text"], [idx[x] for x in v], idx[t]))
rng = np.random.default_rng(SEED); rng.shuffle(rows)
print(f"{SIG}: {len(rows)} rows, {J} juror slots, mode={MODE}, device={dev}")

tok = AutoTokenizer.from_pretrained(BASE)

class DS(Dataset):
    def __init__(self, rs): self.rs = rs
    def __len__(self): return len(self.rs)
    def __getitem__(self, i):
        t, v, y = self.rs[i]
        e = tok(t, truncation=True, max_length=256)
        return e["input_ids"], v, y

def collate(b):
    mx = max(len(x[0]) for x in b)
    ids = torch.tensor([x[0] + [tok.pad_token_id]*(mx-len(x[0])) for x in b])
    att = torch.tensor([[1]*len(x[0]) + [0]*(mx-len(x[0])) for x in b])
    return ids, att, torch.tensor([x[1] for x in b]), torch.tensor([x[2] for x in b])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BASE)
        h = self.enc.config.hidden_size
        self.head = nn.Linear(h, L*J if MODE == "perjuror" else L)
    def forward(self, ids, att):
        o = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        z = (o * att.unsqueeze(-1)).sum(1) / att.sum(1, keepdim=True).clamp(min=1)
        g = self.head(z)
        return g.view(-1, J, L) if MODE == "perjuror" else g

net = Net().to(dev)
dl = DataLoader(DS(rows), batch_size=32, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW(net.parameters(), lr=5e-5)
steps = EPOCHS*len(dl)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-5, total_steps=steps, pct_start=0.1)
ce = nn.CrossEntropyLoss()
t0 = time.time()
for ep in range(EPOCHS):
    net.train(); run = 0.0
    for i, (ids, att, v, y) in enumerate(dl):
        ids, att, v, y = ids.to(dev), att.to(dev), v.to(dev), y.to(dev)
        g = net(ids, att)
        if MODE == "perjuror":
            loss = sum(ce(g[:, j, :], v[:, j]) for j in range(J)) / J
        else:
            q = torch.zeros(len(y), L, device=dev)
            for j in range(J): q.scatter_add_(1, v[:, j:j+1], torch.ones(len(y), 1, device=dev))
            loss = -(q/q.sum(1, keepdim=True) * torch.log_softmax(g, -1)).sum(-1).mean()
        loss.backward(); opt.step(); sch.step(); opt.zero_grad(); run += loss.item()
        if i % 400 == 0: print(f"  ep{ep} {i}/{len(dl)} loss {run/(i+1):.4f}", flush=True)
print(f"trained in {time.time()-t0:.0f}s")

@torch.no_grad()
def score(name, fn):
    p = ROOT/"data/eval"/f"{name}.jsonl"
    if not p.exists(): return
    ev = [r for r in load_jsonl(p) if r["tier"] in idx]
    net.eval(); pred = []
    for i in range(0, len(ev), 64):
        b = [r["text"] for r in ev[i:i+64]]
        e = tok(b, truncation=True, max_length=256, padding=True, return_tensors="pt").to(dev)
        g = net(e["input_ids"], e["attention_mask"])
        pr = torch.softmax(g, -1)
        pred += (pr.mean(1) if MODE == "perjuror" else pr).argmax(-1).cpu().tolist()
    acc = np.mean([LABELS[a] == r["tier"] for a, r in zip(pred, ev)])
    print(f"  {name:<34} n={len(ev):5d}  acc={acc:.4f}")

for nm in (f"{SIG}-real-gold-v2", f"{SIG}-real-gold", f"{SIG}-entsec-gold"):
    score(nm, None)
