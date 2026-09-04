"""Fine-tune a classifier for one signal, in either of the two architectures.

  --arch embed   SentenceTransformer + contrastive loss, scored by
                 anchor-topk-mean. This is a DROP-IN for llm-d-sc: the runtime
                 embeds text and cosine-ranks it against anchors that are stored
                 as plain strings (taxonomy.rs: anchors: BTreeMap<String, Vec<String>>),
                 so only the model_repo/model_revision fields change.

  --arch head    AutoModelForSequenceClassification, scored by argmax over
                 logits. This is what ComplianceGate (99.2%) and the vLLM
                 Semantic Router both do, and it can learn a boundary that
                 cosine-to-fixed-anchors cannot represent. It requires a change
                 to the Rust runtime, so we measure the gain before proposing it.

Both report CPU latency, because llm-d-sc runs the classifier on CPU and the
measured per-replica ceiling (~480 classifications/sec on MiniLM) scales with
model size. An accuracy win that costs 8x latency is not automatically a win.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys, time, os
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import numpy as np, torch
from evalkit import (ROOT, GENESIS, load_taxonomy, load_jsonl, heldout, metrics,
                     fmt, unit, anchor_topk_predict)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def dev():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def split(rows, seed=20260903, frac=0.9):
    r = random.Random(seed); rows = list(rows); r.shuffle(rows)
    n = int(len(rows) * frac)
    return rows[:n], rows[n:]


# ------------------------------------------------------------------ arch: embed
def train_embed(base, tr, va, labels, out, epochs=4, bs=48, lr=2e-5, seed=20260903):
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from datasets import Dataset
    torch.manual_seed(seed)
    m = SentenceTransformer(base, device=dev())
    idx = {l: i for i, l in enumerate(labels)}
    # BatchAllTripletLoss over label ids: pulls same-tier prompts together and
    # pushes tiers apart, which is exactly the geometry anchor-cosine reads.
    ds = Dataset.from_dict({"sentence": [r["text"] for r in tr],
                            "label": [idx[r["tier"]] for r in tr]})
    loss = losses.BatchAllTripletLoss(m)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "_ck"), num_train_epochs=epochs,
        per_device_train_batch_size=bs, learning_rate=lr, warmup_ratio=0.1,
        fp16=False, bf16=False, logging_steps=200, save_strategy="no",
        report_to=[], batch_sampler="group_by_label", seed=seed)
    SentenceTransformerTrainer(model=m, args=args, train_dataset=ds, loss=loss).train()
    m.save(str(out))
    return m


# ------------------------------------------------------------------- arch: head
# Signals whose tiers form a genuine total order. cost and sensitivity escalate
# monotonically (MINIMAL<LOW<MODERATE<HIGH; PUBLIC<INTERNAL<CONFIDENTIAL<
# REGULATED<NEVER_EGRESS). complexity does NOT: the rubric defines COMPLEX and
# REASONING as different KINDS of hard -- breadth of design versus depth of
# derivation -- so treating them as adjacent ranks would encode a relationship
# the taxonomy does not claim.
ORDERED = {"cost", "sensitivity"}


def ordinal_smooth(q, labels, alpha=0.12, decay=0.5):
    """Move a little target mass onto neighbouring TIERS.

    78% of this model's errors are adjacent-tier confusions, but plain
    cross-entropy treats every wrong class as equally wrong: predicting COMPLEX
    for a MEDIUM prompt is scored exactly like predicting REASONING. Leaking
    `alpha` of the mass to immediate neighbours (and `alpha*decay` to
    next-nearest) tells the model that near misses are near, which is what an
    ordinal-aware objective buys -- better calibration and fewer severe
    misclassifications (arXiv 2507.00736, 2606.25769).

    Applied only where the tiers really are ordered; see ORDERED.
    """
    n = len(labels)
    out = np.zeros(n, dtype=np.float32)
    for i, m in enumerate(q):
        if m <= 0:
            continue
        keep = m * (1.0 - alpha)
        out[i] += keep
        spill = m * alpha
        w = {}
        for j in range(n):
            d = abs(i - j)
            if d == 0:
                continue
            if d <= 2:
                w[j] = decay ** (d - 1)
        tot = sum(w.values()) or 1.0
        for j, wt in w.items():
            out[j] += spill * wt / tot
    return (out / out.sum()).tolist()


def soft_target(row, labels, smooth=0.03):
    """Per-example target DISTRIBUTION over labels.

    Rows the labellers agreed on become a near-one-hot with a little smoothing.
    Rows they split on become the vote distribution itself, so a 1-1 split
    trains toward 0.5/0.5 rather than being discarded or forced to a coin-flip
    hard label. Those split rows ARE the boundary cases -- the confusion
    matrices put nearly all their mass on the same boundaries -- so throwing
    them away removes precisely the supervision the model most needs.

    Mixing hard and soft targets beats either alone (arXiv 2605.26246); here the
    mixture is per-example rather than per-loss-term, which keeps the objective
    a single soft cross-entropy.
    """
    idx = {l: i for i, l in enumerate(labels)}
    # A distilled row already carries the teacher's full distribution, which is
    # strictly more informative than any vote list reconstructed from it.
    dist = row.get("soft_dist")
    if dist and len(dist) == len(labels):
        q = np.asarray(dist, dtype=np.float32)
        return (q / max(1e-9, q.sum())).tolist()
    q = np.full(len(labels), smooth / len(labels), dtype=np.float32)
    votes = row.get("votes") or [row["tier"]]
    votes = [v for v in votes if v in idx]
    if not votes:
        votes = [row["tier"]]
    w = (1.0 - smooth) / len(votes)
    for v in votes:
        q[idx[v]] += w
    return (q / q.sum()).tolist()


def class_weights(tr, labels, escalate=0.0, signal=None):
    """Per-class loss weights: inverse-frequency, optionally escalating by tier.

    Two problems, one mechanism.

    FREQUENCY. Real sensitivity traffic is ~85% PUBLIC and the enterprise corpus
    is dominated by INTERNAL, so an unweighted loss is optimised by answering the
    majority class. Measured on the enterprise-secrets eval: INTERNAL recall
    0.94, REGULATED 0.46-0.51, NEVER_EGRESS 0.69-0.85 — 0.78 accuracy carried
    almost entirely by the largest class.

    ASYMMETRY. For an egress gate the errors are not interchangeable. Letting a
    NEVER_EGRESS prompt through leaks a live credential; over-flagging a PUBLIC
    one costs a cheap round trip. `escalate` raises the weight on higher tiers
    so under-calling sensitivity is penalised harder than over-calling it, which
    is the operationally correct asymmetry rather than a symmetric accuracy
    objective.

    Inverse-frequency uses a square root: full inverse frequency over-corrects
    on a 20:1 imbalance and destroys majority-class precision.
    """
    import collections
    n = collections.Counter(r["tier"] for r in tr)
    tot = sum(n.values())
    w = []
    for i, l in enumerate(labels):
        f = max(1, n.get(l, 1)) / tot
        base_w = (1.0 / f) ** 0.5
        w.append(base_w * (1.0 + escalate * i / max(1, len(labels) - 1)))
    w = np.array(w, dtype=np.float32)
    return (w / w.mean()).tolist()


def train_head(base, tr, va, labels, out, epochs=4, bs=32, lr=2e-5, seed=20260903,
               maxlen=256, use_cpu=False, soft=False, ordinal=False,
               supcon=0.0, supcon_tau=0.1, ordered=False,
               lora=0, wd=0.01, sched="linear", warmup=0.10, cw=None):
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    from datasets import Dataset
    torch.manual_seed(seed)
    idx = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labels),
        id2label={i: l for l, i in idx.items()}, label2id=idx)
    if lora:
        # LoRA as an alternative finetuning mechanism, matching the vLLM
        # Semantic Router recipe (TaskType.SEQ_CLS, rank/alpha/dropout). Only
        # the adapter trains, which is a much stronger regulariser than full
        # fine-tuning -- potentially useful against a corpus with 5-6.5% label
        # noise, where full fine-tuning has the capacity to memorise the noise.
        from peft import LoraConfig, TaskType, get_peft_model
        targets = (["query", "key", "value", "dense"]
                   if "bert" in base.lower() and "modern" not in base.lower()
                   else ["Wqkv", "Wo", "Wi"])
        try:
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.SEQ_CLS, r=lora, lora_alpha=lora * 2,
                lora_dropout=0.1, target_modules=targets,
                modules_to_save=["classifier", "score"]))
            model.print_trainable_parameters()
        except Exception as e:
            print(f"  LoRA setup failed ({type(e).__name__}: {str(e)[:80]}); "
                  f"falling back to full fine-tuning")

    def prep(rows):
        cols = {"text": [r["text"] for r in rows],
                "labels": [idx[r["tier"]] for r in rows]}
        if soft:
            q = [soft_target(r, labels) for r in rows]
            if ordinal:
                q = [ordinal_smooth(np.array(x), labels) for x in q]
            cols["soft"] = q
        d = Dataset.from_dict(cols)
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=maxlen),
                     batched=True, remove_columns=["text"])

    def acc(p):
        preds = p.predictions
        preds = preds[0] if isinstance(preds, tuple) else preds
        return {"accuracy": (preds.argmax(-1) == p.label_ids).mean()}

    class SoftTrainer(Trainer):
        _warned = False

        """Cross-entropy against a target DISTRIBUTION rather than an index.

        Reduces to ordinary cross-entropy when the target is one-hot, so the
        agreed rows behave exactly as before and only the contested rows change.
        """
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            q = inputs.pop("soft", None)
            labels_in = inputs.pop("labels", None)
            if q is None and not self._warned:
                raise RuntimeError(
                    "soft targets requested but absent from the batch — check "
                    "remove_unused_columns; silently falling back to hard labels "
                    "is how this went unnoticed for a whole round of experiments")
            outputs = model(**inputs, output_hidden_states=(supcon > 0))
            logits = outputs.logits
            W = (torch.tensor(cw, device=logits.device) if cw else None)
            if q is None:
                loss = torch.nn.functional.cross_entropy(logits, labels_in, weight=W)
            else:
                ll = -(q * torch.log_softmax(logits, dim=-1))
                if W is not None:
                    ll = ll * W.unsqueeze(0)      # weight per TRUE-class mass
                loss = ll.sum(-1).mean()
            if supcon > 0:
                loss = loss + supcon * _hier_supcon(
                    outputs.hidden_states[-1], inputs["attention_mask"],
                    (labels_in if labels_in is not None else q.argmax(-1)),
                    n_lab=logits.shape[-1], tau=supcon_tau, ordered=ordered)
            if labels_in is not None:
                inputs["labels"] = labels_in
            return (loss, outputs) if return_outputs else loss

    class SoftCollator(DataCollatorWithPadding):
        """Pad the batch normally, then re-attach the soft targets as a tensor."""
        def __call__(self, features):
            soft_col = [f.pop("soft") for f in features] if "soft" in features[0] else None
            batch = super().__call__(features)
            if soft_col is not None:
                batch["soft"] = torch.tensor(soft_col, dtype=torch.float)
            return batch

    # transformers 5.x dropped warmup_ratio and use_mps_device; compute the
    # warmup in steps so the schedule is identical across versions.
    steps = max(1, (len(tr) // bs) * epochs)
    args = TrainingArguments(
        output_dir=str(out / "_ck"), num_train_epochs=epochs,
        per_device_train_batch_size=bs, per_device_eval_batch_size=64,
        learning_rate=lr, warmup_steps=max(10, int(steps * warmup)), weight_decay=wd,
        lr_scheduler_type=sched,
        eval_strategy="epoch", save_strategy="no", logging_steps=200,
        report_to=[], seed=seed, use_cpu=use_cpu, max_grad_norm=1.0,
        # HF Trainer defaults this to True and silently DROPS any dataset column
        # that is not a parameter of model.forward(). `soft` is not a model
        # input, so it was deleted before collation, SoftCollator never saw it,
        # and compute_loss fell back to plain cross-entropy -- meaning every
        # --soft and --ordinal arm trained on hard labels with no warning.
        remove_unused_columns=not soft)
    if supcon > 0 and not soft:
        raise SystemExit("--supcon requires --soft: the contrastive term lives "
                         "in SoftTrainer.compute_loss, and Trainer would drop it "
                         "silently — the same failure mode as the soft-target bug")
    cls = SoftTrainer if soft else Trainer
    coll = SoftCollator(tok) if soft else DataCollatorWithPadding(tok)
    t = cls(model=model, args=args, train_dataset=prep(tr),
            eval_dataset=prep(va), data_collator=coll, compute_metrics=acc)
    t.train()
    model.save_pretrained(out); tok.save_pretrained(out)
    return model, tok



def _hier_supcon(hidden, attn_mask, y, n_lab, tau=0.1, ordered=False):
    """Hierarchy-aware supervised contrastive loss over masked mean-pooled states.

    Plain SupCon (Khosla et al. 2020) treats every non-same-label example as an
    equally bad neighbour. That is wrong for an ORDERED taxonomy: pushing
    INTERNAL away from CONFIDENTIAL as hard as from NEVER_EGRESS discards the
    fact that the ladder is a ladder, and it is the adjacent pair that the model
    actually confuses (§55).

    So positives are soft. Pair weight is 1 for the same tier and decays with
    rank distance for ordered signals:  w = 1 - |rank_i - rank_j| / (K-1).
    For unordered taxonomies (complexity) it degenerates to standard SupCon.

    Why add this at all: §56/§60 showed threshold moves only slide along the
    containment curve, and §57 showed the failure is where the two classes sit
    in representation space. A contrastive term acts on exactly that geometry,
    which reweighting the cross-entropy does not.
    """
    h = hidden * attn_mask.unsqueeze(-1)
    z = h.sum(1) / attn_mask.sum(1, keepdim=True).clamp(min=1)
    z = torch.nn.functional.normalize(z, dim=-1)
    sim = (z @ z.T) / tau
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)                 # never contrast with self

    d = (y.unsqueeze(0) - y.unsqueeze(1)).abs().float()
    W = (1.0 - d / max(1, n_lab - 1)) if ordered else (d == 0).float()
    W = W.masked_fill(eye, 0.0)
    if W.sum() == 0:
        return z.sum() * 0.0                         # no positives in this batch

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    per_row = (W * log_prob).sum(1) / W.sum(1).clamp(min=1e-8)
    keep = W.sum(1) > 0
    return -(per_row[keep].mean())


# ------------------------------------------------------------------- evaluation
def medoid_anchors(model, tr, labels, counts, seed=0):
    """Re-select anchors IN THE FINE-TUNED SPACE.

    Fine-tuning moves the embedding geometry, but anchor-topk-mean still scores
    against whatever text sits in classifiers/<sig>.json. Keeping the original
    anchors after fine-tuning measures a model against landmarks that have
    drifted out from under it.

    Selection rule: rank each label's training rows by cosine to that label's
    centroid, DROP THE OUTLIER DECILE, then take an even spread across the
    remaining rank order.

    The obvious choice, k-medoids with farthest-point initialisation, is
    actively wrong here and was measured doing so: triplet loss collapses each
    class into a tight cluster, so "maximally distant point" selects
    pathological noise, and Lloyd cannot recover because each outlier claims a
    singleton cluster and stays its own medoid. Its picks sat at cosine +0.05 to
    +0.31 from their class centroid where the class mean was +0.78 to +0.98, and
    it scored 0.0703 on cost real-gold against 0.7190 for the incumbent anchors.

    Trimming the outlier decile and spreading over the rest scores 0.7424 --
    ahead of the incumbents, deterministic, and no clustering required. Anchors
    stay plain text, so the output is a drop-in `classifiers/<sig>.json`.
    """
    import collections
    by = collections.defaultdict(list)
    for r in tr:
        by[r["tier"]].append(r["text"])
    rng = np.random.default_rng(seed)
    out = {}
    for l in labels:
        txt = by.get(l, [])
        k = counts[l]
        if len(txt) <= k:
            out[l] = list(txt)
            continue
        if len(txt) > 1500:                 # cost, not quality: 1500 is plenty
            txt = [txt[i] for i in rng.choice(len(txt), 1500, replace=False)]
        X = unit(model.encode(txt, batch_size=64, convert_to_numpy=True,
                              show_progress_bar=False).astype(np.float64))
        mu = unit(X.mean(0, keepdims=True))[0]
        order = np.argsort(-(X @ mu))                    # most central first
        keep = order[:max(k, int(len(order) * 0.90))]    # drop the outlier decile
        step = max(1, len(keep) // k)
        out[l] = [txt[int(keep[i * step])] for i in range(k)]
    return out


def eval_embed(model, tax, rows, labels, top_k=3, anchors=None):
    anc = anchors or tax["anchors"]
    a_txt = [t for l in labels for t in anc[l]]
    a_lab = [l for l in labels for _ in anc[l]]
    qv = model.encode([r["text"] for r in rows], batch_size=64, convert_to_numpy=True,
                      show_progress_bar=False)
    av = model.encode(a_txt, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    pred, _ = anchor_topk_predict(qv, av, a_lab, labels, top_k)
    return pred


def eval_head(model, tok, rows, labels, bs=64, maxlen=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            b = tok([r["text"] for r in rows[i:i+bs]], truncation=True,
                    max_length=maxlen, padding=True, return_tensors="pt")
            out += model(**b).logits.argmax(-1).tolist()
    return [labels[i] for i in out]


def cpu_latency(fn, texts, n=60):
    """p50/p99 for single-request CPU inference, which is how llm-d-sc serves."""
    ts = []
    for t in texts[:n]:
        s = time.perf_counter(); fn(t); ts.append((time.perf_counter() - s) * 1000)
    ts.sort()
    return ts[len(ts)//2], ts[min(len(ts)-1, int(len(ts)*0.99))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("signal"); ap.add_argument("--arch", default="embed",
                                               choices=["embed", "head"])
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--train", default=None,
                    help="comma-separated jsonl paths; corpora are concatenated")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20260903,
                    help="training seed; varying it gives ensemble members that\n                          differ by initialisation and data order alone")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cpu", action="store_true",
                    help="train on CPU; DeBERTa-v3 diverges under MPS")
    ap.add_argument("--maxlen", type=int, default=256,
                    help="token truncation; 8.1%% of real prompts exceed 256")
    ap.add_argument("--classweight", action="store_true",
                    help="inverse-sqrt-frequency class weights")
    ap.add_argument("--escalate", type=float, default=0.0,
                    help="extra weight on higher tiers; penalises UNDER-calling "
                         "sensitivity harder than over-calling it")
    ap.add_argument("--lora", type=int, default=0,
                    help="LoRA rank; 0 = full fine-tuning (vLLM SR uses 8-32)")
    ap.add_argument("--wd", type=float, default=0.01,
                    help="weight decay; vLLM SR uses 0.1 for regularisation")
    ap.add_argument("--sched", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--warmup", type=float, default=0.10,
                    help="warmup as a fraction of steps; vLLM SR uses 0.06")
    ap.add_argument("--ordinal", action="store_true",
                    help="leak target mass to neighbouring tiers; only valid "
                         "where the tiers are genuinely ordered (cost, sensitivity)")
    ap.add_argument("--supcon", type=float, default=0.0,
                    help="weight on the hierarchy-aware supervised contrastive "
                         "term; acts on representation GEOMETRY, which "
                         "reweighting cross-entropy cannot")
    ap.add_argument("--supcon-tau", type=float, default=0.1,
                    dest="supcon_tau", help="contrastive temperature")
    ap.add_argument("--soft", action="store_true",
                    help="train on per-example jury vote distributions, and "
                         "include the contested rows instead of dropping them")
    a = ap.parse_args()

    tax = load_taxonomy(a.signal); labels = tax["labels"]
    srcs = [pathlib.Path(x) for x in (a.train.split(",") if a.train
            else [str(ROOT / f"data/train/{a.signal}-v2.jsonl")])]
    rows, srcnames = [], []
    for s_ in srcs:
        r = load_jsonl(s_)
        rows += r
        srcnames.append(f"{s_.stem}:{len(r)}")
    src = pathlib.Path("+".join(srcnames))
    tr, va = split(rows, seed=a.seed)
    tag = a.tag or f"{a.signal}-{a.arch}-{a.base.split('/')[-1]}"
    out = ROOT / "models" / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] train={len(tr)} val={len(va)} labels={labels} src={src.name}")

    t0 = time.time()
    if a.arch == "embed":
        m = train_embed(a.base, tr, va, labels, out, epochs=a.epochs,
                        bs=a.bs or 48, lr=a.lr)
        counts = {l: len(tax["anchors"][l]) for l in labels}
        newanc = medoid_anchors(m, tr, labels, counts)
        json.dump({**{k: tax[k] for k in ("classifier_id", "signal", "method",
                                          "top_k", "labels")},
                   "taxonomy_revision": tax["taxonomy_revision"] + "-" + tag,
                   "model_repo": tag, "model_revision": "",
                   "anchors": newanc}, open(out / "classifier.json", "w"), indent=1)
        variants = {"": None, " [re-anchored]": newanc}
        predict = lambda rs: eval_embed(m, tax, rs, labels, tax.get("top_k", 3))
        m.to("cpu"); lat = lambda t: m.encode([t], show_progress_bar=False)
    else:
        m, tok = train_head(a.base, tr, va, labels, out, epochs=a.epochs,
                            bs=a.bs or 32, lr=a.lr, use_cpu=a.cpu,
                            maxlen=a.maxlen, soft=a.soft, seed=a.seed,
                            ordinal=a.ordinal and a.signal in ORDERED,
                            lora=a.lora, wd=a.wd, sched=a.sched, warmup=a.warmup,
                            supcon=a.supcon, supcon_tau=a.supcon_tau,
                            ordered=a.signal in ORDERED,
                            cw=(class_weights(tr, labels, a.escalate, a.signal)
                                if (a.classweight or a.escalate) else None))
        if a.ordinal and a.signal not in ORDERED:
            print(f"  NOTE: --ordinal ignored for {a.signal}; its tiers are not "
                  f"a total order (COMPLEX and REASONING are sibling kinds of hard)")
        variants = {"": None}
        predict = lambda rs: eval_head(m, tok, rs, labels, maxlen=a.maxlen)
        m.to("cpu")
        lat = lambda t: eval_head(m, tok, [{"text": t}], labels, maxlen=a.maxlen)
    print(f"  trained in {time.time()-t0:.0f}s")

    import collections as _c
    report = {"tag": tag, "signal": a.signal, "arch": a.arch, "base": a.base,
              "train_rows": len(tr), "epochs": a.epochs, "source": src.name,
              "maxlen": a.maxlen, "soft_targets": a.soft, "lr": a.lr,
              "ordinal": bool(a.ordinal and a.signal in ORDERED),
              "class_weighted": bool(a.classweight or a.escalate),
              "escalate": a.escalate,
              "supcon": a.supcon, "supcon_tau": a.supcon_tau,
              "lora_rank": a.lora, "weight_decay": a.wd,
              "scheduler": a.sched, "warmup_ratio": a.warmup,
              "train_prior": dict(_c.Counter(r["tier"] for r in tr))}
    evalsets = [("heldout-v1", heldout(a.signal))]
    for nm, f in [("real-gold", f"{a.signal}-real-gold"),
                  ("real-contested", f"{a.signal}-real-contested"),
                  ("enterprise-gold", f"{a.signal}-enterprise-gold"),
                  # the slice that can actually see the egress tiers: 166
                  # NEVER_EGRESS and 182 REGULATED rows, against 5 and 7 before
                  ("entsec-gold", f"{a.signal}-entsec-gold"),
                  ("refined-gold", f"{a.signal}-real-gold-v2")]:
        pth = ROOT / f"data/eval/{f}.jsonl"
        if pth.exists():
            evalsets.append((nm, load_jsonl(pth)))
    evalsets.append(("val-internal", va))
    for name, rs in evalsets:
        if not rs:
            continue
        for suffix, anc in variants.items():
            pred = (eval_embed(m, tax, rs, labels, tax.get("top_k", 3), anchors=anc)
                    if a.arch == "embed" else predict(rs))
            mm = metrics([r["tier"] for r in rs], pred, labels,
                         [r.get("hard") for r in rs])
            key = name + suffix.replace(" ", "")
            report[key] = {k: mm[k] for k in ("n", "accuracy", "macro_f1", "wilson95")}
            report[key]["confusion"] = mm["confusion"]
            # Per-label recall matters more than accuracy for a gate: missing a
            # NEVER_EGRESS is not interchangeable with mislabelling a PUBLIC.
            report[key]["recall"] = {
                l: (mm["confusion"][l][l] / max(1, sum(mm["confusion"][l].values())))
                for l in labels}
            print(fmt(a.signal, f"{a.arch}:{name}{suffix}", mm))
    p50, p99 = cpu_latency(lat, [r["text"] for r in heldout(a.signal)])
    report["cpu_p50_ms"], report["cpu_p99_ms"] = round(p50, 2), round(p99, 2)
    print(f"  CPU single-request latency: p50={p50:.1f}ms p99={p99:.1f}ms")
    (ROOT/"reports").mkdir(exist_ok=True)
    json.dump(report, open(ROOT/f"reports/{tag}.json", "w"), indent=1)
