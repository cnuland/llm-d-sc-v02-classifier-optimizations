"""Anthropic client for synthetic generation and blind labelling.

Design rules the rest of the project depends on:

  * Structured output. Every call passes a JSON schema via
    `output_config.format`, and label fields carry an `enum`, so the model
    cannot return an invalid label or malformed JSON. This removes the whole
    class of "parse the JSON out of the prose" failures.
  * Blind labelling. A verifier is NEVER shown the label it is checking.
    Showing it produces acquiescence -- the model ratifies whatever it is
    handed and the "verified" corpus inherits the generator's mistakes.
    It sees the text and the rubric, and labels from scratch.
  * Disk cache keyed by (model, prompt, schema, effort, seed_tag), so re-runs
    are free and an interrupted batch resumes instead of re-billing.
"""
from __future__ import annotations
import os, json, hashlib, pathlib, time, threading, random
from concurrent.futures import ThreadPoolExecutor

import anthropic

ROOT = pathlib.Path("/Users/cnuland/llm-d-sc-accuracy")
LLM_CACHE = ROOT / "data" / ".llmcache"
_client = None
_lock = threading.Lock()

# Non-transient: retrying a malformed request just burns the retry budget.
class Refusal(RuntimeError):
    """The model declined to answer. Terminal for this input; bisect around it."""


FATAL = (anthropic.BadRequestError, anthropic.AuthenticationError,
         anthropic.PermissionDeniedError, anthropic.NotFoundError)


def client():
    global _client
    with _lock:
        if _client is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY unset (source llm-d-sc-genesis/.envrc)")
            _client = anthropic.Anthropic(api_key=key, max_retries=5, timeout=600.0)
    return _client


def ask_json(prompt: str, schema: dict, model="claude-sonnet-5", max_tokens=8000,
             system=None, effort=None, cache=True, seed_tag=""):
    """One structured call. Returns the parsed object matching `schema`."""
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.blake2b(json.dumps(
        [model, prompt, system, max_tokens, schema, effort, seed_tag],
        sort_keys=True).encode(), digest_size=16).hexdigest()
    f = LLM_CACHE / f"{key}.json"
    if cache and f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            f.unlink()                       # corrupt cache entry; refetch

    oc: dict = {"format": {"type": "json_schema", "schema": schema}}
    if effort:
        oc["effort"] = effort
    kw = dict(model=model, max_tokens=max_tokens, output_config=oc,
              messages=[{"role": "user", "content": prompt}])
    if system:
        kw["system"] = system

    last = None
    for attempt in range(6):
        try:
            r = client().messages.create(**kw)
            # With `effort` set, reasoning tokens are drawn from the SAME budget
            # as the reply, so a batch that fits comfortably without effort can
            # be cut mid-string with it. Truncation is not a parse bug -- grow
            # the budget and retry rather than discarding the batch.
            if r.stop_reason == "max_tokens":
                kw["max_tokens"] = min(32000, kw["max_tokens"] * 2)
                last = RuntimeError("truncated at max_tokens")
                continue
            txt = "".join(b.text for b in r.content if b.type == "text")
            if not txt.strip():
                if r.stop_reason == "refusal":
                    # Terminal, not transient. Real traffic contains prompts a
                    # model will not engage with; retrying six times cannot
                    # change that and the batch must degrade instead of dying.
                    raise Refusal("model declined to label this batch")
                # Otherwise the whole budget went to reasoning: drop effort and
                # grow it before retrying.
                kw["output_config"].pop("effort", None)
                kw["max_tokens"] = min(32000, kw["max_tokens"] * 2)
                last = RuntimeError(f"empty reply (stop_reason={r.stop_reason})")
                continue
            obj = json.loads(txt)
            if cache:
                f.write_text(json.dumps(obj))
            return obj
        except anthropic.BadRequestError as e:
            # Not every model accepts `effort`; drop it once and retry rather
            # than maintaining a hand-kept list of which models support what.
            if "effort" in str(e) and "effort" in kw["output_config"]:
                kw["output_config"] = {k: v for k, v in kw["output_config"].items()
                                       if k != "effort"}
                continue
            raise
        except (Refusal, *FATAL):
            raise
        except json.JSONDecodeError as e:
            kw["max_tokens"] = min(32000, kw["max_tokens"] * 2)
            last = e
        except Exception as e:               # overloaded / rate limit / transient
            last = e
            time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"LLM failed after 6 retries: {type(last).__name__}: {last}")


def amap(fn, items, workers=12, desc=""):
    """Thread-pooled map preserving input order. Failures propagate."""
    out = [None] * len(items)
    done, lk = [0], threading.Lock()
    step = max(1, len(items) // 10)

    def run(i):
        out[i] = fn(items[i])
        with lk:
            done[0] += 1
            if desc and done[0] % step == 0:
                print(f"  {desc}: {done[0]}/{len(items)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(run, range(len(items))):
            pass
    return out


def label_schema(labels: list[str], with_reason=True) -> dict:
    """Schema for a batch of blind label assignments; `enum` pins the label set."""
    props = {"n": {"type": "integer"},
             "label": {"type": "string", "enum": list(labels)},
             "confidence": {"type": "number"}}
    req = ["n", "label", "confidence"]
    if with_reason:
        props["why"] = {"type": "string"}
        req.append("why")
    return {"type": "object", "additionalProperties": False,
            "required": ["items"],
            "properties": {"items": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": req, "properties": props}}}}


def rubric(signal: str) -> str:
    return (ROOT / "rubrics" / f"{signal}.md").read_text()


def ask_text(prompt: str, model="claude-sonnet-5", max_tokens=1200,
             system=None, cache=True, seed_tag=""):
    """Plain-text completion, memoised on disk like ask_json.

    Added for the quality-delta measurement: that experiment needs ANSWERS to
    real prompts, not structured labels, and every other call in this harness is
    schema-constrained. Caching matters here for a different reason than usual --
    a rerun must reproduce the same answers or the paired comparison changes
    underneath the judge.
    """
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.blake2b(json.dumps(
        ["TEXT", model, prompt, system, max_tokens, seed_tag],
        sort_keys=True).encode(), digest_size=16).hexdigest()
    f = LLM_CACHE / f"{key}.txt"
    if cache and f.exists():
        return f.read_text()
    kw = dict(model=model, max_tokens=max_tokens,
              messages=[{"role": "user", "content": prompt}])
    if system:
        kw["system"] = system
    last = None
    for _ in range(4):
        try:
            r = client().messages.create(**kw)
            out = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            if out.strip():
                if cache: f.write_text(out)
                return out
            last = RuntimeError("empty completion")
        except Exception as e:
            last = e
            if "refus" in str(e).lower():
                raise Refusal(str(e))
            time.sleep(2)
    raise RuntimeError(f"ask_text failed: {type(last).__name__}: {last}")
