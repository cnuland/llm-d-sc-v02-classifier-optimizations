"""Synthetic data generation for the llm-d-sc classifiers.

The v1 corpora failed in two measurable ways, and the design here answers both:

  1. LABEL DRIFT. cost-train.jsonl only reached 0.8637 five-fold CV against
     itself and a head trained on it LOST to hand-written anchors (0.633 vs
     0.833) -- its labels encoded a different task from the eval. Fix: every
     generator call is grounded in the SAME rubric that was just shown to
     reproduce the gold labels (complexity .9875 / cost 1.000 / sensitivity
     1.000), and every row is blind re-labelled before it is kept.

  2. GENERATOR TELLS. If one model writes everything in one voice, the encoder
     learns the voice. Fix: a (tier x domain x register) grid, plus explicit
     ANTI-CUE generation -- LOW cost items that never say "brief", MEDIUM
     complexity items that superficially read as COMPLEX -- so the model cannot
     pass by keyword matching.

The highest-value shape is the MINIMAL PAIR: two prompts alike in topic and
register that differ by exactly the feature that moves the tier. Boundaries are
where every confusion matrix puts its mass, so we sample them deliberately
rather than hoping the random draw covers them.
"""
from __future__ import annotations
import json, hashlib, pathlib, random, re, sys
sys.path.insert(0, "/Users/cnuland/llm-d-sc-accuracy/harness")
import llmkit as L

ROOT = pathlib.Path("/Users/cnuland/llm-d-sc-accuracy")

DOMAINS = [
    "software engineering", "devops and cloud infrastructure", "cybersecurity",
    "data science and statistics", "medicine and clinical care", "law and contracts",
    "finance and accounting", "human resources", "manufacturing and logistics",
    "education and teaching", "scientific research", "marketing and sales",
    "customer support", "public sector and government", "retail and e-commerce",
    "energy and utilities", "agriculture", "travel and hospitality",
    "media and publishing", "real estate and construction", "insurance",
    "telecommunications", "automotive", "biotech and pharma", "gaming",
    "non-profit and NGO operations", "personal productivity", "home and DIY",
    "cooking and nutrition", "fitness and sport", "music and audio",
    "history and humanities", "mathematics", "physics and chemistry",
    "environment and climate", "transportation and aviation",
]

REGISTERS = [
    "plain professional English",
    "terse, almost telegraphic, no pleasantries",
    "rambling and over-explained, with background the model does not need",
    "casual chat style, lowercase, contractions",
    "non-native English with slightly off idiom but clear meaning",
    "contains a typo or two and inconsistent capitalisation",
    "formal and bureaucratic",
    "pasted from a ticket or email thread, with a header line",
    "includes a short code block or log excerpt",
    "phrased as a question to a colleague",
]

ITEM_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["text", "tier"],
        "properties": {"text": {"type": "string"}, "tier": {"type": "string"}},
    }}},
}


def _schema_for(labels):
    s = json.loads(json.dumps(ITEM_SCHEMA))
    s["properties"]["items"]["items"]["properties"]["tier"]["enum"] = list(labels)
    return s


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def key(t: str) -> str:
    return hashlib.blake2b(norm(t).encode(), digest_size=12).hexdigest()


# ---------------------------------------------------------------- generators

def gen_grid(signal, labels, domain, register, tier, n, model, tag):
    """n prompts for one (tier, domain, register) cell."""
    p = (f"{L.rubric(signal)}\n\n---\n\n"
         f"Write {n} DISTINCT user requests to an AI assistant.\n"
         f"Domain: {domain}\nRegister: {register}\n"
         f"Every one must be tier {tier} under the rubric above.\n\n"
         "Rules:\n"
         "- Vary length, phrasing and sub-topic; do not reuse a sentence frame.\n"
         "- Write what a real user would type, not a textbook example.\n"
         "- Do not mention the tier, the rubric, or classification.\n"
         "- No numbering or quotes inside the text field.\n")
    return L.ask_json(p, _schema_for(labels), model=model, max_tokens=8000,
                      seed_tag=tag)["items"]


def gen_pairs(signal, labels, a, b, domain, n, model, tag):
    """n MINIMAL PAIRS across the a|b boundary: same topic, tier flipped."""
    p = (f"{L.rubric(signal)}\n\n---\n\n"
         f"Domain: {domain}\n\n"
         f"Write {n} MINIMAL PAIRS of user requests that straddle the {a} / {b} "
         f"boundary.\n\nIn each pair the two requests must share topic, length "
         f"and tone, and differ ONLY by the feature that moves the tier. Someone "
         f"skimming should find them hard to tell apart; someone applying the "
         f"rubric should find them unambiguous.\n\n"
         f"Return {2*n} items alternating {a}, {b}, {a}, {b}, ... Label each "
         f"correctly. Do not mention tiers in the text.\n")
    return L.ask_json(p, _schema_for(labels), model=model, max_tokens=8000,
                      seed_tag=tag)["items"]


def gen_anticue(signal, labels, tier, decoy, domain, n, model, tag):
    """Items of `tier` that superficially LOOK like `decoy`.

    Without these the encoder learns surface cues -- 'briefly' => LOW cost,
    'design' => COMPLEX -- and collapses on traffic that does not use them.
    """
    p = (f"{L.rubric(signal)}\n\n---\n\n"
         f"Domain: {domain}\n\n"
         f"Write {n} user requests that are genuinely tier {tier}, but which a "
         f"careless reader would guess are {decoy}.\n\n"
         f"Use the surface features associated with {decoy} -- its vocabulary, "
         f"framing and apparent scale -- while the substance stays firmly {tier} "
         f"under the rubric. These are adversarial examples for a classifier that "
         f"has learned keywords instead of meaning.\n"
         f"Label every item {tier}. Do not mention tiers in the text.\n")
    return L.ask_json(p, _schema_for(labels), model=model, max_tokens=8000,
                      seed_tag=tag)["items"]


def gen_from_real(signal, labels, exemplars, tier, n, model, tag):
    """Rewrite real traffic into `tier` while keeping real users' voice."""
    ex = "\n".join(f"- {e[:220]}" for e in exemplars)
    p = (f"{L.rubric(signal)}\n\n---\n\n"
         f"Here are real prompts people actually sent to an AI assistant:\n{ex}\n\n"
         f"Write {n} NEW requests that read like they came from the same "
         f"population -- same messiness, same voice, same kinds of goals -- but "
         f"which are all tier {tier} under the rubric.\n"
         f"Do not copy or paraphrase the examples; borrow only the register.\n"
         f"Do not mention tiers.\n")
    return L.ask_json(p, _schema_for(labels), model=model, max_tokens=8000,
                      seed_tag=tag)["items"]


# ---------------------------------------------------------------- verification

def blind_relabel(signal, labels, texts, model, batch=20, effort="low", pass_tag=""):
    """Independent label from text + rubric alone. The judge never sees the
    generator's intended tier, so agreement is evidence rather than assent."""
    out = []
    chunks = [texts[i:i+batch] for i in range(0, len(texts), batch)]

    def one(chunk):
        """Label a batch; on failure bisect, and only give up per item.

        A single prompt the model will not answer (refusal, or a paste that
        blows the budget) must not take down a run of thousands. Unanswerable
        items come back labelled None and are dropped by the caller, which is
        recorded rather than silently absorbed.
        """
        listing = "\n".join(f"{i+1}. {t}" for i, t in enumerate(chunk))
        p = (f"{L.rubric(signal)}\n\n---\n\nClassify each request into exactly one "
             f"of: {', '.join(labels)}. Apply the rubric strictly, including its "
             f"boundary rules.\n\n{listing}")
        try:
            # pass_tag joins the cache key, so repeated passes over identical
            # prompts genuinely re-query instead of replaying one cached answer.
            # Without it a self-consistency measurement reads 100% by construction.
            items = L.ask_json(p, L.label_schema(labels, with_reason=False),
                               model=model, effort=effort, max_tokens=8000,
                               seed_tag=pass_tag)["items"]
            if len(items) == len(chunk):
                return items
        except Exception:
            pass
        if len(chunk) == 1:
            return [{"label": None, "confidence": 0.0}]
        h = len(chunk) // 2
        return one(chunk[:h]) + one(chunk[h:])

    for part in L.amap(one, chunks, workers=14, desc=f"relabel {signal}"):
        out.extend(part)
    return out
