"""High-precision pattern detector for live credentials.

Motivation from the gate table: at the NEVER_EGRESS threshold the classifier
contains 82.39% of at-or-above content at 4.34% over-block. The 17.6% that
escapes is live-secret material leaving the network, which is the single worst
failure this system can have -- and unlike every other boundary in the taxonomy
it is NOT a judgement call. "Is this an AWS access key" has a right answer that a
regex knows, and a 22M-parameter bi-encoder is the wrong tool for it.

So this is a precision-first pre-check, OR'd with the classifier: if a rule
fires, the tier floors at NEVER_EGRESS. Rules are written to be near-zero false
positive; recall gaps are left to the model rather than papered over with loose
patterns, because a false positive here blocks legitimate traffic forever while
a false negative merely falls back to the status quo.

Entropy gating is applied only INSIDE an assignment-like context (`key = ...`,
`"token": "..."`), never to free text -- base64 blobs and hashes in prose are
common and are not credentials.
"""
import re, math

def _entropy(s):
    if not s: return 0.0
    from collections import Counter
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in Counter(s).values())

# Each rule is (name, compiled pattern). Anchored and specific by design.
RULES = [
    ("aws_access_key",   re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("aws_secret",       re.compile(r"(?i)aws.{0,20}?(?:secret|private).{0,20}?['\"][0-9a-zA-Z/+]{40}['\"]")),
    ("private_key",      re.compile(r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|PGP|ENCRYPTED)?\s*PRIVATE KEY(?: BLOCK)?-----")),
    ("gh_token",         re.compile(r"\b gh[pousr]_[0-9A-Za-z]{36,255}\b".replace(" ", ""))),
    ("slack_token",      re.compile(r"\bxox[abposr]-[0-9A-Za-z-]{10,}\b")),
    ("stripe_key",       re.compile(r"\b[sr]k_(?:live|test)_[0-9A-Za-z]{16,}\b")),
    ("google_api_key",   re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("openai_key",       re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_key",    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("jwt",              re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("bearer",           re.compile(r"(?i)\bauthorization:\s*bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("conn_string",      re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:@/]+:[^\s:@/]+@")),
    ("private_key_pem",  re.compile(r"\bPuTTY-User-Key-File-\d")),
    ("npm_token",        re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("sendgrid",         re.compile(r"\bSG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b")),
    ("azure_sas",        re.compile(r"(?i)\bsig=[A-Za-z0-9%/+]{30,}&?")),
]

# assignment-shaped, so entropy is judged on a VALUE and not on prose
_ASSIGN = re.compile(
    r"""(?ix)
    \b(?:api[_\-]?key|secret|token|password|passwd|pwd|access[_\-]?key|
        client[_\-]?secret|auth[_\-]?token|session[_\-]?token|credential)
    \b\s*[:=]\s*['"]?([A-Za-z0-9/+_\-\.]{16,})['"]?""")

def detect(text):
    """Return the list of rule names that fired. Empty means no opinion."""
    hits = [n for n, p in RULES if p.search(text)]
    for m in _ASSIGN.finditer(text):
        v = m.group(1)
        # placeholders are still secrets by policy (§rubric) but they are also
        # the main false-positive source for entropy, so require real disorder
        if _entropy(v) >= 3.4 and not re.fullmatch(r"(?i)[x*.\-_]+|your[_\-].*|<.*>", v):
            hits.append("high_entropy_assignment"); break
    return hits
