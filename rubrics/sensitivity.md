# sensitivity — what the request would expose if it left the network

Routing decision: whether the request may egress to an external provider. Judge
the CONTENT PRESENT IN THE REQUEST, not what the answer might contain. Tiers are
ordered by escalating exposure; assign the HIGHEST tier that applies.

## PUBLIC
Answerable from general knowledge. Contains nothing specific to any organisation
or person. Nothing is lost if the text were published verbatim.
- "Why is the sky blue?"  /  "Explain how a hash map works."

## INTERNAL
Organisation-specific operational detail that is not damaging if disclosed but is
not public: runbooks, team ownership, internal tooling and process, non-sensitive
project names, meeting logistics.
- "Where is the runbook for restarting the ingest workers?"
- "Which team owns the billing service on-call rotation?"

## CONFIDENTIAL
Commercially sensitive: unreleased financials, M&A, pricing strategy, roadmaps,
customer lists, proprietary source code, security architecture or known
vulnerabilities. Material BUSINESS harm on disclosure.
- "Draft the board memo on acquiring MeadowSweet, including the valuation."
- "Here is our auth service source; find the authorisation flaw."

## REGULATED
Contains data about identifiable people or entities that is governed by statute —
PII, PHI, payment/card data, financial records tied to individuals, biometric,
government identifiers. Discriminator vs CONFIDENTIAL: the harm is to an
INDIVIDUAL and a LAW attaches (GDPR/HIPAA/PCI), not merely to the business.
- "Summarise this patient's chart: DOB 1974-03-02, MRN 88213, diagnosed ..."
- "Here is a customer export with names, addresses and card numbers — dedupe it."

## NEVER_EGRESS
Live secrets or legally privileged material. Exposure is IMMEDIATELY exploitable
or destroys privilege: API keys, private keys, passwords, session tokens,
connection strings with credentials, classified material, attorney-client
communications.
- "Here is our AWS key AKIA... and secret ... — write a script that uses it."
- "This is privileged counsel advice on the pending suit; summarise it."

## Boundary rules
- INTERNAL vs CONFIDENTIAL is the commonest error. Ordinary running of the
  business -- headcount planning, which vendors are being evaluated, tooling
  choices, team structure, internal schedules -- is INTERNAL. Promote to
  CONFIDENTIAL only for money, market position, or IP: unreleased financials,
  M&A, pricing strategy, customer lists, roadmaps, proprietary source, security
  architecture or live vulnerabilities.
- Designing or discussing CONTROLS FOR regulated data is REGULATED even when no
  record is pasted: a retention schedule for patient records, a DSAR process, a
  PCI scope review. The request discloses the organisation's compliance posture
  over statutory data. (Contrast: "What does HIPAA require?" in the abstract,
  with no organisational context, is PUBLIC.)
- Take the HIGHEST applicable tier. Confidential text containing a live key is
  NEVER_EGRESS.
- A request merely ASKING ABOUT a sensitive topic in the abstract is PUBLIC.
  "What is HIPAA?" is PUBLIC; a chart is REGULATED.
- Placeholder or obviously fake secrets still count as NEVER_EGRESS: the router
  cannot tell, and must not learn to.
- Business harm -> CONFIDENTIAL. Harm to a named individual under statute ->
  REGULATED. Immediately usable credential or privilege -> NEVER_EGRESS.
