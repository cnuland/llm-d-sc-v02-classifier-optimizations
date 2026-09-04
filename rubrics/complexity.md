# complexity — how much reasoning capacity the request needs

Routing decision: which model tier answers. Judge the REQUEST, never the answer.
Length is not complexity: a long verbose question can be SIMPLE, a short one can
be REASONING.

## SIMPLE
One fact, definition, or mechanical transformation. Answerable from memory or by
a single rote step, with no composition and no judgement.
- "What is the capital of Mongolia?"
- "Convert 45 mph to km/h."
- "What does the `-r` flag do in `cp`?"

## MEDIUM
ONE well-specified deliverable produced by applying ordinary competence. There is
a conventional way to do it, and two practitioners would produce recognisably the
same thing. Routine planning, routine explanation, and everyday code are MEDIUM
even when they involve small judgement calls or several steps.
- "Write a Python function that parses a CSV and sums a column."
- "Draft a two week training plan for a first time half marathon runner."
- "Create a packing checklist for a five day alpine backpacking trip."
- "Explain how airport ground crews sequence departures during a delay."
- "Write a shell command that renames every .jpeg file in a folder."

## COMPLEX
System-level design or strategy. Apply the three tests below; COMPLEX requires
**at least two** of them to be true. If only one is true, the request is MEDIUM.

  **T1 — MULTIPLE INTERACTING PARTS.** The request names two or more components,
  services, teams, systems or stages that must work together, and the answer has
  to say how they fit. Not "a program with several functions" — separate things
  that interact.

  **T2 — A CHOICE BETWEEN NAMED ALTERNATIVES.** The request asks which of several
  approaches to take, or a competent answer must pick one and justify it against
  others. "Compare X, Y and Z and recommend one" qualifies; "write X" does not.

  **T3 — COMPETING OBJECTIVES.** Two or more goals that trade against each other
  are in play — cost against latency, speed against safety, scope against
  deadline — and the answer must resolve the tension rather than optimise one.

- "Design a multi-region failover strategy for a payments service."
  (T1 regions+services, T3 cost/latency/availability)
- "Plan a migration from a monolith to services, with sequencing and rollback."
  (T1 services+stages, T2 sequencing options, T3 speed/risk)
- "Compare three caching strategies for this workload and recommend one."
  (T2 named alternatives, T3 memory/hit-rate)

Counter-examples that are MEDIUM despite looking complex:
- "Write a complete Go package implementing a rate limiter with tests."
  One artifact, conventional approach. T1 no, T2 no, T3 no.
- "Create a watering schedule for a greenhouse across a season."
  Many parameters but one deliverable and one conventional method. T3 alone at most.
- "Write a program that detects the tempo of a music file."
  Algorithmic choices exist, but one artifact and a standard approach. T2 at most.

## REASONING
A correct answer requires an explicit multi-step derivation in which every
intermediate step must be right — proof, formal logic, non-trivial mathematics,
algorithmic complexity analysis, constraint satisfaction, careful causal
deduction. The answer is checkably right or wrong.
Discriminator vs COMPLEX: **depth of derivation, not breadth of design.**
- "Prove by induction that sum(1..n) = n(n+1)/2."
- "Three people make statements; exactly two lie. Who is truthful?"
- "Derive the worst-case complexity of this recursive routine and prove the bound."

## Boundary rules
- SIMPLE vs MEDIUM: is anything PRODUCED? Recall or a one-step transform is
  SIMPLE. A written artifact, plan, or working code is MEDIUM.
- MEDIUM vs COMPLEX: apply T1/T2/T3 above and COUNT them. Two or more true means
  COMPLEX; one or zero means MEDIUM. Do not judge on how hard or how long the
  task feels — a single artifact that takes a day is still MEDIUM, and a
  three-sentence question about how two services should interact is COMPLEX.
  This boundary drew 2.2x more labeller disagreement than its frequency
  predicts under the previous wording ("system-level tradeoffs, no single right
  answer"), which asked labellers to imagine how other experts would answer
  rather than to inspect the request.
- COMPLEX vs REASONING: if correctness is established by a derivation the reader
  can check step by step, REASONING. Design questions stay COMPLEX however hard.
- A MEDIUM task that also demands a specific numeric derivation (e.g. "explain X
  and calculate Y") is REASONING only if the derivation is the point; if the
  calculation is incidental, keep MEDIUM.
- Verbose phrasing of a trivial ask stays SIMPLE.
