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
Architecture or strategy for a SYSTEM with interacting parts, where the request
is genuinely open-ended and competent experts would return materially DIFFERENT
good answers. Requires weighing competing objectives (cost vs latency vs risk vs
maintainability), not merely making a sensible plan.
Discriminator vs MEDIUM: **is there a conventional right answer?** If yes, MEDIUM.
- "Design a multi-region failover strategy for a payments service."
- "Plan a migration from a monolith to services, with sequencing and rollback."
- "Compare three caching strategies for this workload and recommend one."

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
- MEDIUM vs COMPLEX (the hard one, and where most errors live): MEDIUM has a
  conventional answer; COMPLEX requires SYSTEM-level tradeoffs with no single
  right answer. A schedule, checklist, plan, budget, or single program is MEDIUM
  even if it has many parts and requires care. Do NOT promote to COMPLEX merely
  because a task "involves tradeoffs" or "many edge cases".
- COMPLEX vs REASONING: if correctness is established by a derivation the reader
  can check step by step, REASONING. Design questions stay COMPLEX however hard.
- A MEDIUM task that also demands a specific numeric derivation (e.g. "explain X
  and calculate Y") is REASONING only if the derivation is the point; if the
  calculation is incidental, keep MEDIUM.
- Verbose phrasing of a trivial ask stays SIMPLE.
