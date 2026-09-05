# reasoning — does this request need a reasoning model?

Routing decision: send to the large reasoning-capable model, or the small fast one.

Assign YES only if the request requires extended deliberation the model must
perform BEFORE it can answer. Otherwise NO.

## YES
The answer depends on a chain of derivation, search, or proof that cannot be
produced by recall or by writing fluently. Getting it wrong is a matter of
reasoning failure, not of style.
- Multi-step mathematics, formal logic, or proof.
- Constraint satisfaction: scheduling, packing, puzzles with interacting rules.
- Debugging that requires tracing state through a program to find a cause.
- Problems where a plausible-sounding answer is likely to be WRONG unless the
  steps are actually worked through.

## NO
Everything else, including things that are long, sophisticated, or expert.
- Writing, summarising, translating, formatting, rewriting — however long.
- Explaining or teaching a topic the model already knows.
- Producing code from a clear specification.
- Design and architecture discussion, recommendations, comparisons, opinions.
- Open-ended advice where many good answers exist.

## Boundary rules
- **Length is not reasoning.** A 5,000-word report is NO. A three-line puzzle
  with interacting constraints is YES.
- **Expertise is not reasoning.** Explaining general relativity is NO; the model
  knows it. Deriving an unfamiliar result is YES.
- **Difficulty for a HUMAN is irrelevant.** Judge whether the MODEL must
  deliberate.
- **Ambition is not reasoning.** "Design a distributed system" invites a long
  considered answer, but the model can write it directly: NO. "Given these five
  latency and consistency constraints, determine whether a valid configuration
  exists" is YES.
- If a competent fluent answer can be produced by writing straight through
  without stopping to work anything out, it is NO.
