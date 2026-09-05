# complexity (2 tier) — which model should serve this request?

One decision: the small fast model, or the large capable one. There is no third
option and no "it depends" tier. Assign SIMPLE or REASONING.

## SIMPLE
The small model can produce a good answer directly. Most traffic is here.
- Recall, definitions, explanations of established topics.
- Short or medium writing: emails, summaries, rewrites, translations, posts.
- Code from a clear specification; a function, a script, a query, a fix.
- Formatting, extraction, classification, list-making.
- Casual conversation, roleplay, brainstorming, opinions.
- Long output does NOT move a request here to REASONING. A 3,000-word article
  on a familiar topic is SIMPLE.

## REASONING
The large model is worth its cost because a small model would produce a
confident WRONG answer, or a shallow one that misses what the request is for.
Assign REASONING if ANY of these hold:
- **Derivation.** Multi-step maths, proof, formal logic, or a calculation whose
  intermediate steps must be right for the answer to be right.
- **Constraints.** Several requirements interact and must be satisfied together:
  scheduling, allocation, puzzles, configurations with trade-offs.
- **Diagnosis.** Finding a cause by tracing behaviour through a system — a bug
  from a stack trace, an outage from logs, a failure from symptoms.
- **System design.** Architecture, migration, or strategy where components
  interact and named alternatives must be weighed against competing objectives.
- **Specialist depth.** Expert-level judgement in law, medicine, finance or
  research where a plausible-sounding wrong answer is likely and costly.

## Boundary rules
- **Length is not complexity.** Judge the thinking required, not the words
  produced. That is what the cost signal is for.
- **Ambition is not complexity.** "Write a novel" is SIMPLE. "Given these five
  constraints, determine whether a valid schedule exists" is REASONING.
- **Would a confident wrong answer be easy to produce?** If yes, REASONING. If
  the main risk is a dull answer rather than an incorrect one, SIMPLE.
- **Topic difficulty for a human is irrelevant.** Explaining general relativity
  is SIMPLE; the model knows it. Deriving an unfamiliar result is REASONING.
- When genuinely torn, choose SIMPLE. The small model serves most traffic and
  the cost of the large model is paid on every request routed to it.
