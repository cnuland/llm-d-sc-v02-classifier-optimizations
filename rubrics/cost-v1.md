# cost — expected GENERATION cost of serving the request

Routing decision: capacity reservation and batching. Cost is driven by how much
text must be PRODUCED and how much must be READ. It is NOT reasoning difficulty:
"prove Fermat's last theorem" is cheap to attempt; "write a 40-page report" is not.

Two questions decide the tier, in order:
  Q1. Does the request supply or span a LARGE BODY of material, or demand
      exhaustive coverage of an entire domain?          -> HIGH
  Q2. Otherwise, how long is the requested output?      -> MINIMAL / LOW / MODERATE

## MINIMAL
A fact, a number, a name, or a device action. Answerable in a phrase or a
sentence with no composition.
- "What is the boiling point of ethanol?"
- "Round 7.86 to one decimal place."
- "Dim the kitchen lights to fifty percent."
- "What is the difference between a comet and an asteroid?"   (a one-line contrast)

## LOW
Short COMPOSED output — a brief paragraph, a handful of lines, a small snippet.
Almost always carries an explicit brevity cue: brief, quick, short, concise,
briefly, in a few sentences, a two sentence ...
- "Write a brief summary of what DNS resolution does."
- "Suggest four titles for a podcast about urban gardening."
- "Explain the tradeoffs between renting and buying a home in a few sentences."

## MODERATE
ONE substantial but BOUNDED deliverable: a full document, guide, playbook,
article, or a complete module with tests. Cued by detailed, thorough, complete,
comprehensive, full, extensive — applied to a SINGLE artifact.
- "Write a complete integration test suite for a hotel booking API."
- "Produce a full curriculum outline for a semester of introductory statistics."
- "Write a comprehensive comparison of four cloud storage providers."

## HIGH
BULK. Either a large supplied corpus must be ingested, or the deliverable spans
an entire domain exhaustively. Look for: these N <documents>, this entire ...,
every page/chapter/component, all of our ..., this whole ..., this archive.
- "Read these thirty research papers and produce a full literature review."
- "Audit this whole monorepo and produce a per package security assessment."
- "Translate this full technical manual of six hundred pages into German."
- "Generate a design system spec covering every component, state, and token."

## Boundary rules
- MINIMAL vs LOW: is anything COMPOSED? Recall or a one-line transform is
  MINIMAL; even a short written paragraph is LOW.
- LOW vs MODERATE: brevity cue ("brief", "short") holds it at LOW no matter the
  topic. "Briefly explain quantum error correction" is LOW.
- MODERATE vs HIGH: is the scope ONE artifact, or MANY/ALL? "A comprehensive
  guide to X" is MODERATE. "Comprehensive analysis of these forty reports" is
  HIGH. Quantity words (thirty, two hundred, six hundred, every, all, entire,
  whole, archive) are the strongest HIGH signal.
- Reasoning difficulty NEVER raises the tier. Length and volume do.
