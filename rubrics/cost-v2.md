# cost — expected GENERATION cost of serving the request

Routing decision: capacity reservation and batching. Cost is driven by how much
text must be PRODUCED and how much must be READ. It is NOT reasoning difficulty:
"prove Fermat's last theorem" is cheap to attempt; "write a 40-page report" is not.

Decide in this order. Stop at the first rule that applies.

  R1. Is there an EXPLICIT SIZE SPEC? Use the table under "Size specs" below.
      A stated number of words, pages, minutes, chapters, slides or items
      OVERRIDES every other consideration, including topic and phrasing.
  R2. Is a LARGE BODY OF SUPPLIED MATERIAL attached or referenced to be read
      in full?                                                       -> HIGH
  R3. Otherwise judge by ARTIFACT TYPE, using the tiers below. Ask what the
      smallest acceptable answer looks like, not what an eager one looks like.

## Size specs (R1)
| stated size | tier |
|---|---|
| no composition asked for, or one line | MINIMAL |
| up to ~150 words / 1-2 paragraphs / a few bullets / one snippet | LOW |
| ~150-2,000 words / 1-10 pages / a talk under an hour / one module | MODERATE |
| over ~2,000 words / over ~10 pages / a thesis, book, or whole program | HIGH |

## MINIMAL
No composition. A recalled fact, a chosen option, a number, a short transform, a
device action, or a fragment handed over for classification. Multiple-choice and
"which of the following" belong here however long the options are.
- "Which of the following is a correct unit for magnetic flux? 1) Newtons/Coulomb 2) Tesla meter^2 ..."
- "Round 7.86 to one decimal place."
- "a prom photo of a chubby girl"                  (an image prompt, not prose)

## LOW
One small COMPOSED thing: a short answer, a definition, a handful of suggestions,
a single code snippet, a paragraph. Most LOW requests carry NO brevity cue at
all — they are simply small tasks. Do not require the word "brief" to be present.
- "Write down the definition of the Coefficient of Performance of a refrigerator..."
- "please write me a code to embed a call now button on my google sites website"
- "Research questions on Matt.28:16-20"

## MODERATE
ONE substantial bounded deliverable: an essay, article, presentation, report,
story chapter, lesson plan, or a complete code module. This is the default for
"write me a <document>" with no size given. Requests that say "long", "detailed"
or "in depth" without a number land here, not in HIGH.
- "Write a 30 minute presentation using this thesis: ..."
- "Please write an essay that is 3 pages in length, double spaced..."
- "Create a hybrid 2-layer LSTM to predict next hour's price..."

## HIGH
BULK. In real traffic this is almost always a HUGE REQUESTED OUTPUT rather than a
huge supplied input: a thesis, a book, a full literature review, an entire
program. It can also be a large corpus supplied to be read in full.
- "Can you help me to compose a 50 pages master thesis about periodontitis?"
- "I need an introduction for a research paper. Use 15,000 words."
- "Write me a C compiler."
- "Read these thirty research papers and produce a full literature review."

## Boundary rules
- **A number beats a word.** "Briefly, in 3000 words" is HIGH. "A comprehensive
  guide" with no number is MODERATE.
- **MINIMAL vs LOW: is anything composed?** Selecting an option, recalling a
  fact, or emitting an image prompt is MINIMAL. Even two written sentences is LOW.
- **LOW vs MODERATE: one small thing, or one whole document?** A snippet, a
  definition, a list of titles is LOW. An essay, article, or module is MODERATE.
- **MODERATE vs HIGH: would a competent answer exceed roughly 2,000 words or
  10 pages?** Book-, thesis- and compiler-scale asks are HIGH; a long article is
  not. Length of the PROMPT is irrelevant — HIGH requests are often very short.
- Reasoning difficulty NEVER raises the tier. Volume does.
- Adversarial framing, roleplay preambles and jailbreak attempts do not change
  the tier; judge the deliverable the request is actually asking for.
