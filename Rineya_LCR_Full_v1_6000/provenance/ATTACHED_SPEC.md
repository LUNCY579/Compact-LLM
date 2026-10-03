Hand the section below to the generating agent verbatim. Everything in it is

derived from measurements in this repository; the rationale for each constraint is

in REHEARSAL\_COMPOSITION.md, REHEARSAL\_BLEND.md and DATASET\_REQUIREMENTS.md.

MISSION

Generate a long-context supervised fine-tuning corpus whose purpose is

rehearsal: it will be interleaved with new-skill training data so that a 596M

dense model learns a new skill without losing its long-context retrieval and

abstention behaviour.

This corpus is not an evaluation set and must never be used as one. It

exists to be trained on.

WHY THIS SHAPE — the measurements you are designing against

A 596M model (Supa 1.7) was fine-tuned on a new skill and re-measured on a frozen

256-row long-context suite. Training cost it 11–43 rows depending on configuration.

Rehearsal on short conversational data made it worse; rehearsal composition

determined which capability survived, and only rehearsal resembling the protected

capability protected it.

There is currently no long-context training data available, which is why this

corpus is being commissioned.

Per-stratum scores on the frozen suite (higher is better, /64 or /32):

stratum

Supa (untrained)

best trained arm

worst trained arm

needle early

20/64

29/64

7/64

needle middle

55/64

47/64

36/64

needle late

47/64

50/64

39/64

needle distributed

39/64

38/64

32/64

alias\_resolution

4/32

7/32

0/32

rule\_exception

15/32

15/32

8/32

causal\_disambiguation

31/32

23/32

12/32

count\_with\_exclusion

16/32

23/32

7/32

temporal\_correlation

29/32

32/32

22/32

latest\_verified

24/32

24/32

24/32

multi\_condition

18/32

18/32

16/32

object\_state\_tracking

24/32

24/32

21/32

variant unanswerable

13/64

13/64

0/64

difficulty hard

84/160

81/160

45/160

Design implications, in priority order:

Early-position needles are the structural weakness. Even untrained, the model

scores 20/64 when the answer-bearing evidence sits early in the context, against

55/64 when it sits in the middle. Under bad rehearsal it collapses to 7/64.

Over-weight early placement.

alias\_resolution is near-absent (4/32 untrained). The model cannot follow an

entity referred to by a different name later in the record.

Abstention is the most fragile thing in the suite. Untrained 13/64; four of six

trained configurations scored 0–3/64. It recovered only when rehearsal was

concentrated on abstention-like material.

causal\_disambiguation and rule\_exception degrade reliably under training

even when other task types do not.

latest\_verified, multi\_condition and object\_state\_tracking are robust —

generate them for coverage, not volume.

HARD CONSTRAINTS — a row violating any of these is unusable

Total length ≤ 2048 tokens under the Qwen3 tokenizer shipped with the target

model, measured on the fully rendered chat sequence (system + user + assistant

template overhead + EOS), not on raw characters. Reserve ≥ 128 tokens for

the completion. Rows that do not fit must be rejected and recorded, never

truncated.

Target context length 1,100–1,600 tokens so the row resembles the \~1,000-word

evaluation contexts while leaving template and answer headroom.

Single-turn only: exactly system, user, assistant. No multi-turn.

The answer must be derivable from the context by construction, and verified by

an independent checker that did not write the answer.

Short answers. The assistant turn is a short exact string — a name, a number,

a date, a short noun phrase — or the literal token UNKNOWN. No prose, no

explanation, no restatement of the question. Median answer ≤ 8 words.

Unanswerable rows answer exactly UNKNOWN — that literal string alone, nothing

else. No "I don't know", no "The context does not say". The scorer matches the

literal.

No overlap with any existing evaluation material. Fresh entity names, fresh

scenario graphs, fresh templates, fresh lexicon. Do not copy, paraphrase, or

re-seed from any existing long-context demonstration suite.

COMPOSITION

Generate 6,000 rows with this stratification. Every dimension must be crossed and

balanced, not merely present in aggregate.

Needle position — weighted toward the measured weakness

position

share

rows

early (evidence in first 25% of context)

40%

2,400

middle

20%

1,200

late

20%

1,200

distributed (2–3 fragments that must be combined)

20%

1,200

Answerability variant

variant

share

rows

answerable, direct

45%

2,700

unanswerable → UNKNOWN

30%

1,800

counterfactual (context contradicts a plausible prior)

15%

900

paraphrase (question wording differs from the context's)

10%

600

30% unanswerable is deliberately higher than the evaluation suite's 25%, because

abstention is the most fragile capability measured.

The four ways a row may be unanswerable — balance them roughly evenly, because a

model that learns one pattern will not generalise to the others:

the queried entity is never mentioned;

the entity is mentioned but the queried attribute is absent;

the information was present and then explicitly retracted/superseded;

the question presupposes a relation the context contradicts.

Critical: an unanswerable row must be non-trivially unanswerable. It must not

be detectable by surface cues — same length, same register, same entity density, same

question phrasing distribution as answerable rows. If a classifier trained on surface

features alone can separate answerable from unanswerable above chance, the split is

broken.

Task type

task type

rows

note

alias\_resolution

1,200

entity introduced, later referred to by a different name/role/pronoun-chain; query uses one name, evidence uses the other

causal\_disambiguation

1,000

several candidate causes present; exactly one satisfies the stated criterion

rule\_exception

1,000

a general rule plus an explicit exception; the query targets a case where the exception governs

count\_with\_exclusion

800

count items matching a predicate while excluding an explicitly disqualified subset

temporal\_correlation

700

order/co-occurrence of events across the record

object\_state\_tracking

500

an entity's attribute changes across the record; query asks the state at a point

latest\_verified

400

several versions/updates; query asks for the most recent verified one

multi\_condition

400

the answer must satisfy 2–3 stated conditions simultaneously

Difficulty

60% hard, 40% medium. Hard means: ≥ 8 distractor entities, ≥ 2 near-miss

candidates that satisfy all but one criterion, and no lexical overlap between the

question wording and the evidence sentence.

THE LENGTH/COUNT PROBLEM — read this carefully

Rehearsal protection was measured to be a threshold in budget share, not a

gradient: a rehearsal component at 1,295 rows restored a capability completely,

and the same component at 648 rows restored nothing.

Long-context rows are 10–20× longer than the short rows that produced that result.

At a fixed \~108,000-token rehearsal budget, \~1,400-token rows buy only \~77 rows —

roughly 6% of the row count that worked.

It is unknown whether the threshold is in rows or in tokens. The corpus must let

that be tested. Therefore:

Emit a length\_band field on every row: short (300–600 tokens), medium

(600–1,100), long (1,100–1,600).

Generate 2,000 rows in each band, with the full stratification crossed within

each band.

The short band must still be genuine long-context retrieval — a real record with

distractors — not a toy.

This lets a downstream experiment hold tokens constant and vary rows, or vice versa.

Do not skip this; it is the single most useful property of the corpus.

PARTITION — before generation, not after

Assign immutable scenario graphs, entity pools, templates and lexicons to

train / dev / holdout before generating any row. A new random seed or a

renamed entity is not a partition. Paired variants of one scenario stay in the same

split. Emit split on every row: 80% train, 10% dev, 10% holdout.

OUTPUT FORMAT

JSONL, one object per line, UTF-8, LF endings:

{

"id": "lcr\_v1\_\<task\_type>*<8-hex>",*
*
"group\_id": "lcr\_v1\_scenario*<8-hex>",

"split": "train",

"messages": [

{"role": "system", "content": "\<instruction: answer from the record only; reply with the exact short answer, or UNKNOWN if the record does not contain it>"},

{"role": "user", "content": "\<the record, then the question>"},

{"role": "assistant", "content": "\<short exact answer OR UNKNOWN>"}

],

"metadata": {

"task\_type": "alias\_resolution",

"variant": "answerable",

"needle\_position": "early",

"needle\_char\_spans": [[412, 498]],

"length\_band": "long",

"difficulty": "hard",

"distractor\_entities": 11,

"near\_miss\_candidates": 2,

"unanswerable\_reason": null,

"answer\_type": "entity\_name",

"context\_tokens": 1387,

"total\_tokens": 1451,

"verified\_by": "\<checker name + version>",

"scenario\_seed": 20260916,

"template\_id": "alias\_chain\_v3"

}

}

needle\_char\_spans must give the exact character offsets of the evidence in the

user content. It is what makes the position claim auditable.

REQUIRED RECEIPTS — the corpus is not accepted without these

Emit a reports/ directory containing:

verification.json — every row independently re-derived by a checker that did

not generate it. Zero disagreements permitted; any disagreement is a build

failure, not a row to drop.

token\_profile.json — p50/p95/max total tokens under the real target

tokenizer, per length\_band and per task\_type, plus the count and reason for

every rejected row.

leakage.json — exact, NFKC-normalised, and digit-blinded overlap of contexts,

questions and answers against (a) the other splits and (b) any existing evaluation

suite. Report entity-pool, template and scenario-graph overlap separately from

text overlap. Structural overlap is the one that matters and is the one most

often missed.

shortcut\_floors.json — compute and publish, before any model sees this data:

majority-answer baseline, overall and per task\_type;

a "always answer UNKNOWN" baseline;

a "copy the most recent entity mentioned" baseline;

a "answer the first/last candidate in the record" baseline;

a question-only baseline (no context) and a context-only baseline.

If any of these clears 25% on the answerable rows, the corpus has a shortcut and

must be regenerated, not patched.

balance.json — the realised cross-tabulation of every stratum pair. Report

achieved counts, not targets.

ANTI-REQUIREMENTS

Do not generate answers with a language model and trust them. Answers come from

the generator's own ground truth and are re-checked independently.

Do not let the assistant turn explain, hedge, or restate. Short exact answers or

UNKNOWN.

Do not make unanswerable rows shorter, simpler, or differently phrased than

answerable ones.

Do not truncate to fit the window. Reject and record.

Do not reuse entity names, scenario templates or record formats from any

existing evaluation suite.

Do not balance by discarding rows after the fact. Generate to the target

distribution.

Do not report a nominal distribution; report the achieved one.

ACCEPTANCE

The corpus is accepted when: every row verifies; every row fits 2048 tokens under the

real tokenizer with ≥128 reserved; no shortcut baseline clears 25%; splits are

structurally disjoint; all six strata are balanced as specified within ±3%; and all

five receipts are present with real measured numbers.

No model has been trained on it and no quality claim attaches to it at delivery.

It is a candidate corpus until an arm measures it.