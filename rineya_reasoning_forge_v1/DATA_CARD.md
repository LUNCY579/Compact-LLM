# Data card — Rineya Reasoning Forge 1.0.0

**Created:** 2026-09-11. **Language:** English. **Status:** machine-validated candidate, not model-validated. **Generation:** original deterministic procedural Python; no external model inference or imported data rows.

## Composition

Each of 16 families contributes 500 training, 50 validation, 50 IID test, 20 transfer test, and 5 long-input stress rows: **625 per family, 10,000 total**. There is one canonical question per world. Group identifiers equal canonical row identifiers. Derived layout views share their parent test world's group identifier.

The 512-example pilot and 128-example development pilot are existing short-prompt subsets with balanced family counts. Training layouts are balanced across early, middle, late, and distributed core placement: 2,000 rows each. For short prompts, fixed instructions and large core blocks can prevent literal placement entirely inside a particular third of the full prompt; measured character locations, rather than a fictional token position, are retained in metadata.

Training word-tier targets: 2,000 rows around 256 words, 2,400 around 512, 1,600 around 1,024, 1,200 around 2,048, and 800 around 4,096. These count the user content, not the system message or assistant response. The exact realized range and totals appear in the build report. Whitespace words are not model tokens.

## Construction and verification

1. Sample a fictional world, task parameters, and an independently specified expected answer.
2. Render natural/controlled-language facts and rules with arbitrary record identifiers.
3. Add both same-schema distractors and original scene prose. Record identifiers and content length do not label a record as relevant.
4. Distribute core records across the context and place the question at the end.
5. Recompute the reference using a separate solver that parses the rendered prompt. Reject any disagreement.
6. Reject previously generated exact canonical cases. Preserve source hashes, layout positions, length measures, and generation parameters outside the training columns.
7. Re-read the written files through the independent release validator.

The two implementations reduce some common-mode risk but are not equivalent to independently audited formal proofs. Parser and generator authorship is shared. A human should review a stratified sample before a substantial run.

Core-record annotations identify the sampled task world, not a guaranteed minimum proof. Several core records can be unnecessary to a particular answer. Long contexts are often short reasoning problems embedded in many distractions; they are not proof that solving the task requires reasoning about every paragraph.

## Shortcut protections and remaining weaknesses

Both positive and negative target-conclusion rules occur in every deductive instance. A disabled proof is missing a prerequisite, replaced by a decoy fact; the answer cannot be obtained merely by detecting the conclusion rule. Relational unknown cases retain the same number of primary records, replacing a necessary edge with a nonmatching one. Scheduling classes use overlapping constraint-count ranges. Truth-puzzle contradictions use feasible honesty counts and inconsistent statements, not impossible population sizes.

Label/position crossing, nonchronological records, future/retracted drafts, aliasing, signed values, and alternative entities provide additional controls. These measures do not eliminate all possible shortcuts. Fixed fact grammars, repeated scene templates, finite vocabulary, and a limited scenario bank remain substantial limitations. There is no claim of template-free generalization or human-novel realism.

## Transfer and length evaluation

Transfer instances increase deduction/relational depth, candidate count, ledger or program length, graph size, truth-puzzle size, or Hanoi disk count, depending on family. Summary transfer uses four target scenario templates excluded from training summary prompts and distractors. Communication transfer uses held-out event descriptions. Source-boundary transfer adds new quoted-attack forms.

The transfer split is not a universal topology-heldout benchmark: many mechanisms and rendering grammars are shared with training. Its sample size is only 20 per family. The length stress set is only five per family and should not support precise broad claims.

The 160 derived probes use 32 IID test worlds with five views each. Four long views share precisely the same records and change only layout. The short view preserves all core records while reducing distractors. Analyze paired outcomes by parent world. These 160 views do not increase the number of independent worlds by 160.

All published test data and labels are developer-visible. They are holdouts for the proposed training split, **not cryptographically sealed or independently administered benchmarks**. Keep the user's existing sealed evaluations untouched. A private external overlap audit has not been run.

## Natural-language tasks

Summary references preserve a requested goal, obstacle, adopted action, and confirmed or explicitly unconfirmed outcome. Communication references satisfy a small set of polite-decline, clarification, correction, and progress-update schemas. Mechanical checking covers these recorded slots and format limits. Human fluency, varied phrasing, nuanced social judgment, and general semantic quality remain unreviewed.

The provided scorer deliberately does not combine free-form lexical slot coverage with exact-answer accuracy. Literal slot presence can miss a valid paraphrase and can be fooled by a response that includes both the right fact and an unsupported claim. Use human/rubric scoring for these families.

## Intended use and exclusions

Appropriate: assistant-loss-only skill SFT, small controlled recovery adjuncts, oracle/debugging tests, matched context-layout comparisons, and expanding the generator with new audited schemas.

Not appropriate: sole broad language recovery corpus; training on evaluation files; replacing an untouched benchmark with a public procedural holdout; declaring CED superiority from this corpus alone; inferring real people's honesty from the formal truth-teller puzzles; treating fictional case records as real facts; or assuming words equal tokens.

No model was trained, no real-model tokenizer was run, no model quality score was measured, and no GPU fit was established in this release.
