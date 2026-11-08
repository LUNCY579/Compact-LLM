# Typed retention scoring contract, proposed v1

This is the parser specification for `RETENTION_GATE_DESIGN.md`, written before any new
candidate evaluation. It is not executable scorer code, human approval, a new source baseline,
or a qualification. The historical scorer and evidence remain unchanged.

The reference-only machine audit binds all 150 monitor rows to their original full messages.
Every numeric reference recomputes exactly. Four references name a different counted object
from the operative question; those corrections belong in a reviewed target overlay, never
in an in-place rewrite of historical data. See
`evidence/retention-design-v1/PROBE_LABEL_AUDIT.json` for the full question/recomputation packet.

## Common answer grammar

Consume the entire generated answer. Never extract a matching number from arbitrary prose,
ignore a second answer, or discard a contradictory unit. Preserve raw text and tokens for
every row. EOS termination is checked separately and is required for correctness.

Normalize outer whitespace, repeated whitespace and English letter case. Accept an optional
single final sentence period. Allow one optional prefix from this fixed list: `Answer:`,
`The answer is`, `It is`, `It will be`. Clock answers may additionally begin with `At`.
No other text is stripped. Negation, uncertainty, disjunctions, ranges and explanations are
outside this bounded grammar and score incorrect. This contract measures exact answers in
the declared format, not every semantically equivalent natural-language response.

Numbers use ASCII digits, an optional sign, and exact rational arithmetic. Accept integers,
finite decimal notation (including `.5`), or one numerator/positive-denominator fraction.
Allow comma thousands separators only in valid groups of three. Decimal trailing zeros do
not change value. Reject NaN, infinity, scientific notation, arithmetic expressions, division
by zero, mixed fractions and malformed separators. No floating-point tolerance is used.

Each target declares its kind, exact rational value or clock minute, canonical unit,
permitted unit aliases, optional entity qualifier, and whether a bare value is meaningful
in the question's requested unit. These declarations are frozen before source or candidate
measurement. Matching numeric values are insufficient when explicit types/units disagree.

## Target kinds

| Kind | Canonical representation and accepted units |
|---|---|
| Number | Exact rational; no money/percent/measurement suffix |
| Generic quantity | Exact rational in `unit`; `unit`/`units` |
| Count | Nonnegative exact integer plus the prompt's counted object; integer-valued decimals/fractions are allowed; explicit noun aliases below |
| Money | Exact rational dollars; `$`, `dollar`, `dollars`. National currency is unspecified, so do not invent USD or currency exchange |
| Percent | Exact rational percentage points; `%`, `percent`, `per cent`. A bare value means percentage points when allowed, not a proportion |
| Distance | Exact rational kilometers; `km`, `kilometer(s)`, `kilometre(s)`; meters may convert exactly at 1,000 meters/km |
| Cup volume | Exact rational cups; `cup(s)` only. Cup size is unspecified, so no liter/milliliter conversion |
| Duration | Exact rational seconds; `second(s)`/`sec(s)`/`s`, `minute(s)`/`min(s)`, `hour(s)`/`hr(s)`/`h` |
| Clock | Integer minutes since midnight in a 24-hour day |

For the current monitor, a bare numeric value is proposed as acceptable only where the
operative prompt fixes the requested quantity/unit, with the decision recorded per row.
In particular, only three of the 19 duration questions explicitly ask for minutes:
`time_duration_s04__01221`, `time_duration_s07__02229`, `time_duration_s07__02318` (suffixes
of their full source IDs). The other 16 duration answers require an explicit unit: a bare
`90` or `1.5` does not identify minutes versus hours. No new output-unit instruction is
silently inserted into the prompt. Number targets are intrinsically unitless.

Explicitly wrong units fail. For a colored count or a water volume, omission of the qualifier
uses the question's qualifier; an explicitly different qualifier fails. Blue count syntax
is `number blue noun-alias`, with blue preceding the entire alias. Water volume syntax is
`number cup(s) of water`; bare `number cup(s)` can use the implicit water qualifier. Other
qualifier positions are outside the bounded grammar. Unknown aliases are not invented after
looking at model outputs.

Proposed count aliases, before source measurement:

| Prompt entity | Permitted noun strings |
|---|---|
| sensor tags | sensor tag, sensor tags, tag, tags |
| timing chips | timing chip, timing chips, chip, chips |
| shipping labels | shipping label, shipping labels, label, labels |
| folding chairs | folding chair, folding chairs, chair, chairs |
| drill bits | drill bit, drill bits, bit, bits |
| page | page, pages |
| access badges | access badge, access badges, badge, badges |
| safety helmets | safety helmet, safety helmets, helmet, helmets |

Clock syntax accepts `h:mm AM/PM` (case-insensitive; the permitted meridiem strings are
`am`, `pm`, `a.m.`, `p.m.`, `a.m`, `p.m` after whitespace/letter-case normalization),
or an unambiguous 24-hour `H:mm`. A missing meridiem means the latter, not a guessed AM/PM.
An explicitly suffixed whole hour such as `7 PM` means `19:00`. Hours and minutes must be in
range; `24:00`, `13:20 PM`, and a naked hour are rejected. `12 AM` is midnight and `12 PM`
is noon. Bare `noon` and `midnight` are accepted at their exact clock values. No timezone,
date, or extra day offset is silently discarded.

Durations accept one quantity/unit or a compound of distinct units (hours, minutes, seconds)
in any order, joined by whitespace, a comma or `and`. Sum the units exactly.
A bare value is allowed only with the row's explicit implicit-unit permission. Reject
negative durations, repeated compound units, unknown units, alternatives and extra text. Clock times and
durations are different types even when their number sequences coincide.

## Counterexample fixtures to implement before measuring source

These synthetic examples specify parsing semantics, not new arm observations. Every row
must be tested in the future authoritative scorer, with exact parsed values and error reasons.

| Typed target | Prediction | Correct? |
|---|---|---|
| number 3 | `3.` | yes |
| number 62.5 | `The answer is 62.500.` | yes |
| number 0.5 | `1/2` | yes |
| number 1000 | `1,000` | yes |
| number 1000 | `10,00` | no |
| number 5 | `not 5` | no |
| number 5 | `5 or 6` | no |
| number 5 | `5, but actually 6` | no |
| number 5 | `5 hours` | no |
| number 5 | `$5` | no |
| number 5 | `5%` | no |
| number 5 | `NaN` | no |
| number 5 | `1/0` | no |
| number 5 | `5e0` | no |
| number 5 | `5?` | no |
| number 5 | `5...` | no |
| 54 sensor tags; bare allowed | `54` | yes |
| 54 sensor tags | `54 sensor tags.` | yes |
| 54 sensor tags | `54 spare fuses.` | no |
| 15 blue folding chairs; `chairs` alias declared | `15 blue chairs` | yes |
| 15 blue folding chairs | `15 red folding chairs` | no |
| 40 dollars; bare allowed | `40` | yes |
| 40 dollars | `$40.00` | yes |
| 40 dollars | `40 euros` | no |
| 40 percent; bare allowed | `40` | yes |
| 40 percent | `40%` | yes |
| 40 percent | `0.4` | no |
| 3.2 cups | `16/5 cups` | yes |
| 3.2 cups | `3.7 cups` | no |
| 3.2 cups | `0.8 liters` | no |
| 12 cups of water | `12 cups of water` | yes |
| 12 cups of water | `12 cups of oil` | no |
| 5 kilometers | `5000 meters` | yes |
| 5 kilometers | `5 miles` | no |
| duration 45 minutes; bare unit minute | `45` | yes |
| duration 45 minutes | `0.75 hours` | yes |
| duration 45 minutes | `2700 seconds` | yes |
| duration 75 minutes | `1 hour and 15 minutes` | yes |
| duration 75 minutes | `15 minutes and 1 hour` | yes |
| duration 90 minutes; prompt requests elapsed time without a unit | `90` | no |
| duration 45 minutes | `45 hours` | no |
| duration 45 minutes | `45 minutes or 1 hour` | no |
| clock 10:25 AM | `10:25 PM` | no |
| clock 10:25 AM | `10:55 AM` | no |
| clock 4:20 PM | `16:20` | yes |
| clock 4:20 PM | `4:20` | no |
| clock 12:00 AM | `midnight` | yes |
| clock 12:00 PM | `12 PM` | yes |
| clock 12:00 PM | `12 AM` | no |
| clock 1:20 PM | `13:20 PM` | no |
| clock 12:00 AM | `24:00` | no |

Malformed prediction syntax is an incorrect answer with a recorded parse reason. A malformed
or ambiguous *target specification*, changed row binding, missing output or nonfinite model
computation invalidates the evaluation itself; it is not an ordinary wrong-answer count.

## Implementation and review boundary

Implement one pure parser plus exact typed comparison under the hashed package. Use the same
implementation for source, initialization and every training checkpoint. Save full predictions,
typed parses, correctness, termination, and target/scorer hashes. Check that caller-supplied
correctness cannot override recomputation. Unknown target kinds must fail closed.

This contract deliberately excludes some valid free-form expressions. The source baseline
must therefore be measured anew under this exact grammar, and the certificate must name its
format/type scope. If later evidence reveals a scorer bug, preserve previous outputs and
version/requalify the scorer and baselines; never quietly change an alias or threshold to
make an observed candidate pass.

Human review of the target overlay, multi-turn correction precedence and alias lists remains
pending. The reference audit is complete machine arithmetic checking, not human attestation.
