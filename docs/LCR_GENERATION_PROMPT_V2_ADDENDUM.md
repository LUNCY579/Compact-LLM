# LCR generation — corrected spec (v2 addendum)

Hand this to the generating agent **together with** `LCR_GENERATION_PROMPT.md`. Where
the two disagree, **this document wins**. It exists because
`rineya_long_context_cc0_v1_main` was built against v1 and failed on four points,
each of which is now stated unambiguously. See `LCR_CC0_AUDIT.md` for the evidence.

## C1. The budget is the RENDERED SEQUENCE, not the context

The v1 delivery sized **contexts** to a stratum nominal of 2,048 and shipped rows
whose shortest *rendered* sequence was 2,551 tokens. Zero of 50,016 rows were usable.

**Binding definition.** Let `S` be the full chat sequence produced by
`tokenizer.apply_chat_template([system, user, assistant], tokenize=True,
add_generation_prompt=False, enable_thinking=False)` and `P` the same call over
`[system, user]` with `add_generation_prompt=True`.

- `len(S) <= 2048` — **hard**
- `len(P) <= 1920` — **hard** (128-token completion reserve)
- Target `len(P)` in **1,200–1,700**. Do not aim at the ceiling.

**A proxy word-count is not a measurement.** Every row must be measured with the
supplied Qwen tokenizer (`Qwen2TokenizerFast`, the four hash-locked assets in
`LCR_Inputs.zip`). A delivery whose own manifest does not record real
tokenizer-measured `len(S)` and `len(P)` per row will be rejected without review.

**Emit these fields per row:** `rendered_total_tokens`, `rendered_prompt_tokens`,
`completion_tokens`, `unused_capacity = 1920 - rendered_prompt_tokens`.

## C2. Fixed-cardinality scaffolding must fit the budget

v1's evidence register was **16 entries at ~126 tokens each = ~2,036 tokens**, larger
than the whole prompt budget, and its cardinality was fixed corpus-wide. No length
constant could fix that.

**Requirement.** Every fixed structure must be sized so that, at its maximum
cardinality, `len(P) <= 1920` still holds with ≥ 200 tokens of headroom. Practically
this means **≤ 8 register entries**, or shorter entries. Vary cardinality across rows
rather than fixing it. Report the token cost of the scaffolding separately from the
evidence-bearing content.

## C3. Match the instrument's FORM, not just its length

Rehearsal protects only what it **resembles**; short conversational rehearsal made
long context *worse* (161 → 118). v1 shipped semicolon-delimited `key=value` register
tables with hex ids: 3.43 chars/token against the instrument's 5.11, with 20× the
punctuation density and 12× the digit density.

**Requirement: flowing natural prose.** A record should read like a report, a
chronicle, a case file or a dossier written for a human.

**Measurable acceptance gate:**
- chars/token **≥ 4.6** (instrument is 5.11)
- density of `;=:|/` **≤ 0.010** (instrument 0.0028)
- digit density **≤ 0.020** (instrument 0.0062)

Report all three per row and as a corpus distribution. Identifier-shaped tokens
(`M-TR-BU-030-14`, `ANCH-GAMES-03-A`) must not appear in model-visible text; keep
them in metadata.

## C4. Generate the instrument's OWN task types

v1's task-name intersection with the instrument was **empty**. Generate exactly
these eight, and use these names:

| task type | rows | why this weight |
|---|---:|---|
| `alias_resolution` | 1,200 | 4/32 untrained — near-absent |
| `causal_disambiguation` | 1,000 | degrades reliably, 31 → 12 |
| `rule_exception` | 1,000 | degrades reliably, 15 → 8 |
| `count_with_exclusions` | 800 | |
| `temporal_correction` | 700 | |
| `object_state_tracking` | 500 | robust — coverage only |
| `latest_verified_record` | 400 | robust — coverage only |
| `multi_condition_identity` | 400 | robust — coverage only |

Do **not** invent a parallel taxonomy. If a family cannot be built, say so and leave
it empty rather than substituting a different one.

## C5. The abstention split must not be detectable from the question

v1 hard-coded the literal `certified archival shelf date` into **4,168 of 4,168**
unanswerable questions and **0 of 45,848** answerable ones. A one-keyword,
question-only classifier scored **1.0000**.

**Requirements.**
- No lexical marker may correlate with answerability. Draw question templates from
  the **same pool** for answerable and unanswerable rows.
- Match the distributions of question length, entity count, context length and
  register between the two classes.
- Balance the four unanswerability mechanisms (entity absent; attribute absent;
  retracted/superseded; presupposition contradicted).
- **Run the adversary yourself before delivery:** fit a question-only classifier
  (bag-of-words + length) and an exhaustive single-threshold search over every
  surface feature. Report balanced accuracy for each. **Anything above 0.55 is a
  build failure**, not a row to drop.
- Unanswerable rows answer the literal string `UNKNOWN` and nothing else.

Target **30%** unanswerable.

## C6. Pre-delivery gate — mechanical, not narrative

Ship `reports/acceptance.json` containing, computed with the supplied tokenizer over
**100% of rows** (not a sample):

```json
{
  "rows_total": 0,
  "rows_rendered_total_le_2048": 0,
  "rows_rendered_prompt_le_1920": 0,
  "rendered_total_tokens": {"min": 0, "p50": 0, "p90": 0, "max": 0},
  "rendered_prompt_tokens": {"min": 0, "p50": 0, "p90": 0, "max": 0},
  "chars_per_token": {"min": 0, "p50": 0},
  "punct_density_p90": 0, "digit_density_p90": 0,
  "task_type_counts": {},
  "unanswerable_fraction": 0,
  "question_only_classifier_balanced_accuracy": 0,
  "best_single_surface_feature_balanced_accuracy": 0,
  "shortcut_floors_on_answerable": {"majority": 0, "always_unknown": 0,
                                    "copy_last_entity": 0, "first_candidate": 0,
                                    "last_candidate": 0, "question_only": 0},
  "quarantine_overlap": {"panel_ids": 0, "panel_8gram": 0,
                         "demo_domains": 0, "demo_8gram": 0, "preservation_ids": 0}
}
```

**Acceptance is `rows_rendered_total_le_2048 == rows_total`.** If it is not, do not
ship; report the histogram and ask.

## C7. Size and length bands

**6,000 rows** is sufficient and preferred over 50,000. Protection is a threshold in
budget share — 1,295 rows of a component worked, 648 did not — so a few thousand
usable rows beat tens of thousands of unusable ones.

Three bands by `rendered_prompt_tokens`, 2,000 rows each, half-open:
`short [600, 1000)`, `medium [1000, 1400)`, `long [1400, 1920]`.
Emit `length_band`. The full task/variant/position stratification crosses **within**
each band. This is what lets a downstream arm hold tokens constant and vary rows.

## C8. Unchanged from v1, and still required

Needle position with 40% early; `needle_char_spans` (empty for absent-entity rows,
with a separate field for related/retraction spans); single-turn
`system`/`user`/`assistant` only; independent re-derivation of every answer;
structural split partitioning before generation; and the full receipt set.

## What v1 got right and should be preserved

Contamination discipline (zero overlap with every quarantined instrument on every
axis), the CC0 source registry and rights policy, per-row anchor/derivative lineage,
and honest `human_review_status` reporting. Keep all of it.
