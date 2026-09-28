# CC0 long-context corpus: audit verdict — DO NOT USE, re-commission

**`rineya_long_context_cc0_v1_main`, 50,016 SFT rows + 10,000 DPO rows, 1.7 GB.
Audited against the commission in `docs/LCR_GENERATION_PROMPT.md`.**

The corpus has real strengths and one root-cause error that makes it unusable as
delivered. It should be regenerated, not discarded: the generator, the anchor cards
and the rights work are sound.

## What is good, and should be kept

- **Contamination is clean.** Zero overlap with every quarantined instrument, on
  every axis checked: against the 150-row arithmetic panel — 0 id, 0 `reply_class`,
  0 normalised 8-gram; against `Rineya_Long_Context_Demo_v1` — 0 domain, 0 task-name,
  0 8-gram; against the v3 preservation slice — 0 id. The 7 "shared proper nouns"
  the fingerprint flagged are `After`, `First`, `Give`, `Harbor`, `North`, `What`,
  `Which` — artefacts of a naive capitalisation regex, not entity collisions.
- **Provenance is auditable.** A real source registry, CC0/public-domain rights
  policy, anchor and derivative card ids on every row, and per-row lineage records.
- **Scale and balance.** 12 capability groups at 3,668 rows each in train, with
  `calibrated_abstention` and `entity_resolution` present.
- **Honest self-reporting.** `human_review_status: machine_validated_human_review_pending`
  on every row, and the lineage files record the very numbers that condemn it.

## Blocker 1 — zero rows fit the qualified window, by census

| measurement | value |
|---|---|
| rows fitting 2048 total, prompt + 128 ≤ 2048 | **0 of 50,016** |
| shortest full rendered sequence in the corpus | **2,551 tokens** |
| shortest prompt | 2,546, against a 1,920-token ceiling |
| my own cross-shard sample (12,800 rows, all 32 shard files) | min **2,558**, p50 5,938, max 54,491 |

Measured with the real Supa tokenizer (`Qwen2TokenizerFast`, transformers 4.57.1).
`compile_row` and `compile_dialogue` both raise
`OVERLENGTH: no truncation, silent skip or evidence removal`
(`serialization.py:84-85`) on **every** row. There is no subset to select and no
filter to write. The subset is empty, not small.

**Root cause:** the generator sized the **context** to the stratum nominal
(`STRATUM_NOMINAL['2k'] = 2048`, `scripts/corpus_core.py:39`) using a regex
word-splitter, not the **rendered sequence**. The commission asked for total
rendered ≤ 2048 with context 1,100–1,600. The smallest stratum therefore starts at
roughly the window ceiling before the system prompt, chat template, question and
answer are added.

**It is structural, not a tuning constant.** The mandatory 16-entry evidence
register costs ~2,036 tokens on its own (126.3 tokens/entry, 16 entries in
50,016/50,016 lineage records) — more than the entire 1,920-token prompt budget, in
a row with zero padding. Reaching an instrument-matched band needs roughly **8
entries**, which is a record-schema change.

## Blocker 2 — the abstention split is separable from the question alone, at 100%

I verified this over the full population:

| | rows containing the literal `certified archival shelf date` |
|---|---:|
| unanswerable (`reply_class: unknown`) | **4,168 / 4,168** |
| answerable | **0 / 45,848** |
| question-only classifier accuracy | **1.0000** |

The phrase is hard-coded (`scripts/corpus_core.py:420`, `scripts/main_templates.py`)
and is carried into the sealed eval bank (`scripts/sealed_templates.py:74`).

The commission's own acceptance test forbids exactly this: *"If a classifier trained
on surface features alone can separate answerable from unanswerable above chance,
the split is broken."* A model rehearsed on this learns the phrase, not abstention —
and abstention is the single most fragile capability measured (13/64 → 0–3/64).

## Blocker 3 — it rehearses the wrong tasks

Rehearsal protects **only what it resembles** (`REHEARSAL_COMPOSITION.md`). This
corpus resembles neither the instrument's tasks nor its surface form.

- **Task overlap with the instrument: empty.** The suite scores `alias_resolution`,
  `object_state_tracking`, `temporal_correction`, `rule_exception`,
  `count_with_exclusions`, `causal_disambiguation`, `multi_condition_identity`,
  `latest_verified_record`. The corpus implements twelve *different* families
  (`fact_lookup`, `comparison`, `linked_fact`, …). The intersection is **∅**.
  `causal_disambiguation` and `rule_exception` — the two flagged as degrading
  reliably — have no analogue at all.
- **Surface form is a key=value register, not prose.** Instrument contexts render at
  5.11 chars/token; corpus user turns at 3.43 — a 1.49× token-density penalty, with
  20× the `;=:|/` density and 12× the digit density. Short *conversational*
  rehearsal already made long context **worse** (161 → 118); form mismatch is a
  known failure mode here, not a cosmetic quibble.

## Blocker 4 — the acceptance test was never run

`reports/main_lineage/*.jsonl` records `context_proxy_tokens` for all 50,016 rows.
**Zero** are ≤ 1,920. Per-stratum minima: 2k → **2,012**, 4k → 4,060, 8k → 8,156,
16k → 16,348, 32k → 32,732. And that is *context only*, before rendering.

The failure was visible in the supplier's own lineage file before delivery, and
`LCR_Input_Preflight/PREFLIGHT_STATUS.json` records
`target_tokenizer_corpus_measurement: NOT_RUN`. **Any future delivery needs a
mechanical, tokenizer-based pre-delivery gate**, not a proxy word count.

## Verdict

**DO_NOT_USE.** Re-commission a regeneration against the corrected spec in
`docs/LCR_GENERATION_PROMPT_V2_ADDENDUM.md`. Keep the generator, the anchor cards,
the rights work and the contamination discipline; discard the 1.7 GB of rows.

Not recommended as a workaround: waiving the window to 4096. At 4096 roughly 39.6%
of rows would fit (~17,426 rows, comfortably above the 1,295-row threshold), but
that requires amending the frozen design contract
(`RRP006_WINDOW_4096_BLOCKER.md`) **and** would still leave blockers 2 and 3 intact.
Spending a contract amendment on a corpus that rehearses the wrong tasks with a
broken abstention split buys nothing.

The DPO split (10,000 rows) is moot regardless: this programme has never run
preference optimisation and has no DPO code path.

## Method note

The `audit:leakage` agent in the audit workflow died on an API content-filter error;
that dimension was re-run directly and its numbers above are mine. The window census,
the abstention separator and the lineage recount were each independently reproduced
before being reported.
