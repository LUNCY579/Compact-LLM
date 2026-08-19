# Rineya LCR SFT v1

**Closed-context long-document reasoning corpus for supervised fine-tuning**

This package contains a synthetic, invented-world corpus designed to teach a model to read a long supplied document, bind multiple facts, and return a factually complete answer. It is correctness-first: response length is not scored, and a necessary name, serial, quantity, condition, route, or `UNKNOWN` must not be omitted merely to make the answer shorter.

The corpus is **train/dev only** and is disjoint from the frozen `Rineya_Long_Context_Demo_v1` evaluation pack. It uses `LCR-*` identifiers and does not contain or continue any `RLC-*` scenario.

## Contents

| Path | Purpose |
|---|---|
| `data/lcr_sft_v1.jsonl` | Complete corpus: 2,000 packs / 8,000 rows |
| `samples/lcr_sft_v1_sample_12_packs.jsonl` | Twelve complete packs, one from each domain |
| `samples/sample_index.json` | IDs and coverage of the sample |
| `scripts/generate_lcr_sft_v1.py` | Deterministic corpus generator |
| `scripts/validate_lcr_sft_v1.py` | Structural, semantic-field, uniqueness, length, and v1-overlap validator |
| `scripts/score_predictions.py` | Correctness-first prediction scorer |
| `reports/generation_progress.log` | Required one-line progress record every 50 packs |
| `reports/generation_summary.json` | Generation counts and balance summary |
| `reports/validation_report.json` | Full validation result |
| `checksums/MANIFEST.sha256` | SHA-256 checksums for package files |

The generator and utility scripts use only the Python standard library.

## Corpus scale

- **2,000 complete scenario packs**
- **8,000 JSONL rows**
- Four rows per pack: `base`, `paraphrase`, `counterfactual`, and `unanswerable`
- **1,800 train packs / 7,200 train rows**
- **200 dev packs / 800 dev rows**
- Split assignment is at pack level; the four variants never cross splits
- IDs run continuously from `LCR-0001-B` through `LCR-2000-U`
- **8,000 unique prompts** and **2,000 unique titles**
- **2,000 unique base contexts**
- **2,000 unique plot signatures**, using a distinct evidence/plot mechanism for every occurrence of a domain–task pair

## Pack semantics

Each `scenario_id` has exactly four coordinated rows:

| Variant | Context and question behavior | Gold behavior |
|---|---|---|
| `base` | Canonical invented document; direct question | Answer is determined by two or more bound facts |
| `paraphrase` | Context is character-identical to `base`; question is reworded | Same answer as `base` |
| `counterfactual` | Same story with one decisive controlling fact changed | Different correct answer |
| `unanswerable` | The decisive mapping is damaged, missing, disputed, or unsigned | Answer begins with `UNKNOWN` |

The validator confirms that base/paraphrase contexts are byte-identical, base/counterfactual answers differ, and the counterfactual and unanswerable contexts each alter only the designated decisive paragraph relative to the base context.

## Document design

Every context is wholly invented and combines several of the following:

- Unique people, aliases or call-signs, and job titles
- Invented internal addresses, rooms, and routes
- Unique object serials and confusable neighboring serials
- Dates, times, quantities, and planned-versus-completed actions
- Local definitions that override ordinary English, such as a file-specific meaning of “cleared”
- Rule hierarchies and explicit exceptions
- Later corrections that supersede earlier notes
- Signed versus unsigned evidence
- High-value distractors such as similar names, duplicate scans, proposed destinations, intermediate locations, and rumors

A correct answer requires binding at least two facts, such as identity plus alias, route code plus destination key, local definition plus exclusions, or authority rule plus exception. The decisive evidence is balanced across early, middle, late, and distributed needle positions.

## Coverage

### Domains

The 12 invented domains are balanced at 166–167 packs each:

- `academy_chronicle`
- `detective_file`
- `station_log`
- `expedition_journal`
- `museum_conservation`
- `lab_chronology`
- `civic_record`
- `estate_business_saga`
- `clinic_inventory` — logistics/inventory only, with no medical advice
- `warehouse_logistics`
- `archive_catalog`
- `ships_log`

### Task families

Each task family has exactly 200 packs:

- `alias_resolution`
- `object_state_tracking`
- `temporal_correction`
- `rule_exception`
- `count_exclusions_duplicates`
- `causal_disambiguation`
- `multi_condition_identity`
- `latest_verified_record_vs_rumor`
- `definition_binding`
- `address_route_tracking`

Every domain–task combination appears 16 or 17 times.

### Length and needle balance

| Bucket | Packs | Rows | Context-word range | Mean |
|---|---:|---:|---:|---:|
| `compact_long` | 800 | 3,200 | 1,150–1,231 | 1,180.587 |
| `standard_long` | 800 | 3,200 | 1,300–1,572 | 1,406.991 |
| `extended_long` | 400 | 1,600 | 1,699–2,177 | 1,915.977 |

Across the full corpus, context length is 1,150–2,177 words, with a mean of 1,418.227 and a median of 1,335 words. Each needle profile—`early`, `middle`, `late`, and `distributed`—contains exactly 500 packs.

## Row format

Each UTF-8 JSONL line follows the requested schema. Important relationships are:

- `prompt == "CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"`
- `messages[1].content == prompt`
- `messages[2].content == answer`
- The system message is identical in every row
- `answerable` is `false` only for `unanswerable`
- `context_words` is the exact whitespace-token word count of `context`
- `accepted_answers` contains the canonical answer and multiple fully correct paraphrases
- `supporting_facts` contains two to six context-grounded facts and is verified against that row’s own context

The fixed system message is:

> Use only the supplied context. Answer correctly and completely. Do not guess. If the context does not determine the answer, reply UNKNOWN. Brevity is optional; correctness is required.

## Correctness-first evaluation

The intended evaluation does **not** reward brevity. A prediction is accepted automatically when either:

1. Its normalized text exactly matches one of `accepted_answers`; or
2. It contains every `must_include` string, contains none of the `must_not_include` strings, and is nonempty.

The second route is a lexical approximation. Because arbitrary semantic contradictions cannot be identified perfectly with substring rules, `score_predictions.py` reports lexical-only passes separately for optional human review.

Prediction JSONL example:

```json
{"id":"LCR-0001-B","prediction":"The final verified location of navigation wafer ST-000-020 was Ilyevomin maintenance cell A-17."}
```

Run the scorer:

```bash
python scripts/score_predictions.py \
  data/lcr_sft_v1.jsonl \
  predictions.jsonl \
  --report reports/prediction_scores.json
```

Reported metrics include overall and per-variant accuracy, all-four-variants-correct pack rate, base/paraphrase joint correctness, counterfactual sensitivity, abstention accuracy, and breakdowns by split, domain, and task.

## Validation

From the package root:

```bash
python scripts/validate_lcr_sft_v1.py \
  data/lcr_sft_v1.jsonl \
  --expected-packs 2000 \
  --v1 /path/to/Rineya_Long_Context_Demo_v1/data/rineya_long_context_demo_full.jsonl \
  --report reports/validation_report.json
```

The completed validation returned:

- `valid: true`
- **0 errors**
- **0 warnings**
- 2,000 packs / 8,000 rows
- All ID, schema, prompt, message, answer, four-variant, split, word-count, uniqueness, balance, and supporting-fact checks passed
- No forbidden v1 names or `RLC-*` IDs
- **0 exact 20-word shingle overlaps** with the frozen v1 JSONL
- Main JSONL SHA-256: `8f0bc0174408eb513851443a6f9c32e2e79a00563fb9d35d659e7d27ab792c9c`

The shingle test is a conservative exact-sequence leakage check; it does not claim to measure every possible semantic resemblance. The corpus was generated without web access and uses invented entities and events.

## Reproduction

Generation is deterministic with seed `271828`:

```bash
python scripts/generate_lcr_sft_v1.py \
  --output data/lcr_sft_v1.jsonl \
  --packs 2000 \
  --seed 271828 \
  --log reports/generation_progress.log \
  --summary reports/generation_summary.json
```

The generator writes each JSONL row directly to disk and emits one progress line after every 50 completed packs. It refuses repeated titles, people, sites, object serials, prompts, base-context hashes, or plot signatures.

## Training guidance

Treat `split == "dev"` as a development set and never mix those packs into training. Because all four variants of a scenario are intentionally related, batching or sampling should preserve pack-level split boundaries. The frozen `Rineya_Long_Context_Demo_v1` should remain untouched and should be used only as an external demonstration/evaluation set, not as training data.

For SFT, `messages` is ready for chat-template processing. Retaining the complete context and question is essential; truncating from the middle can remove the controlling evidence. Inspect exact tokenizer lengths for the chosen Rineya checkpoint before selecting `max_seq_length`.
