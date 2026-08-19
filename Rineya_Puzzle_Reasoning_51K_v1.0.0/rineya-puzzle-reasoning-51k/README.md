# Rineya Puzzle Reasoning 51K

A procedurally generated reasoning corpus for a puzzle-specialized Rineya edition. It contains **51,000 unique English reasoning problems**, concise solver-derived reasoning traces, compact state summaries, and short final answers.

The dataset is designed around a specific behavior:

> perform enough structured reasoning to solve the problem, then return a very short final answer.

No public benchmark examples are copied into the corpus. The families are original procedural generators inspired by established reasoning-evaluation themes such as epistemic logic, constraint satisfaction, rule deduction, state tracking, and search-based planning.

## Dataset totals

| Split | Rows | Intended use |
|---|---:|---|
| `train` | 40,800 | Model training |
| `validation` | 4,080 | Hyperparameter and checkpoint selection |
| `test` | 4,080 | Frozen in-distribution evaluation |
| `challenge` | 2,040 | Held-out phrasing and harder structural settings |
| **Total** | **51,000** | |

Every split contains all 15 families. Each family contributes 2,720 train, 272 validation, 272 test, and 136 challenge examples.

## Reasoning families

| Family | Core capability |
|---|---|
| `epistemic_hats` | Knowledge-of-knowledge, public announcements, red/blue-hat-style deduction |
| `knights_knaves` | Truth/liar consistency and assignment elimination |
| `prisoner_alibis` | Exact truth-count reasoning over suspects and prisoners |
| `mislabeled_boxes` | Permutation constraints and wrong-label deduction |
| `logic_grid` | Zebra/Einstein-style bijective constraint satisfaction |
| `linear_order` | Ranking, adjacency, position, and ordering deduction |
| `rule_deduction` | Forward chaining with explicit Yes/No/Unknown open-world semantics |
| `object_tracking` | Multi-step swaps, moves, rotations, and state updates |
| `code_breaking` | Mastermind-style exact-position and misplaced-digit constraints |
| `calendar_reasoning` | Temporal offsets, dates, weekdays, and constraint propagation |
| `family_relations` | Multi-hop kinship graph reasoning |
| `shortest_path` | Unique weighted shortest paths and first-hop planning |
| `arithmetic_constraints` | Integer systems expressed as compact word problems |
| `boolean_switches` | Boolean circuit evaluation and missing-switch inference |
| `hanoi_planning` | Legal state-space search and optimal Tower-of-Hanoi moves |

## What is stored in each row

Each base JSONL row includes:

- `prompt`: the natural-language puzzle.
- `reasoning_steps`: concise, solver-derived reasoning steps.
- `reasoning_summary`: the same reasoning in one string.
- `compact_state`: a compact representation of the solved or reduced state.
- `final_answer`: the canonical short response.
- `accepted_answers`: exact-answer variants accepted by the evaluator.
- `structure`: machine-readable puzzle state and constraints.
- `verification`: solver metadata such as candidate counts, proof depth, path distance, or solution count.
- `template_id`: identifies the surface-language template group.
- `structural_signature` and `prompt_sha256`: deduplication and integrity identifiers.

See `SCHEMA.md` for the complete schema.

## Load the data

The JSONL files can be streamed directly or loaded with Hugging Face Datasets. The generation and verification utilities require Python 3.10+ and NumPy; model training libraries are intentionally not pinned.

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/validation.jsonl",
        "test": "data/test.jsonl",
        "challenge": "data/challenge.jsonl",
    },
)
```

A readable example from every family is in `examples/FAMILY_PREVIEW.md`. Message-format samples are included for both reasoning-visible and short-only SFT.

## Integrity and verification

The release was independently re-solved after generation:

- **51,000 / 51,000 rows passed** family-specific symbolic verification.
- 51,000 unique IDs.
- 51,000 unique prompt hashes.
- 51,000 unique structural signatures.
- Train, validation, test, and challenge template groups are disjoint.
- Median final-answer length is one word; the 95th percentile is three words.

The machine-readable result is in `reports/validation_report.json`. Aggregate statistics are in `reports/dataset_stats.json`; prose-lint and end-to-end smoke-test results are in `reports/text_lint_report.json` and `reports/SMOKE_TESTS.md`.

## Build training views

The base files preserve all supervision while letting you choose how reasoning should appear during training.

### 1. Reasoning followed by a short answer

```bash
python scripts/build_training_views.py \
  --input data/train.jsonl \
  --output derived/train.think_sft.jsonl \
  --view think_then_short \
  --reasoning-tags think
```

The assistant target becomes:

```text
<think>
1. ...
2. ...
</think>
Short answer.
```

### 2. Final-answer-only SFT

```bash
python scripts/build_training_views.py \
  --input data/train.jsonl \
  --output derived/train.short_sft.jsonl \
  --view short_only
```

### 3. Separate reasoning and final targets

```bash
python scripts/build_training_views.py \
  --input data/train.jsonl \
  --output derived/train.dual.jsonl \
  --view dual_fields
```

This view is suitable for a custom trainer that applies separate masking or loss weights to `analysis_target` and `final_target`.

### 4. Concision preference pairs

```bash
python scripts/build_training_views.py \
  --input data/train.jsonl \
  --output derived/train.preference.jsonl \
  --view preference_pair
```

This produces a correct concise `chosen` answer and a correct but visibly verbose `rejected` response. Use it only after the model has learned to solve the problems; otherwise brevity optimization may suppress useful reasoning before it is acquired.

## Recommended Rineya curriculum

For a compact Rineya model, a staged curriculum is safer than training only on visible chains or only on final answers.

1. **Reasoning acquisition:** train on easy `think_then_short` examples so the model learns the operations and exact answer format.
2. **Mixed reasoning:** train on easy and medium examples using approximately 60–75% reasoned targets and 25–40% short-only targets.
3. **Response compression:** finish with short-only SFT so the public response remains concise.
4. **Optional preference tuning:** prefer the correct short answer over unnecessary visible explanation, while retaining a correctness gate.

A machine-readable starting point is included in `configs/recommended_curriculum.yaml`.

Important: explicit reasoning SFT does not itself guarantee that reasoning is neurologically “internal” or invisible. A raw model may emit the `<think>` block. To present only the short response, either use a runtime parser that hides the reasoning segment or complete the final-answer-only compression stage and evaluate whether reasoning accuracy survives.

## Evaluation

Prepare prediction JSONL rows with `id` and `prediction`:

```json
{"id":"rineya-pr51k-hat-tes-00001","prediction":"Blue."}
```

Then run:

```bash
python scripts/evaluate_predictions.py \
  --gold data/test.jsonl \
  --predictions predictions/test_predictions.jsonl \
  --report reports/model_test_report.json
```

The evaluator:

- extracts text after `</think>`, inside `<final>`, or after `Final:`;
- performs punctuation-, case-, and whitespace-normalized exact matching;
- reports accuracy by family, difficulty, and answer type;
- records coverage, missing IDs, duplicate IDs, and output length.

For a publishable Rineya result, keep `test` and `challenge` frozen. Do not train on them or repeatedly choose checkpoints against them. Report all-family macro results as well as per-family accuracy; a single aggregate can hide severe failures on epistemic or planning tasks.

## Reproduce the corpus

```bash
python -m pip install -r requirements.txt
python scripts/generate_dataset.py --output-dir . --seed 20260818
python scripts/validate_dataset.py --root . --expect-default-layout
python scripts/lint_dataset_text.py --root . --report reports/text_lint_report.json
```

The default generator deterministically recreates the 51,000-row split layout.

For a smoke test:

```bash
python scripts/generate_dataset.py --output-dir /tmp/rineya-smoke --tiny
python scripts/validate_dataset.py --root /tmp/rineya-smoke
```

## Suggested specialization name

This dataset can support a separately published branch such as **Rineya Reasoning**, **Rineya Puzzle**, or **Rineya Logic Edition**, while preserving the main Rineya model’s short-response identity.

## Limitations

- The corpus is synthetic and controlled. Strong performance does not prove equally strong reasoning in unrestricted natural language.
- Concise solver traces are supervision artifacts, not evidence that a trained model uses the same internal process.
- Test and challenge examples use held-out language templates, but they still belong to the same 15 generator families.
- Some tasks reward systematic enumeration. A model may learn useful algorithms, shallow template cues, or a mixture of both.
- Public demonstrations should include newly authored puzzles and external untouched benchmarks in addition to this corpus.

## License

The generator code and generated dataset in this package are released under Apache License 2.0. See `LICENSE`.
