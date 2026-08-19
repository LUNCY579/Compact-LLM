# Rineya Sequence-Independence Reasoning Dataset v1.0

A paired synthetic dataset for testing whether a model can learn useful reasoning behavior from **independent compact-state transitions** instead of receiving every solution as one uninterrupted reasoning trajectory.

## Research question

Let a complete solution history at step `t` be `H_t`, and let `s_t` be a compact state derived from that complete trajectory. The central hypothesis is:

```text
P(next reasoning step | original problem, complete prior history)
≈
P(next reasoning step | original problem, compact current state)
```

The experiment compares two adapters trained from the same base checkpoint:

1. **Full-context adapter** — one original problem maps to one complete reasoning response.
2. **Split-transition adapter** — the same problem maps to multiple independently sufficient state-to-next-step records. Those records are globally shuffled, so chronological row adjacency is intentionally unavailable.

This is not a brevity experiment and not a training-speed experiment. The variable of interest is the training representation: complete trajectories versus unordered compact-state transitions.

## Delivered scale

| Item | Count |
|---|---:|
| Unique original problems | **48,000** |
| Categories | **30** |
| Problems per category | **1,600** |
| Full-context TRAIN / DEV / TEST | **43,200 / 2,400 / 2,400** |
| Split-transition TRAIN / DEV / TEST | **170,941 / 9,460 / 9,521** |
| Total compact-state transitions | **189,922** |
| Mean transitions per problem | **3.9567** |
| Minimum / maximum transitions | **2 / 9** |

Every normalized problem text is unique. TRAIN, DEV, and TEST are assigned at the **problem level before transition expansion**, so no steps from one problem cross split boundaries.

## Coverage

The 30 balanced categories are:

```text
integer_arithmetic      order_of_operations      fractions

decimals                percentages              ratios

linear_equations        inequalities             geometry

unit_conversion         elapsed_time             rate_distance

statistics              probability              sequences

set_reasoning           boolean_logic            ordering_constraints

scheduling              inventory                finance

cooking                 physics                  chemistry

biology                 code_trace               string_processing

data_table              language_rules           recursive_state_updates
```

The tasks span mathematics, logic, symbolic transformations, data interpretation, planning, state updates, finance arithmetic, cooking ratios, introductory science, code tracing, language-rule application, and explicit recursive-state problems. Most prompts provide any formula or rule needed, emphasizing application and reasoning rather than factual memorization.

## Repository layout

```text
rineya_sequence_independence_dataset_v1/
├── README.md
├── DATA_CARD.md
├── EXPERIMENT_PROTOCOL.md
├── LICENSE_NOTES.md
├── CHANGELOG.md
├── data/
│   ├── full_context/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   ├── split_transitions/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   └── test.jsonl
│   └── manifests/
│       ├── category_distribution.csv
│       ├── dataset_stats.json
│       ├── problem_manifest.jsonl
│       ├── validation_report.json
│       └── sha256sums.txt
├── samples/
│   ├── PAIRED_EXAMPLES.md
│   └── paired_examples.jsonl
├── schemas/
│   ├── full_context.schema.json
│   └── split_transition.schema.json
└── scripts/
    ├── common.py
    ├── generators_math.py
    ├── generators_reasoning.py
    ├── generate_dataset.py
    ├── validate_dataset.py
    ├── export_training_view.py
    └── build_epoch_view.py
```

## Version A: full-context records

Each row contains one problem and one complete reasoning target.

```json
{
  "variant": "full_context",
  "problem_id": "rineya_si_v1_linear_equations_0042",
  "split": "train",
  "problem": "Solve the equation 7x + 5 = 47.",
  "answer": "x = 6",
  "full_response": "Reasoning:\n1. Subtract 5 ...\n2. Divide ...\n3. Therefore ...",
  "messages": [
    {"role": "user", "content": "Solve the equation 7x + 5 = 47."},
    {"role": "assistant", "content": "Reasoning:\n1. ..."}
  ]
}
```

For ordinary chat-template SFT, train from the `messages` field.

## Version B: split-transition records

Each row contains:

```text
original problem
+
compact current state
→
next correct reasoning step
```

The transition rows are globally shuffled. A later step can appear anywhere in the JSONL file and does not depend on the preceding row.

```json
{
  "variant": "split_transition",
  "problem_id": "rineya_si_v1_linear_equations_0042",
  "transition_id": "rineya_si_v1_linear_equations_0042::step_02",
  "step_index": 2,
  "total_steps": 3,
  "problem": "Solve the equation 7x + 5 = 47.",
  "current_state": "Isolated equation: 7x = 42.",
  "target_step": "Divide both sides by 7: x = 42 ÷ 7 = 6.",
  "next_state": "Solved value: x = 6.",
  "done": false,
  "messages": [
    {"role": "user", "content": "Original problem: ... Current compact state: ..."},
    {"role": "assistant", "content": "Divide both sides by 7 ..."}
  ]
}
```

Two message views are embedded in every transition:

- `messages`: next-reasoning-step-only supervision. This is the most direct realization of the hypothesis.
- `stateful_rollout_messages`: a JSON target containing `next_reasoning_step`, `updated_compact_state`, `done`, and `final_answer`. This optional view is easier to use in an iterative inference harness because the model explicitly emits the state to feed into its next call.

Do not combine both views in the same adapter run unless that is an intentional third condition.

## Quick export commands

The source files already contain trainer-ready message fields. To create minimal JSONL views containing only the selected messages and identifiers:

```bash
python scripts/export_training_view.py \
  --view full_context \
  --split train

python scripts/export_training_view.py \
  --view split_next_step \
  --split train

python scripts/export_training_view.py \
  --view split_stateful \
  --split train
```

The outputs are written under `data/derived_views/`.

## Problem-count-matched epoch views

The split dataset has more rows because each source problem expands into several transitions. To create an optional view with exactly one transition per source problem per epoch:

```bash
python scripts/build_epoch_view.py \
  --split train \
  --epochs 3 \
  --message-view next_step
```

For every epoch, this produces exactly **43,200** rows. The selected transition rotates across epochs, and all selected rows are globally shuffled.

This view matches the number of source problems per epoch, but it does **not** guarantee exact tokenizer-token equality. Exact token-budget matching should be calculated using the tokenizer of the model being trained.

## Validation

Run:

```bash
python scripts/validate_dataset.py
```

The validator checks:

- all 48,000 problem texts are unique after whitespace/case normalization;
- all IDs are unique;
- TRAIN/DEV/TEST are disjoint at the problem level;
- every transition exactly matches its source full-context trajectory;
- transition indexes are complete and contiguous;
- only the final transition has `done=true`;
- hashes match;
- no adjacent split-transition rows share a problem ID;
- every record can be deterministically regenerated from its stored seed.

The released build passes all checks. See `data/manifests/validation_report.json`.

## Regeneration

The dataset is deterministic and uses only the Python standard library:

```bash
python scripts/generate_dataset.py \
  --per-category 1600 \
  --seed 20260818

python scripts/validate_dataset.py
```

Changing the seed creates a new synthetic realization. Changing `--per-category` requires a value divisible by 20 so every category preserves the exact 90/5/5 split.

## Recommended first experiment

Train two fresh adapters from the **same untouched base checkpoint**:

```text
Adapter FULL:
  data/full_context/train.jsonl
  message field: messages

Adapter SPLIT:
  data/split_transitions/train.jsonl
  message field: messages
```

Keep model, tokenizer, LoRA targets, rank, alpha, dropout, optimizer, precision, maximum sequence length, random-seed schedule, checkpoint-selection rule, and evaluation code fixed.

Run at least two budget interpretations:

1. **Native-regime comparison:** one epoch over each native dataset. This asks how the two complete training recipes behave, but the split adapter sees more rows.
2. **Matched-budget comparison:** equalize tokenizer tokens, optimizer updates, or source-problem exposures. This better isolates representation from training quantity.

The detailed design is in `EXPERIMENT_PROTOCOL.md`.

## Provenance and limitations

- All examples are synthetic and procedurally generated for this project.
- No sentence or problem from the motivating Hugging Face dataset was copied.
- “Unique” means unique normalized problem text and unique problem ID; the dataset is deliberately template-based so answers and transitions can be verified exactly.
- This version emphasizes simple and moderate deterministic tasks. It does not claim to represent open-ended theorem proving, real-world medical/legal judgment, or broad factual knowledge.
- The explicit reasoning targets are synthetic supervision traces, not evidence of a model’s private internal reasoning process.
