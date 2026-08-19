# Rineya State Sufficiency Lab

A standalone, reproducible repository for training and evaluating the hypothesis:

> **A compact current state may contain enough information for the next correct reasoning decision, without replaying the complete reasoning history.**

The repository trains the **same immutable base model** into two separate adapters using the paired dataset bundled under `assets/datasets/`:

- **FULL adapter:** one problem is supervised as one uninterrupted reasoning trajectory and final answer.
- **SPLIT adapter:** the same canonical trajectory is expanded into independently sufficient compact-state transitions. Transition rows are globally shuffled, so chronological adjacency is unavailable to the optimizer.

This is **not** a verb-shortening experiment, a short-response-only experiment, or a training-speed claim. The treatment variable is the supervision representation.

## Release contents

```text
Rineya_State_Sufficiency_Lab_v1.0.1_CORRECTED/
├── assets/datasets/                 # complete paired dataset archive
├── configs/
│   ├── train/                       # FULL, SPLIT, ordered, next-step, QLoRA
│   ├── eval/                        # DEV, sealed TEST, reference, HTTP server
│   ├── experiment/                  # staged two-adapter experiment
│   └── accelerate/                  # single GPU, multi-GPU, FSDP, ZeRO-2
├── docs/                            # experiment and evaluation specifications
├── schemas/                         # response and result contracts
├── scripts/                         # PowerShell and Unix launchers
├── src/rineya_state_lab/
│   ├── data/                        # safe extraction and strict alignment checks
│   ├── training/                    # LoRA/QLoRA SFT pipeline
│   ├── inference/                   # local, HTTP, replay, and reference backends
│   └── evaluation/                  # protocols, parser, scorers, stats, reports
├── tests/                           # deterministic unit and integration tests
├── BUILD_REPORT.md                  # executed validation evidence
└── pyproject.toml
```

## Dataset scale

The bundled archive contains matched TRAIN/DEV/TEST partitions:

| Representation | TRAIN | DEV | TEST | Total |
|---|---:|---:|---:|---:|
| Full-context source problems | 43,200 | 2,400 | 2,400 | **48,000** |
| Compact-state transitions | 170,941 | 9,460 | 9,521 | **189,922** |

The dataset covers 30 balanced categories and 21 answer types. Preparation and validation operate on the actual archive; no download is required.

## Quick start on Windows 11

Open PowerShell in the extracted repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"

python -m rineya_state_lab data prepare `
  --archive assets/datasets/Rineya_Sequence_Independence_Dataset_v1.0_COMPLETE.zip `
  --output data/prepared

python -m rineya_state_lab data validate --data-root data/prepared --strict
python -m rineya_state_lab smoke-test --work-dir outputs/smoke --force
```

Set the same base checkpoint for both adapters:

```powershell
$env:RINEYA_BASE_MODEL = "C:\models\your-model"
python -m rineya_state_lab doctor --model-path $env:RINEYA_BASE_MODEL --data-root data/prepared --require-cuda
```

Train sequentially so only one model occupies GPU memory:

```powershell
python -m rineya_state_lab train --config configs/train/full_context_lora.yaml
python -m rineya_state_lab train --config configs/train/split_stateful_lora.yaml
```

Run DEV evaluation:

```powershell
python -m rineya_state_lab evaluate --config configs/eval/full_adapter_dev.yaml
python -m rineya_state_lab evaluate --config configs/eval/split_adapter_dev.yaml

python -m rineya_state_lab compare `
  --baseline outputs/eval/full_adapter_dev/scored_predictions.jsonl `
  --candidate outputs/eval/split_adapter_dev/scored_predictions.jsonl `
  --baseline-protocol direct_solve `
  --candidate-protocol generated_rollout `
  --metric score.correct_strict `
  --noninferiority-margin 0.02 `
  --output outputs/comparison/dev_primary
```

The complete staged orchestration is also available:

```powershell
python -m rineya_state_lab experiment run --config configs/experiment/main_two_adapter.yaml
```

## Training architecture

The training path is based on Transformers + PEFT and supports any compatible decoder-only causal language model supplied by local path or model identifier.

Implemented controls include:

- LoRA and QLoRA with configurable rank, alpha, dropout, target modules, RSLoRA, and DoRA flags;
- assistant-only causal loss, so user prompts and compact states are context rather than prediction targets;
- target-preserving truncation and dynamic right padding;
- BF16/FP16 auto-selection, TF32, gradient checkpointing, fused or paged optimizers;
- checkpoint resume and append-safe manifests;
- deterministic source-row selection;
- inverse-transition-count weighting so long trajectories do not automatically dominate by having more rows;
- one-transition-per-problem, globally shuffled, and trajectory-ordered ablations;
- matched `max_steps` in the primary FULL and SPLIT configs;
- exact or sampled token-exposure accounting;
- single-GPU, multi-GPU, FSDP, and DeepSpeed launcher examples.

The primary SPLIT target uses `stateful_rollout_messages`, which teaches the model to return:

```json
{
  "next_reasoning_step": "...",
  "updated_compact_state": "...",
  "done": false,
  "final_answer": null
}
```

This is required for autonomous free rollout. The evaluator never silently restores the gold state after a model error.

## Evaluation hierarchy

The harness keeps four research questions separate:

1. **Direct end-to-end solving:** Can an adapter answer the original problem?
2. **Oracle compact-state transition:** Given the canonical state, can it produce the next step and next state?
3. **State-sufficiency ablations:** How does compact state compare with full prior history, the last step, no state, shuffled state, and distractor state?
4. **Generated-state rollout:** Starting from only the initial state, can the SPLIT adapter maintain its own state and finish correctly?

The autonomous rollout detects and reports:

- malformed or partial structured output;
- missing updated state;
- premature completion;
- `done=true` without a final answer;
- implicit final answers;
- repeated-output loops;
- repeated-state loops;
- state-size overflow;
- backend errors;
- maximum-step nontermination;
- first exact and first lenient divergence from the canonical trajectory.

## Arbitrary response handling

Every raw response is retained. The parser recognizes:

- exact JSON, fenced JSON, JSON embedded in prose, trailing-comma repair, and safe Python-literal dictionaries;
- `<final>`, `<answer>`, `Final answer:`, `Answer:`, conclusion sentences, and LaTeX `\boxed{...}`;
- empty output, refusal, missing answer, and conflicting explicit answers;
- unstructured labeled state fields as a lenient diagnostic path.

Two tracks are always reported:

- **Strict:** unambiguous, contract-compliant, canonical scoring.
- **Lenient:** conservatively recoverable answers and state paraphrases, with numeric-conflict protection.

The lenient track is diagnostic; it does not silently replace the preregistered primary metric.

## Deterministic scorers

Task-aware scoring supports the dataset’s integer, decimal, fraction, probability, money, measurement, quantity, rate, time, Boolean, equation, inequality, set, state-assignment, statistic, program-output, ordering, text-transformation, and string formats. Numeric tolerances are configurable. Model text is never executed.

## Statistical decision standard

The primary comparison pairs FULL and SPLIT results by `problem_id` and reports:

- baseline and candidate accuracy with Wilson intervals;
- paired candidate-minus-baseline bootstrap intervals;
- exact McNemar disagreement testing;
- paired sign-flip randomization;
- category, domain, and difficulty effects;
- deterministic discordant-case audit samples;
- `SUPERIOR`, `EQUIVALENT`, `NONINFERIOR`, `INFERIOR`, or `INCONCLUSIVE` under a preregistered margin.

The default 2-percentage-point margin is a placeholder. Freeze and justify the real margin before opening TEST.

## Sealed TEST control

TEST configs require an explicit environment acknowledgement:

```powershell
$env:RINEYA_ALLOW_TEST = "I_UNDERSTAND_TEST_IS_SEALED"
python -m rineya_state_lab evaluate --config configs/eval/full_adapter_test.yaml
python -m rineya_state_lab evaluate --config configs/eval/split_adapter_test.yaml
```

Do not set this variable while tuning models, prompts, parser rules, margins, or generation settings.

## Resume and auditability

Training checkpoints are resumed with `resume_from_checkpoint: auto`. Evaluation skips already scored `request_id` values and appends each completed generation with `flush` + `fsync`. Every run records:

- resolved configuration and hash;
- dataset file hashes;
- base model and adapter paths;
- Python, package, CUDA, and hardware information;
- token and problem exposure;
- raw outputs, parser provenance, strict/lenient scores, and failure labels;
- machine-readable JSON plus Markdown and HTML reports.

## Important limitations

- The dataset is synthetic and procedurally generated. Strong performance does not by itself establish broad real-world reasoning ability.
- Exact semantic equivalence between arbitrary natural-language states is undecidable by a deterministic parser. The repository therefore reports canonical exactness, a conservative token/numeric equivalence diagnostic, and raw outputs for human audit.
- Teacher-forced transition quality can be high while free rollout fails through error propagation. Both must be reported.
- FULL and SPLIT contain unequal row counts. Use matched updates, inverse-problem weighting, token accounting, and the supplied ablations before attributing a result to representation alone.
- A successful no-model reference smoke test verifies the harness, not the scientific hypothesis. Real GPU training is still required.

See `docs/RUNBOOK.md` for the recommended experimental order and `BUILD_REPORT.md` for the executed release checks.
