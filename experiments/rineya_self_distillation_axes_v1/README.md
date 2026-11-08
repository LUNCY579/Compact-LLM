# DSD-001 — independent self-distillation axes for Rineya

**Source:** Zuin et al., *Disentangling Self-Distillation: Measuring and Modeling
Acquisition and Retention*, arXiv:2609.39494v1 (2026-09-30).

**Branch:** `research/rineya-self-distillation-axes-v1`  
**Parent:** `post-training-Mimo@03db218ceebc61a2bc04f2482cf7e094045e01c1`  
**Status:** 130 CPU tests passed; 144 finite controlled-model runs completed.
**Exact Rineya checkpoint / Transformers / CUDA: NOT EXECUTED, NOT QUALIFIED.**

This branch asks whether privileged-context self-distillation changes the
acquisition/retention tradeoff of the exact 0.6B-class Rineya model. It is not a
claim that self-distillation prevents forgetting or that the paper's best setting
transfers to a smaller model. The BF16/GProj experiment remains on its own branch:
no precision correction, production training default, existing evidence, game or
workspace code is changed here.

Read [PROTOCOL.md](PROTOCOL.md), [SOURCE_MAP.md](SOURCE_MAP.md), and
[VERIFICATION.md](VERIFICATION.md) before interpreting results.

## Reviewed publication — 2026-10-09

The additive preparation and its selected evidence are archived in the
[publication record](publication/2026-10-09/README.md). All **196 CPU fixtures**
pass on the publication checkout. The 130-test build and controlled results above
remain historical implementation evidence. Native qualification, teacher
competence, training benefit and memory fit remain unestablished by this work.
Published metadata is portable and non-executable; old packet identities require
a fresh exact proposal and review before any future native phase.

## What is implemented

| Component | Implemented behavior |
|---|---|
| Independent axes | Student/teacher rollouts; forward `KL(q||p)` / reverse `KL(p||q)`; separate EMA rate |
| Paper LLM grid | 2 sources x 5 rates `{0,.02,.05,.1,.25}` x 2 KL directions = 20 arms, plus SFT |
| Controlled model | Appendix E finite autoregressive construction, exact detached prefix occupancies, fixed teacher context displacement, shared trainable readout |
| Teacher bank | Shares immutable backbone; stores only changed parameters in CPU FP32; exact frozen/synchronized endpoints; restores swaps on error |
| Native pilot | Explicit local Qwen3-derived Rineya binding, batch one, short sequences, selected FP32 master parameters, no downloads or remote code |
| Evaluation | Original student prompt only; exact behavioral scorers, final-three acquisition and final macro-task retention, source successes retained/damaged/repaired |
| Analysis | Paired contrasts, Pareto points and seed-parity cross-fitting with fixed-axis checks |
| Safety | Hash-pinned checkpoint/data/code/runtime, no-update qualification, explicit training opt-in, create-only outputs and failure receipts |

The teacher sees the reference response in a fixed user-message template. The
student sees only the original task prompt. They score the **same sampled response
prefix**, using their own correctly shifted prompt offsets. Teacher probabilities
and trajectories are stop-gradient. KL uses the full vocabulary, not sampled-token
cross-entropy or a top-k approximation. SFT learns the exact reference plus EOS.

`ema_rate` is the paper's teacher-update coefficient, applied **after** the student
optimizer step. It is never reused to mean KL direction or JSD mixing. Rate zero
is frozen; rate one is an additional synchronized-endpoint diagnostic, not one of
the paper's five LLM rates.

## Run without model weights

Use an isolated environment with PyTorch and pytest. Tested: Python 3.13.5,
PyTorch 2.10.0+cpu, pytest 9.0.2. Run from repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  -c experiments/rineya_self_distillation_axes_v1/pytest.ini \
  --confcutdir=experiments/rineya_self_distillation_axes_v1 \
  experiments/rineya_self_distillation_axes_v1/tests -q

# Proposed 126-cell pilot manifest only: two tasks, three seeds, 21 methods.
python -m experiments.rineya_self_distillation_axes_v1 plan \
  --output /tmp/dsd-plan-new.json

# Fresh ordinary/rotated spatial diagnostic, NOT original paper datasets.
python -m experiments.rineya_self_distillation_axes_v1 synthetic-data \
  --parents-per-split 12 --output /tmp/dsd-data-new.jsonl

# Small smoke: 20 updates; default rates 0, .0025, .02.
OMP_NUM_THREADS=1 python -m experiments.rineya_self_distillation_axes_v1 controlled \
  --output /tmp/dsd-controlled-new.json

# Executed reference geometry/horizon, but only TWO seeds, not the paper's 96.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.rineya_self_distillation_axes_v1 controlled \
  --steps 200 --seeds 16 17 --paper-grid \
  --output /tmp/dsd-reference-new.json
```

The controlled `--paper-grid` is **18 rates**, not the five-rate LLM grid. Geometry
controls `--concentration`, `--context-strength`, `--feature-overlap` and
`--readout-compatibility` are explicit; do not conflate them with method axes.
Per-cell controlled measurements are preserved in `evidence/controlled_reference.csv`.
These are distributional similarities, not LLM accuracy or native capability gains.

## Native sequence (requires local checkpoint and reviewed data)

1. Materialize and hash the exact safetensors checkpoint, including its tokenizer.
2. Copy `configs/native.example.json`, fill actual local paths and hashes, select
   exact parameter names, and prospectively review task/data/scorer/budget choices.
3. Run no-update qualification on that exact config/runtime.
4. Only then launch one explicitly selected arm. Plans never auto-launch a sweep.

```bash
python -m experiments.rineya_self_distillation_axes_v1 manifest \
  --checkpoint /absolute/path/to/exact-rineya-0.6b \
  --output /tmp/dsd-checkpoint-manifest-new.json

python -m experiments.rineya_self_distillation_axes_v1 qualify \
  --config /absolute/path/to/frozen-config.json \
  --output /absolute/path/to/new-qualification-directory

python -m experiments.rineya_self_distillation_axes_v1 train \
  --config /absolute/path/to/frozen-config.json \
  --qualification /absolute/path/to/new-qualification-directory/qualification.json \
  --arm sd-teacher-forward-ema0.02 --execute-training \
  --output /absolute/path/to/new-arm-directory
```

The displayed arm is a syntax example, not a recommended winner. Every cell must
start from the same source; changing task, seed, LR, data, code or runtime requires
a matching qualification. Native dependencies need compatible Transformers Qwen3
and SDPA APIs; 4.57.1 documentation was consulted, but Transformers was not installed
or executed here. Do not upgrade a successful production environment in place.

### Data contract

Each JSONL row contains exactly these nonempty string fields:

```json
{"id":"fresh-1","parent_id":"independent-family-1","task":"new_task","split":"train","purpose":"acquisition","source":"reviewed provenance","admitted_for":"train","prompt":"Original task","reference":"\\boxed{3}","answer":"3","scorer":"boxed_int"}
```

Provide TRAIN and DEV for the chosen acquisition task and an independent DEV
retention cohort. Supported scorers: `exact`, `json_exact`, `boxed_int`,
`boxed_base9`, `boxed_pair`. JSON scoring rejects duplicate keys, NaN and type
coercion. Box scorers implement a documented simple nonnested grammar, not general
symbolic equivalence. References must pass their declared scorer.

The generated spatial pair has identical prompts/parents and different reference
conventions; select **one** task per run. Its arithmetic retention cohort is narrow.
A broad retention claim needs a separately authorized broad evaluation suite.
Provenance declarations and hash integrity do not prove semantic non-leakage or
permission: existing spent/sealed/selection-barred banks cannot be relabeled.

## Limits and interpretation

The native adaptation uses selected parameters and a shared frozen backbone rather
than the paper's full-model fine-tuning. Fixed math SDPA, BF16 operands, FP32 masters,
batch-one sampling at temperature 1, no KV cache, a short budget and an independently
written teacher template are deliberate deviations. No exact paper reproduction,
speedup or 8 GB memory-fit certificate is claimed.

Limits: at most 64 updates per arm, 128 response tokens, 512 total tokens, 20M selected
parameters, 512 rows and a 4 MiB dataset. Checkpointing, compiled graphs, DDP,
quantized loading, sliding-window/scaled-RoPE checkpoints and external FA backends
are outside qualification. Native qualification covers the tested objective
mechanics, not all model capabilities or worst-case memory use. Training failures
are retained, not replaced by silently shorter rows or easier gates.

Keep acquisition, retention, EOS/length-limit behavior, teacher entropy, update
magnitudes and cost separate. Equal prompts/steps/reference information does **not**
mean equal tokens or FLOPs across SFT and distillation. Use a prospectively expanded
SFT LR grid before concluding that a distillation method dominates SFT. No result
from this branch automatically promotes or overwrites a model.
