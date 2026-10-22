# Execution and native qualification

Run from the ChatBubble repository root on the new branch. Nothing here launches a
Codex task or touches a game project. The CPU test environment is not the user's PC.

## 1. Inspect without loading a checkpoint

```sh
python -m experiments.rineya_layered_post_training_v1 plan --config experiments/rineya_layered_post_training_v1/configs/native_2k.json --arm layered
```

Create an `outputs` directory if it does not exist. Substitute an actual, materialized
local checkpoint directory in the next command (not a Hugging Face model ID):

```sh
python -m experiments.rineya_layered_post_training_v1 preflight --config experiments/rineya_layered_post_training_v1/configs/native_2k.json --model-dir PATH_TO_LOCAL_QWEN3_RINEYA_0_6B --output outputs
```

Preflight is read-only and checks CUDA, free RAM/VRAM/disk, local weights and native
software availability. A ready preflight is NOT a model-qualified or efficacy result.
Loading additionally checks native Qwen3 model type, 500M–700M parameters, architecture
projection names, fixed trainable capacity, BF16 CUDA support, safe local loading,
immutable file hashes and configured context. No remote architecture code is trusted.
A custom Rineya architecture fails closed instead of being relabeled as Qwen3.

The initial runtime qualification checks finite response-only loss and sampled-token
versus teacher-forced log probabilities. EOS/truncation is recorded. The training
runtime rejects incomplete RL groups, stale policy versions, nonfinite gradients,
context overflow and exhausted budgets. A loose likelihood tolerance is not a remedy
for a sampler mismatch: investigate the kernel/dtype/cache difference before editing it.

## 2. Prepare local data

See `DATA.md`. Supply `train`, `select` and `confirm` rows with disjoint prompt families.
Default development and confirmation requirements are at least 128 independent groups
per suite; this is a screening minimum, not proof of power for a one-point margin.
Freeze the dataset and protocol before running any arm. Do not include benchmark test
answers in distillation/preference/replay data. Teacher traces and preferences are
imported; this branch does not create synthetic traces with paid APIs or download MiMo.

## 3. Run one short native arm with explicit permission

```sh
python -m experiments.rineya_layered_post_training_v1 run --config experiments/rineya_layered_post_training_v1/configs/native_2k.json --arm layered --model-dir PATH_TO_LOCAL_QWEN3_RINEYA_0_6B --data PATH_TO_FROZEN_DATA.jsonl --output outputs/rlp_layered_s0 --allow-model-load --allow-training
```

Use new directories for `control` and `soup_only` with the same seed/config/data. The
60-update default is a low-dose pilot, not a promised adequate training recipe. Increase
training dose only in a new preregistered campaign, after memory/likelihood qualification.

On an 8GB GPU the reference model is kept on CPU. Reference scoring may be slow. The
runtime keeps multiple sequence graphs for an RL group, so a passing free-memory check
does not prove that a later batch will fit; OOM remains a failed run, not a positive result.
Use short data and the 2K ceiling first. Evaluate a separately registered 4K pilot only
once 2K is valid. The code does not automatically increase batch size or context length.

## 4. Inspect before opening confirmation

Read `run_report.json`, `qualification.json`, per-stage evaluations and the selection
lock. Check terminal-stage scores as well as selected scores: a full pipeline that
selects an earlier checkpoint has NOT shown that every stage helped. Inspect the
soup recipes and explicit base fallback. Check zero-signal RL groups, truncations,
consolidation trace counts, likelihood parity, total token-forward work and critic costs.

```sh
python -m experiments.rineya_layered_post_training_v1 compare --reports outputs/rlp_control_s0/run_report.json outputs/rlp_layered_s0/run_report.json
```

If `NOT_COMPUTE_MATCHED`, results are descriptive. Set matched budgets in a fresh study;
do not retroactively hide reference, search, critic or evaluation work. The command
uses token proxies and does not claim FLOP equality even when those proxies match.

## 5. Confirm the already-frozen candidate

```sh
python -m experiments.rineya_layered_post_training_v1 confirm --model-dir PATH_TO_LOCAL_QWEN3_RINEYA_0_6B --data PATH_TO_FROZEN_DATA.jsonl --output outputs/rlp_layered_s0 --allow-model-load
```

This loads the original config, checks checkpoint and data hashes, evaluates base and
candidate with paired seeds, writes cluster-paired statistics and refuses a second
confirmation artifact. No training flag is needed because no optimization occurs.
A real campaign requires independently trained seeds (e.g. 20261007/8/9), an untouched
confirmation cohort, matched-compute controls and a powered retention assessment.

Only then advance to `full_no_critic`, `full` and `full_no_soup`. The native full arm
halts before RL3 when the critic fails its selection-only audit; it never substitutes
a random head or silently switches to GRPO. The no-critic arm is an explicit control.

## Recovery and stop rules

A failed run has `failure.json` and no promotion lock. Existing output directories and
snapshots are never overwritten. There is no exact optimizer resume in v1: restart
from the same frozen base in a new directory, or design a separately documented warm
start. Do not average or reload Adam moments across a soup. Do not weaken gates after
seeing confirmation. Do not treat a synthetic smoke pass as permission for an unattended
GPU job on the user's Windows machine.
