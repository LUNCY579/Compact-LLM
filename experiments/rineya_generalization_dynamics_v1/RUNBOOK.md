# Operator runbook

## 0. Preserve the existing campaign

Work only on `research/rineya-generalization-dynamics-v1`. Do not merge it automatically,
change any frozen Rineya gate, open old sealed/SPENT/ANY_READ banks, or interrupt a GPU
job. This package is inference-only. It observes existing training exports; it does
not authorize another training campaign or replace the user's existing trainers.

Choose an output directory outside tracked source files, for example
`/absolute/path/to/research-runs/gd001-native-001`. Do not reuse a demo output directory
for native inference. Windows operators can supply absolute Windows paths instead.

## 1. Mechanical verification

```bash
python -m unittest discover -s experiments/rineya_generalization_dynamics_v1/tests -v
python -m experiments.rineya_generalization_dynamics_v1 demo --run /tmp/gd001-mechanical-001
```

The full synthetic demo intentionally chooses different fixture checkpoints for the
ordinary and robust criteria. It tests plumbing, not whether Rineya benefits.

## 2. Prepare an isolated native environment

Use the appropriate PyTorch CPU/CUDA wheel for the host, then the package-specific
requirements. The proposed qualification dependency pins follow the inherited
Transformers 4.51.3 boundary, but **the full HF/PEFT stack has not been exercised in
this delivery environment**. Do not modify a working training environment in place.

```bash
python -m pip install -r experiments/rineya_generalization_dynamics_v1/requirements-native.txt
python -m experiments.rineya_generalization_dynamics_v1 preflight \
  --spec /absolute/path/to/native-spec.json
```

Preflight loads no model. Missing dependencies, host RAM, unavailable CUDA or an active
foreign GPU process block execution. A `PREFLIGHT_ONLY` result is not native qualification.
Actual model/tokenizer export compatibility is checked during loading. A 7.4 GiB free-
RAM guard is intentionally conservative; this implementation cannot run on the delivery
host's roughly 3.6 GiB available RAM, and must not silently reduce that guard.

## 3. Capture checkpoints using the existing trainer

After the existing trainer has completed a normal, immutable export, optionally record
it without altering model state:

```python
from experiments.rineya_generalization_dynamics_v1.integration import record_snapshot

# Run only after save_pretrained/save_model has fully finished.
receipt = record_snapshot(
    receipt_directory="/absolute/path/to/gd001-snapshot-receipts",
    checkpoint_directory="/absolute/path/to/adapter-step-100",
    step=100,
    kind="peft",  # "hf" for full standard causal-LM exports
)
print(receipt["entry"])  # Copy this entry into the experiment snapshot specification.
```

Do not evaluate inside an optimizer step or use the trainer's live, changing export
directory. The record hook refuses to overwrite a step receipt. It does not itself save
weights, call an optimizer, or start a GPU job. Framework-specific callbacks can call
this function after their existing save operation; no trainer integration is assumed.

## 4. Register a complete single-run specification

Copy `configs/native.example.json` to a private, untracked path and replace every path
and the trajectory ID. The example 596,049,920 count is the inherited Qwen3-compatible
0.6B shape, not an assumption that every Rineya export has that count. Supply the exact
count for the intended checkpoint; it is checked after loading.

Required assets: a full base causal-LM safetensors export, the exact matching tokenizer,
an ordered list of full or LoRA snapshots, and a complete UTF-8 training manifest.
The training manifest is one JSON object per line with a `prompt` string and, when
available, a `parent_id`. It must enumerate all actual training prompts, not development
examples or a conveniently small subset. Preserve upstream training and exclusion
receipts for audit. The implemented overlap audit is exact, not semantic.

Base snapshots do not need a duplicate `path`; other snapshot paths must point to stable
exports. All paths are local. CPU FP32 is the default. A CUDA pilot must explicitly
preregister `device: "cuda"`, typically `dtype: "bfloat16"`, and pass the host scheduler
and both RAM/VRAM checks. Change prompt format and thresholds only before registration.

```bash
python -m experiments.rineya_generalization_dynamics_v1 init \
  --run /absolute/path/to/research-runs/gd001-native-001 \
  --spec /absolute/path/to/native-spec.json \
  --policy experiments/rineya_generalization_dynamics_v1/configs/policy.default.json \
  --seed 17 --pairs-per-family 24
```

Initialization hashes local files, checks identities, writes only new experiment
artifacts, and builds DEV. It does not create confirmation items or load a model.
An `init` failure does not justify switching to a stock model.

## 5. Evaluate development and freeze the choices

```bash
python -m experiments.rineya_generalization_dynamics_v1 evaluate-dev \
  --run /absolute/path/to/research-runs/gd001-native-001 --allow-native
python -m experiments.rineya_generalization_dynamics_v1 select \
  --run /absolute/path/to/research-runs/gd001-native-001
```

Development can resume missing checkpoints before selection; completed, validated
receipts are reused and never overwritten. A stale `BUSY.lock` needs manual process
inspection, not blind deletion. After selection, development cannot reopen. Code,
model, tokenizer, manifest or policy changes require a new run identity.

Inspect `development_report.json`: competence by family, shortcut rates, conditional
support, R+S given R and S, reference-fixed retention damage, and adjacent transitions.
`selection.json` contains either `LOCKED` or `BLOCKED_NO_ELIGIBLE_CHECKPOINT`.
A blocked selection is a valid result, not a reason to loosen its thresholds.

## 6. Fresh one-use confirmation, only after operator authorization

Choose an independent confirmation seed not used in development, without inspecting
its generated items. Have the local qualified host available before requesting access:

```bash
python -m experiments.rineya_generalization_dynamics_v1 confirm \
  --run /absolute/path/to/research-runs/gd001-native-001 \
  --seed 101 --allow-confirmation --allow-native
python -m experiments.rineya_generalization_dynamics_v1 report \
  --run /absolute/path/to/research-runs/gd001-native-001
```

**Do not execute this step merely to test whether the native loader works.** Qualify the
loader on DEV first. Confirmation is spent before data generation; a model-load error,
OOM, timeout or interruption after that point still consumes the attempt. There is no
confirmation retry command. Partial files and STARTED receipts must be retained.

## 7. Read the results at their actual evidence level

`confirmation_report.json` records fixed choices, the primary generalization-vs-final
contrast, the ordinary-vs-final control, cluster intervals, gate failures and explicit
limitations. Model generation text, token IDs, likelihoods, stop reasons, inference time
and resource readings remain in per-checkpoint JSON receipts. No checkpoint is promoted
or moved into a production path. Report negative results and failures too.

A useful next research milestone is one qualified native trajectory with a complete
unspent confirmation result. A reproducible advantage would justify a new multi-seed
study and broader, separately admissible retention instruments. No native advantage has
been established by this implementation delivery.
