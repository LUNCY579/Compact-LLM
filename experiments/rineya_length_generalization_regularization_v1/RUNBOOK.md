# Operator runbook

## 1. Before native work

Do not run a native command merely because this branch exists. Bind an authorized
local Rineya 0.6B checkpoint and verify its established loader/export provenance.
The standard Qwen checkpoint is a reference, not a substitute for native evidence.
This package's qualification is mechanical and does not grant permission to use
quarantined data, run CHECK/FINAL, consume an expired training grant, or use a
personal device that has not been authorized for this run.

Run the scoped tests and the CPU smoke first. Use an isolated environment. No
root dependency, production config, workflow, existing branch or laptop scheduler
needs to change. There is no batch sweep or automatic retry command.

## 2. Freeze the plan and data

The three JSON plans in `configs/` are proposals, not launch authorizations.
Record checkpoint/tokenizer/template identity, source/export receipt, data and
verification provenance, trainable scope, device, precision, backend, maximum
context, actual token lengths and run-specific resource authorization.

Each JSONL case contains:

    id: unique view identifier
    parent_id: semantic problem identity shared across length/depth views
    split: train | dev | confirmation
    task: registered task name
    role: long_context | acquisition | retention | abstention
    prompt_ids: nonempty list of token IDs; no padding or hidden template
    target_ids: nonempty verified completion token IDs (training uses all of these)
    answer: exact decoded target answer, without EOS
    answerable: boolean
    synthetic: boolean
    fixture: boolean; true is forbidden in the native lane
    length: exactly len(prompt_ids)
    depth: evidence-placement fraction in [0, 1]

The training objective ignores the prompt, causally shifts labels and averages
completion cross-entropy per sequence, then across accumulation members. Padding
is unnecessary for microbatch one. Train context includes prompt plus target.
Selection is a deterministic bounded prefix, so freeze a representative shuffled
order and retention composition upstream. Do not sort all acquisition before all
replay and then let a short budget accidentally exclude replay.

`build-data` can create synthetic diagnostics using an explicitly supplied local
tokenizer (no model weights) or clearly marked byte-tokenizer fixtures. Generated
native-tokenizer data uses plain-completion-v1. It is not automatically a suitable
assistant chat template or verified skill-acquisition training corpus.

```text
python -m experiments.rineya_length_generalization_regularization_v1 build-data --checkpoint /local/tokenizer --split dev --seed 211 --parents 64 --lengths 1024 2048 4096 8192 --output /new/dev.jsonl
python -m experiments.rineya_length_generalization_regularization_v1 audit-data /data/train.jsonl /data/dev.jsonl
```

Do not open the confirmation file casually to perform an audit. Have the dataset
custodian run the cross-split audit before sealing it and provide its receipt.
The bank reader caps files at 128 MiB; larger programs require a separately
reviewed streaming implementation, not an unrecorded limit increase.

## 3. Eligibility is an attestation, not a generated pass

`configs/eligibility.example.json` intentionally has false flags and an unbound
hash. A responsible operator must supply the actual bank hash and establish the
relevant eligibility flags from the existing data-governance process.
Training requires `train_eligible`, `quarantine_excluded`, and
`cross_split_parent_and_prompt_audit_passed`. Evaluation requires a matching bank
hash and `evaluation_eligible`. Do not set TRAIN eligibility on a held-out bank.
These JSON fields are operator assertions, not proof of data rights/decontamination.

## 4. Native qualifier — explicit opt-in, no optimizer update

Example only, after external authorization and source qualification:

```text
python -m experiments.rineya_length_generalization_regularization_v1 qualify --checkpoint /local/rineya-0.6b --plan experiments/rineya_length_generalization_regularization_v1/configs/C_vpad.json --bank /data/train.jsonl --eligibility /receipts/train-eligibility.json --output /new/lgr-C-qualify --allow-native-load
```

The native lane refuses downloads, remote code, pickle model checkpoints,
quantization, sliding-window changes, RoPE scaling and unexpected shapes. All
checkpoint files are hashed (including tokenizer/template metadata); symlinked
snapshots must be materialized separately before qualification. No source files
are modified. The existing source loader/export qualification is still required.

Resource gates: exactly one visible CUDA GPU; two pre-load free-host-RAM samples
>=7400 MiB; runtime free host RAM >=6144 MiB; process-tree RSS <=4096 MiB; reusable
CUDA headroom >=256 MiB; no foreign compute PID. Unsupported nvidia-smi telemetry,
including some Windows/WDDM configurations, fails closed. Do not weaken the
threshold or bypass foreign-job checks without a new reviewed host-specific lane.

Qualifier performs eval and p=0 identity, finite nonzero-dropout backward, safe
adapter serialization and nonzero-adapter merge checks. It takes no optimizer
step. BF16 merge tolerance is explicitly recorded (.1 absolute/.01 relative);
that tolerance is a mechanical screen, not a final serving-equivalence margin.
The produced receipt binds code, full checkpoint, bank, plan, eligibility and
runtime and expires after 24 hours. A code/config/runtime change invalidates it.

## 5. One bounded training arm

```text
python -m experiments.rineya_length_generalization_regularization_v1 train --checkpoint /local/rineya-0.6b --plan experiments/rineya_length_generalization_regularization_v1/configs/C_vpad.json --bank /data/train.jsonl --eligibility /receipts/train-eligibility.json --qualification /new/lgr-C-qualify/result.json --output /new/lgr-C-train --allow-native-load --allow-training
```

No pretrained run was executed in the implementation session. Each other arm and
seed needs its own bound plan/qualification. Adapter dropout stays zero; do not
substitute PEFT `lora_dropout` for the full-input intervention. The loop writes
CPU adapter-only `.pt` state dictionaries, loaded with `weights_only=True`, plus
SHA-bound JSON sidecars. No optimizer resume is provided. 0/25/50/100% snapshots
are at the first completed step crossing the target; multiple milestones can
share one step on tiny runs. Trace records exact consumed tokens and steps.

Failures produce `failure.json`; the output directory/run ID remains spent. No
retry, device takeover, process killing, paid service, upload or background task
occurs automatically. A different manual run remains a new authorization decision.

## 6. Evaluate, then compare bound reports

```text
python -m experiments.rineya_length_generalization_regularization_v1 evaluate --checkpoint /local/rineya-0.6b --plan experiments/rineya_length_generalization_regularization_v1/configs/C_vpad.json --bank /data/dev.jsonl --eligibility /receipts/dev-eligibility.json --adapter /new/lgr-C-train/adapter-100.pt --output /new/lgr-C-dev --allow-native-load
```

Omitting `--adapter` evaluates the unchanged source (zero-initialized adapter).
The model runs in eval mode, without stochastic activation regularization. The
maximum checkpoint context must include the prompt AND reserved completion.
Greedy continuation-only decoding is used; no silent left truncation, sampling,
best-of-N selection, or post-hoc tolerant answer scoring is introduced.

Confirmation requires an additional `--allow-confirmation` opt-in after the arm,
seeds, bank, hypotheses, power and scorer are frozen and the repository's existing
authority allows opening it. There is no automatic confirmation run.

```text
python -m experiments.rineya_length_generalization_regularization_v1 compare --reference /A/seed101/result.json /A/seed211/result.json /A/seed307/result.json --candidate /C/seed101/result.json /C/seed211/result.json /C/seed307/result.json --output /new/comparison.json
```

CLI comparison requires matching evaluation-contract hashes. It rejects bare,
unbound prediction arrays (the Python comparison API remains useful for fixtures).
Even `REVIEW_ELIGIBLE_NOT_RELEASED` requires independent source/data/seed/power,
serving-export, acquisition and abstention review. The shipped synthetic generator
alone cannot satisfy downstream, acquisition and retention promotion requirements.

## 7. Native limitations requiring follow-up measurements

GPU fit, throughput, exact-loader support, full Rineya format/abstention behavior,
actual generation/cache parity, full-checkpoint save/reload, long-output reasoning,
VPAD/LoRA interaction, and production export are not measured here. No context
beyond the native checkpoint limit is exposed. The merge utility is provided and
mechanically tested but training writes only adapters; actual serving export is
a separate gated operation. Attention scaling and the paper's SmolLM3 architecture
changes are intentionally absent from this initial, controlled transfer experiment.
