# FWT-001 runbook

All commands run in the foreground from the ChatBubble repository root.
Use a separate environment outside both experiment directories. Do not touch the
source model, historical receipts, legacy code, or existing branches to fix a failure.

## Environment and inspection

The official lane deliberately uses CPU FP32, PyTorch `2.10.0+cpu` and
Transformers `4.51.3`, with four intra-op threads. Reuse a compatible isolated
RAB-001 environment or create a new one; do not downgrade a shared project venv.
The native dependency file already exists at
`experiments/architecture_benchmark_001/requirements-native.txt`.
Install the CPU PyTorch build explicitly according to the existing RAB-001 runbook.
No model is downloaded by FWT-001. Installed safetensors/numpy/tokenizers versions,
Python, platform, thread counts and relevant runtime flags are receipt-bound.

```bash
python -m pytest -q experiments/full_width_transition_v1/tests
python -m experiments.full_width_transition_v1 inspect
```

These tests need torch, pytest and safetensors, not Transformers or official
weights. Passing them is not native qualification. `inspect` is read-only and
authorizes nothing. Do not put a venv under this package: source hashing includes
Python files recursively except outputs/verification/cache directories.

## P1: exact official source and active-path mechanical qualification

Set `SRC` to an existing exact ungated Supa 1.7.0-devfreeze directory. It must
contain the seven raw-byte-verified assets named in the unchanged RAB-001 source
lock. Stock Qwen, an adapter-wrapped model, another core revision, permissive tensor
loading, source symlinks for individual assets and automatic downloads are rejected.

```bash
SRC=/absolute/path/to/Rineya-Supa-1.7.0-devfreeze
E=experiments/full_width_transition_v1
python -m experiments.full_width_transition_v1 qualify \
  --source "$SRC" --out "$E/outputs/p1-attempt-001"
```

PowerShell users may pass literal quoted paths on one line rather than POSIX shell
variables/continuations. Every output directory must be a new child of this
package's `outputs/`. Attempts are create-only, including failed attempts. An OS
kill may leave only `attempt.json`; it is not a receipt or a pass. Use a new
attempt directory after diagnosis, never erase failure evidence to reuse a name.

P1 checks native/reference parity before constructing the variants, persists
initialization research checkpoints, checks reload, temporarily activates the new
read path and checks cache/causal behavior. The perturbation is not saved as a
trained model. Source weights are read-only and hashed before/after.

P1 needs at least 8 GiB free output disk for the two FP32 research checkpoints and
headroom. Original and candidate models coexist during some checks: allow roughly
5 GiB for two FP32 weight sets alone, plus activations, temporary tensors and
libraries. This is not a measured RAM bound or 8-GB GPU-fit guarantee. The loader
uses meta construction and releases the native model before candidate checks to
avoid an unnecessary third live full model.

A failed P1 blocks P2. Do not relax the elementwise numerical rule or substitute
argmax agreement. Do not borrow a legacy C28/C40 receipt. The augmentation is a new
arm with its own source/native/active-path qualification, not permission to train
those failed historical conversions.

## P2: matched 25-step engineering smoke only

Use the exact P1 receipt, unchanged implementation/fixtures/runtime/source, and a
new output directory. Requesting this command is separate from running P1; a pass
does not automatically launch it.

```bash
python -m experiments.full_width_transition_v1 smoke \
  --source "$SRC" --parent "$E/outputs/p1-attempt-001/receipt.json" \
  --out "$E/outputs/p2-attempt-001"
```

Both arms are mandatory and run sequentially from the original locked source,
not from temporary active-path perturbations or untrusted saved checkpoints.
Only added parameters train. Frozen teacher logits are computed per batch;
there is no persistent full-vocabulary teacher-cache dataset. The only training
inputs are the eight authored `engineering_train` paragraphs in `fixtures.json`.
No legacy calibration/pilot rows or sealed data are trained on.

The step count, length bound, optimizer and added-only policy are enforced; there
is no CLI override for steps, data, architecture, precision or learning rate.
P2 returns engineering trajectories and a receipt, not a recovered release model.
Changing implementation, tests, protocol, scientific documentation, dependencies,
fixtures or runtime makes the old parent stale. Evidence/verification files are
excluded from the implementation hash to avoid recursive self-hashing.

## What remains blocked after P2

Registered recovery, DEV and sealed evaluation, pure-CED replacement, depth
campaigns, width/FFN/head pruning, efficacy claims and deployment remain false.
The CLI has no command for any of them. Complete the preregistration requirements
in `RESEARCH_PLAN.md` and obtain an independently reviewed new campaign
implementation before opening any of those lanes.

P1/P2 hash bindings detect stale or accidentally changed records; they are not
cryptographic attestations against a malicious author. Never manually edit receipt
booleans or hashes to advance. Preserve raw measurements and failure records.
