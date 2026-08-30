# Real-checkpoint TGSA + MTP warm-start integrity runbook

The original `train_tgsa_mtp_adapter.py` remains the low-level implementation.
Operational runs use `train_tgsa_mtp_adapter_hardened.py`, which requires an
explicit development marker and binds the source checkpoint, data, device, and
training dose before launching it.

## Run

```powershell
$Retrofit = "Rineya_TGSA_Retrofit_v1"

python ".\$Retrofit\scripts\train_tgsa_mtp_adapter_hardened.py" `
  --development-only `
  --model-dir "<LOCAL_RINEYA_OR_SUPA_CHECKPOINT>" `
  --train-jsonl "<TRAIN_JSONL>" `
  --output-dir ".\evidence\supa_tgsa_mtp_warmstart_seed_101" `
  --layers "6,12,18" `
  --stage adapters `
  --steps 500 `
  --batch-size 1 `
  --gradient-accumulation 8 `
  --device cuda `
  --seed 101
```

The wrapper hashes every regular file under `model-dir` before training. This is
intentional: a model name or directory path is not an immutable checkpoint
identity.

## Verify

```powershell
python ".\$Retrofit\scripts\verify_tgsa_mtp_adapter_integrity.py" `
  --output-dir ".\evidence\supa_tgsa_mtp_warmstart_seed_101"
```

## Recorded dose

For the warm start,

```text
sample_presentations =
    optimizer_steps × microbatch_size × gradient_accumulation

effective_epochs =
    sample_presentations ÷ unique_train_rows
```

The manifest records both sample presentations and unique rows. Equal total
presentations with different data diversity are not treated as the same regime.

## Claim boundary

The output is always marked:

```text
development_only = true
confirmatory = false
mechanism_interpretation_allowed = false
```

It is a warm-start artifact, not held-out quality evidence. It cannot certify
the Phase-2 factorial matrix and cannot support a wall-clock speed claim. The
adapter checkpoint must later enter the registered four-arm evaluation with an
external Phase-1 competence prerequisite, sealed held-out data, and a separate
cache-aware decoding benchmark.
