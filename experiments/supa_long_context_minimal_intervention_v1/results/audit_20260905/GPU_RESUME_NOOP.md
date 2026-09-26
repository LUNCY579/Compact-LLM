# Completed-checkpoint GPU restoration check

Prepared for explicit execution by the root task after a fresh smoke arm is fully
complete. The script has been syntax-parsed only; preparation did not load a
model, initialize CUDA, or execute the verifier.

Run one arm per fresh process from the repository root with the required Python:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_LAUNCH_BLOCKING = '1'
$env:PYTORCH_ALLOC_CONF = 'backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8'
$env:PYTORCH_CUDA_ALLOC_CONF = $env:PYTORCH_ALLOC_CONF
& ../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/results/audit_20260905/verify_completed_gpu_resume.py --stage phase1_smoke --arm B
```

Use `--arm C` in its own process. `--stage phase1_smoke_a1` selects a completed A1
smoke arm with the same fixed six-attempt recipe. Never overlap invocations; the
verifier acquires the same global `.phase1_model.lock` as the runner and fails if
another process owns it.

The verifier checks the complete lifecycle receipt, current source/core/data
hashes, regenerated encoded data, actual freshly constructed initializer, and
all five checkpoint fingerprints. It requires a completed six-attempt checkpoint
with no pending validation, then restores through `train(..., max_steps=6,
resume_from_checkpoint=True)`.

Temporary in-memory guards refuse any optimizer step, model forward, validation,
checkpoint save, or progress event. An observer verifies loaded AdamW values and
GPU moment placement, scheduler/GradScaler state, and restored CPU/CUDA/Python/
NumPy RNG. Every restored LoRA tensor must exactly match both `latest.pt` and the
completed adapter safetensors, normalizing only PEFT's `.default` name component.
Frozen base hashes and all source, canonical core, and arm file hashes must remain
unchanged. Exactly one model is loaded and no predictions are generated.

The only result is a timestamped `gpu_resume_noop_<stage>_<arm>_<UTC>.json` in this
audit directory. The report includes before/after hashes, parity checks, the
model load count/duration, and GPU memory peaks. Console output is compact; exit
status zero requires every check to pass. This proves restoration and parameter
parity, not forward/prediction parity or GPU update equivalence.
