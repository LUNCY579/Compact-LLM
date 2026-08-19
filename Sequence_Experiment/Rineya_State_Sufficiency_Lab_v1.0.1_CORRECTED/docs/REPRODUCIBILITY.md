# Reproducibility

## Freeze these inputs

- base model files and tokenizer;
- dataset archive hash;
- prepared data file hashes;
- training and generation configs;
- package lock or environment export;
- CUDA driver/runtime information;
- random seeds;
- TEST unlock time and preregistration.

## Manifests

Every training and evaluation output directory contains a resolved YAML config and `run_manifest.json`. The manifest records dataset hashes, environment, model settings, status, and errors.

## Determinism

Set seeds for Python, NumPy, PyTorch, and Trainer data order. Exact bitwise reproducibility is not guaranteed across different GPU architectures, CUDA kernels, quantization versions, or distributed topologies. Report that limitation rather than silently comparing incompatible runs.

## Resume behavior

Training uses the last Trainer checkpoint when `resume_from_checkpoint: auto`. Evaluation treats `request_id` as idempotency key and skips completed rows. Append writes are flushed and synchronized.

## Release verification

Run:

```bash
python -m compileall -q src tests
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m rineya_state_lab smoke-test --work-dir outputs/smoke --force
python tools/verify_release.py .
```

The build report records the commands actually executed for this release.
