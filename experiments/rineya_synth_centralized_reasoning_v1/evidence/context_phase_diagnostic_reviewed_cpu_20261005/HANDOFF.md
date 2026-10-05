# Staged NO_MODEL context-phase diagnostic

This separately scoped diagnostic prepares observation of imports, CUDA context
and memory conditions. It is not an unchanged M0 retry, model-fit result or cure
for any historical abort. Native stage dispatch/preflight/worker entrypoints
reject before owned controls, imports, resource probes or output creation.

The contract admits only `CUDA_CONTEXT_PHASES_NO_MODEL_ONLY` and an exact closed
manifest. Model directory/config/weights/pins, rows, loaders, model construction,
forward/backward/training/grading and cache/device-reset authority are absent.
Own contract/worker/supervisor, all package/test and named transitive helper
sources form an exact roster; resigned omissions fail. Helpers agree with their
source pins. Preflight requires committed owned config, exact current source and
runtime65 closure (old64 plus AST), runtime versions, offline ENV, resource bounds,
tolerances and profile. No actual config or native launch was created here.

The lazy worker imports Torch and the Qwen3 configuration/model **symbols**;
it does not instantiate either class or call any loader/tensor/forward function.
It separately records attempt/return/failure with UTC, monotonic clock and ordinal
for imports, versions, the original psutil post-import growth observation,
CUDA initialization, direct free/total and allocator reads, precision/thread
configuration, seed operations, peak-counter reset and final reads. Peak-memory
statistics reset is the original worker's process allocator-counter operation;
it is not device reset, empty-cache control or resource reclamation. Internal
framework/context behavior may allocate memory, which this diagnostic observes;
no explicit tensor or model allocation is requested.

The original `Journal` and `make_guard` sources remain unchanged. A separate
Recorder has no guard callback. A guard-only Torch/CUDA view invokes each actual
`is_initialized`/`mem_get_info` callable once and returns its exact object/result
or exception. False initialization bypasses the memory query exactly as before.
Guard start/end boundaries and query boundaries are separately accounted, without
calling Journal or recursively invoking guards. Publication errors are retained
without masking an actual guard error; successful query results return unchanged
and separate recorder-health checks fail the scope afterward.

This closes a specific observability gap: Journal emits CUDA-init's return before
its after-call guard. That guard's memory query precedes the worker's explicit
4GiB preload query, configuration and seeding, so the historical last returned
CUDA-init event identifies none of those unlogged locations. An attempted phase
marker shows a pending boundary, not an exact OS program counter or causal stall.

The owned dispatcher retains the existing coherent monitor, fresh admission,
Job assignment, external NVML authority, sampler join, kill/close containment
and inventory freeze. CPU cadence100ms/freshness200ms, preload GPU4GiB, work
GPU0.5GiB, RSS2GiB, RAM1.25GiB plus original growth reserve, disk4GiB, admission
reserve and whole-scope600s remain unchanged. Optional Windows PID counters are
explicitly unsupported until a separately pinned provider is reviewed; they
cannot replace NVML, imply zero use or prove process/cause attribution.

Twelve CPU controls pass using synthetic APIs only. They cover exact proxy
returns/exceptions/query counts, false-init bypass, same RAM/RSS/deadline/heartbeat/
GPU thresholds, poisoned version/growth/allocator observations, after-init guard
failure, diagnostic IO failures, source/authority/ENV omissions and stage native
refusal. A fresh-root copy of actual reviewed files passes all12 with no ignored
stage dependency. No framework, CUDA/NVML/Win32, model, tokenizer/decoder, GPU,
large runtime binary hashing or actual grading occurred in preparation.

```powershell
python -I -B -m unittest discover -s experiments/rineya_synth_centralized_reasoning_v1/artifacts/context_phase_diagnostic_work -p test_context_phase_diagnostic.py -v
python -I -B experiments/rineya_synth_centralized_reasoning_v1/artifacts/context_phase_diagnostic_work/simulate_context_phase_fresh_root.py
```

Independent source/CPU review, tracked preservation and explicit source-bound
native manifest/admission remain required before any integration or diagnostic
launch. Preserve original scopes and all failed evidence. Diagnostic success
cannot qualify model loading, a raw M0 batch, long context, training, scoring,
transfer or retention; those gates stay closed.
