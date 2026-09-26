# Mechanical audit, 2026-09-05

Audit scope: HANDOFF 4.5 and harness defects D1-D4, inspected before any new training. All commands ran from the repository root with `../Chat-Bubble/.venv/Scripts/python.exe`. No training or GPU model execution occurred. Existing canonical checkpoint files were read only. Source statements and line numbers below describe the pre-fix audit snapshot. After this report was completed, the coordinator authorized Part 2 harness corrections; those later edits deliberately change source hashes and do not change the historical audit finding that the completed old Arm A source hashes matched at audit time.

## Fresh adapter verification

**CONFIRMED, with limited scope:** the current CPU verifier passes on installed `peft 0.19.1`. Both B and C have 122,880 trainable parameters in 24 tensors, on layers 17/18/19 only, with the attention-only and protected-stem checks true. Both adapters start with zero B matrices and reproduce the frozen probe logits exactly. NoRA A-column mean norm is 1.0, maximum deviation 1.2e-7; ordinary LoRA mean is 0.030929 and remains unnormalised. These numbers were read from the newly generated receipt, not carried over from the previous receipt.

Executed command:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/scripts/phase1_verify_adapters.py'
```

Evidence: `results/phase1_adapter_verification/receipt.json` and `run_manifest.json`, start `2026-09-05T21:49:23.804073+00:00`, finish `2026-09-05T21:49:31.056137+00:00`, status PASS, `gpu.available=false`, both peak GPU counters zero. The manifest contains ten matching before/after canonical file hashes and `canonical_checkpoint_unchanged=true`.

The old four files were copied, without modification, to `results/audit_20260905/prior_adapter_verification/` before running the verifier. An initial rerun also passed, but an empty CUDA_VISIBLE_DEVICES value in PowerShell did not hide CUDA: the CPU model remained on CPU while resource accounting reported 491,520 GPU bytes. That first rerun was preserved under `prior_adapter_verification/first_rerun_cpu_with_cuda_accounting/`. The final rerun uses `-1`, and is the authoritative fresh receipt above. Both completed reruns recorded unchanged canonical hashes.

**UNCERTAIN beyond initialization:** reload and merge deltas are both zero, but the test only saves/reloads/merges adapters whose B matrices remain zero. It never calls backward, never creates a nonzero learned update, never verifies learned adapter weights directly, and uses only one short probe. This verifies initialization mechanics, not nonzero-adapter serialization/merge correctness, gradient flow, or successful training. `verify_serialisation_and_merge` reloads onto a deep copy of `get_base_model()` that already carries adapter modules; PEFT emits an existing-`peft_config` warning. A clean-base nonzero-adapter round trip remains desirable before interpreting any trained adapter artifact.

## HANDOFF 4.5 checklist

| Claim | Status | Evidence / settling command |
|---|---|---|
| PEFT targets L17/18/19 attention only | CONFIRMED for current initialization | Fresh verifier command and receipt above; `src/slcmi/adapters.py::build_lora_config`, `verify_adapter`. Installed version independently printed by `& '../Chat-Bubble/.venv/Scripts/python.exe' -c 'import peft; print(peft.__version__)'`: 0.19.1. |
| Protected layers 0/1/2 are refused | CONFIRMED | Fresh verifier directly tests layer 0; additional CPU import call independently called `build_lora_config(layers=[layer,17])` for each layer in `(0,1,2)`, and each raised the expected ValueError naming that layer. Code uses the set intersection with `(0,1,2)`. Current runner asserts protected-stem exclusion. No layer looping/removal/alteration mechanism exists in this adapter path. |
| Gate OFF for training and ON for every evaluation | CONFIRMED for target/preservation generation; REFUTED literally for every evaluation | `src/slcmi/training.py:176` encloses training and `evaluate_loss` calls at lines 231/245 inside `gate_disabled`; restoration is asserted at line 247. `scripts/phase1_run.py:172` asserts nonzero bias before target/preservation generation. Validation loss therefore runs gate-off. This is harmless for loss: canonical `modeling_supa_a11.py::forward` changes returned logits only after the parent computed loss. No fresh trained arm exists to verify run-time restoration after actual training. |
| MAX_LENGTH=2560 retains approximately all curriculum | PENDING separate data-design audit | Delegated to the data-design auditor's real-tokenizer measurement. A duplicate tokenizer measurement was interrupted before completion; no length result from that interrupted job is used. |
| Actual finite nonzero adapter gradients exist | UNCERTAIN / unsupported by saved artifacts | `rg -n 'grad_norm\|steps_applied\|steps_skipped\|trajectory\|NON_FINITE\|SKIPPED' experiments/supa_long_context_minimal_intervention_v1/results experiments/supa_long_context_minimal_intervention_v1/logs` (use `rg -e grad_norm -e steps_applied -e steps_skipped -e trajectory -e NON_FINITE -e SKIPPED` for literal shell reproduction) finds no training trajectories in existing results/logs. `rg --files .../results/phase1_smoke` lists only Arm A info/target/preservation plus shared manifests/summary/resource. `arm_A_info.json.training` is null. The fresh CPU verifier does no backward. |
| Scaler-skipped updates are counted | CONFIRMED in current code, UNCERTAIN in actual training | `training.py:206-214` compares GradScaler scale before/after the update, records each skip, and returns `steps_skipped_by_scaler` and `steps_applied` at lines 262-263. No B/C record verifies observed counts. |

Protected-layer standalone command (the same three calls used in the audit):

```powershell
@'
import sys
sys.path.insert(0, 'experiments/supa_long_context_minimal_intervention_v1/src')
from slcmi.adapters import build_lora_config
for layer in (0, 1, 2):
    try:
        build_lora_config(layers=[layer, 17])
        raise RuntimeError(f'layer {layer} unexpectedly allowed')
    except ValueError as exc:
        print(layer, str(exc))
'@ | & '../Chat-Bubble/.venv/Scripts/python.exe' -
```

The aggregate norm guard only raises for a zero norm on the first attempted step. Later zero norms are recorded without stopping; nonfinite norms are recorded but do not by themselves prevent an update. fp16 GradScaler normally handles nonfinite gradients, whereas bf16 uses a disabled scaler and still calls optimizer.step: this would apply NaN/Inf gradients. Also, scheduler.step is unconditional even after fp16 scaler skips. Thus `max_steps` means attempted updates, not the exact number of applied optimizer updates claimed by the train docstring/preregistration. Matching B/C attempted budgets does not guarantee matched applied budgets if overflow differs by initialization. `adapter_norm` is an absolute trainable-parameter norm, not distance travelled from initialization as its docstring suggests.

## Harness defects

**D1 CONFIRMED.** `provenance.py::Run` writes `started_manifest.json`, `run_manifest.json`, and `resource.json` with `overwrite=True` in the shared stage directory. `phase1_run.py` also overwrites `summary.json` per invocation. Current smoke `run_manifest.json` is COMPLETE with `config.arms=["A"]`; `started_manifest.json` is STARTED with `config.arms=["B"]`; only A output files exist. COMPLETE is an Arm A process receipt, not completion of the three-arm experiment. `--combine` skips missing arms, validates no configuration/source/core compatibility, sums no resources, and does not create a Run/checkpoint-hash stage. `decide()` reports FROZEN/no Phase 2 with A alone, which is not a trained null result. Missing required arms must remain INCOMPLETE and cannot license a scientific stop/winner decision.

**D2 danger CONFIRMED; one specific historical assertion REFUTED.** Run accepts populated directories and overwrites only the files the new process reaches, so obsolete results may survive. However, HANDOFF says smoke summary predates current code. Every source/config SHA in the completed smoke A manifest currently matches the file on disk: there were zero differences. The summary is the last completed A invocation and is stale relative to B's started manifest, not demonstrably from different current source code. No stale stage directories were deleted during this audit.

Executed comparison:

```powershell
@'
import hashlib, json
from pathlib import Path
root = Path('experiments/supa_long_context_minimal_intervention_v1')
m = json.loads((root/'results/phase1_smoke/run_manifest.json').read_text())
for path, recorded in m['source_sha256'].items():
    current = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if recorded != current:
        print('SOURCE_CHANGED', path, recorded, current)
print('run', m['config']['arms'])
print('started', json.loads((root/'results/phase1_smoke/started_manifest.json').read_text())['config']['arms'])
print('summary', json.loads((root/'results/phase1_smoke/summary.json').read_text())['config']['arms'])
'@ | & '../Chat-Bubble/.venv/Scripts/python.exe' -
```

Output: no SOURCE_CHANGED lines; run A, started B, summary A.

**D3 CONFIRMED.** `src/slcmi/training.py:192` uses `float(out.loss)` on a gradient-bearing loss. `float(out.loss.detach())` removes the warning without changing the value. `evaluate_loss` also converts loss, but its `@torch.no_grad()` makes that separate conversion harmless.

**D4 current implementation CONFIRMED; historical chronology UNCERTAIN.** `provenance.py::state_hash` hashes names, original dtype/shape, and streamed native bytes. Its comment describes an earlier float32 implementation, but the old executable implementation is not retained as a separately auditable artifact in this lab. Do not compare stored cross-stage model-state digests unless the hashing source matches. Within-run comparisons remain valid. Canonical *file* SHA-256 is a separate stable algorithm and remains comparable. The claim that native fp16/bf16 hashing detects changes that fp32 upcasting would round away is inaccurate: conversion of fp16/bf16 finite values to fp32 is exact; its clear benefit here is memory use and dtype-aware encoding.

**D5 additional blocker CONFIRMED.** `scripts/phase1_run.py:129` ignores its `core` argument and calls `load_model(CANONICAL_CORE, device)`. Main labels/records/hashes `CORES[args.core]` and uses that core's tokenizer, but `--core a1` would load A1.1 weights in default fp16. `evaluation.py:82` uses `DTYPE` (float16) when dtype is omitted, despite a docstring claiming native dtype detection. Fix both the runner's chosen path and its explicit `dtype=core['dtype']` before a1 execution. Existing Phase 0 uses an explicit selected core/dtype and is not implicated by this Phase 1 runner bug.

These are audit findings only. The CPU verification PASS supports the intended trainable set and zero initialization. It supplies no evidence that a trained adapter has finite nonzero gradients, that a completed B/C smoke run exists, or that Phase 1 has a scientific outcome. Later Part 2 results, if produced, must establish those separately.
