# Cross-scale transformation alignment v2

This caller shares loading, canonical identity, capture, hashing, frozen data, registration and resource exclusion with Track A. Its calibration, objectives, decisions and evidence are independent. Historical graft source is imported through `shared.legacy.legacy_graft()` and remains untouched.

The scientific contract is [transformation_alignment_v2.md](protocol/transformation_alignment_v2.md); executable values are [alignment_v2.json](configs/alignment_v2.json). No real alignment or efficacy claim follows from tiny-model tests. B-Stage 1 is implemented; B-Stages 2/3 are gated and not executed.

Local GPU acceleration has a separate prospective [FP32 amendment](protocol/gpu_fp32_v2_1.md)
and [GPU configuration](configs/alignment_gpu_v2_1.json). The 4B decoder is streamed
one layer at a time; bridge autograd runs on GPU and ridge solves remain CPU FP64.
First freeze/commit the GPU configuration with the unchanged Track B manifest,
then run the independent runtime qualification and commit its passing receipt:

```powershell
& C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe experiments/supa_adaptive_compute_v2/cross_scale/scripts/qualify_gpu.py --config experiments/supa_adaptive_compute_v2/cross_scale/configs/alignment_gpu_v2_1.json --dataset-manifest experiments/supa_adaptive_compute_v2/data/generated/v2/cross_scale/manifest.json
```

For GPU capture/pair/aggregate commands, insert
`--config experiments/supa_adaptive_compute_v2/cross_scale/configs/alignment_gpu_v2_1.json`
before the action, and use a new GPU run ID. A committed qualification receipt
is required for capture and fitting; it licenses Stage 1 only. The existing CPU
commands below retain their original registration and RAM admission rule.

From the repository root in PowerShell:

```powershell
$researchPython = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
$alignmentCaller = 'experiments/supa_adaptive_compute_v2/cross_scale/scripts/run_alignment.py'
& $researchPython $alignmentCaller preflight
# Run this once only if the matching registration has not already been created:
& $researchPython experiments/supa_adaptive_compute_v2/scripts/freeze_protocols.py --track cross_scale --manifest experiments/supa_adaptive_compute_v2/data/generated/v2/cross_scale/manifest.json
# Commit the generated registration and all frozen sources/data before capture.
& $researchPython $alignmentCaller capture --run-id alignment-v2-001 --dataset-manifest experiments/supa_adaptive_compute_v2/data/generated/v2/cross_scale/manifest.json
foreach ($supaDepth in 8,12,16,19) {
  foreach ($donorDepth in 12,16,20,24) {
    $pairRoot = "experiments/supa_adaptive_compute_v2/cross_scale/evidence/02_bridge_objectives/alignment-v2-001/S${supaDepth}_D${donorDepth}"
    $attempts = @(Get-ChildItem -LiteralPath $pairRoot -Directory -ErrorAction SilentlyContinue | Where-Object Name -Match '^attempt_[0-9]+$')
    $completed = @($attempts | Where-Object {
      $receipt = Join-Path $_.FullName 'run_manifest.json'
      (Test-Path -LiteralPath $receipt) -and ((Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json).status -eq 'COMPLETE')
    })
    $unfinished = @($attempts | Where-Object {
      !(Test-Path -LiteralPath (Join-Path $_.FullName 'run_manifest.json')) -and
      !(Test-Path -LiteralPath (Join-Path $_.FullName 'abandonment.json'))
    })
    if ($unfinished.Count) { throw 'Unfinished worker: inspect its PID; explicit abandon-pair is required after it exits.' }
    if ($completed.Count -gt 1) { throw 'Duplicate successful attempts; aggregation is forbidden.' }
    if (!$completed.Count) {
      $nextAttempt = if ($attempts.Count) { 1 + ($attempts | ForEach-Object { [int]($_.Name -replace '^attempt_', '') } | Measure-Object -Maximum).Maximum } else { 0 }
      & $researchPython $alignmentCaller pair --run-id alignment-v2-001 --supa-layer $supaDepth --donor-layer $donorDepth --attempt $nextAttempt
      if ($LASTEXITCODE -ne 0) { throw 'Pair failed; inspect its failure receipt. Rerun this loop to create a new attempt.' }
    }
  }
}
& $researchPython $alignmentCaller aggregate --run-id alignment-v2-001
```

The loop skips completed pairs and resumes failed pairs in fresh attempt directories. Final aggregation independently verifies source/config/data, artifact hashes and complete 96-cell coverage. A failed pair is permitted only when the same pair also has exactly one compatible complete retry; all failures remain in the aggregate provenance. An unsuccessful early aggregation can be retried with a new `--attempt`. Never delete evidence to resume. Failed capture requires a new run ID. Capture and fitting hold the shared machine lease and require 3 GiB available physical memory; schedule them sequentially.

For an interrupted process that left `started.json` without terminal evidence, inspect its PID. After the process is gone, close that exact attempt explicitly before retrying:

```powershell
& $researchPython $alignmentCaller abandon-pair --run-id alignment-v2-001 --supa-layer 8 --donor-layer 12 --attempt 0
& $researchPython $alignmentCaller pair --run-id alignment-v2-001 --supa-layer 8 --donor-layer 12 --attempt 1
```

The recovery command refuses a live/uncertain PID and writes an additional abandonment receipt; original partial files stay intact. The caller never clears the programme lease automatically after an interrupted process; follow shared lease recovery after verifying its owner is gone.

Output: `evidence/01_alignment_matrix/<run-id>/capture.json`, `evidence/02_bridge_objectives/<run-id>/S<i>_D<j>/attempt_<n>/`, and `evidence/01_alignment_matrix/<run-id>/aggregate/attempt_<n>/selection.json`. Local activations/bridges are under `local/<run-id>/` and are hash-verified before reuse. The capture and all pairs must use the same committed registration.

The common future-generation adapter is `src/graft.py:CrossScaleOperator`, exposing `context(model)` and `refine(hidden, context)`. It wraps the already audited historical residual graft; this stage does not provide an efficacy runner or authorize a gate by constructing the adapter.

Next causal experiment: run the complete frozen alignment matrix. A mechanism NO-GO keeps generation closed. A mechanism GO is eligibility for a new small-screen freeze, not a capability result.

Mechanics verification, without real checkpoint loading:

```powershell
& $researchPython -m pytest experiments/supa_adaptive_compute_v2/cross_scale/tests -q
```

Tests use tiny FP32 tensors and tiny actual HF decoders with two CPU threads. They do not establish real-model alignment or capability. Capture checks its full elapsed budget at streaming-layer boundaries; optimization checks each full prompt step. A long forward cannot be preempted safely. Oversized calibration prompts fail instead of truncating the original final token.
