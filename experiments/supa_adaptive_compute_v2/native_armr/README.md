# Native ARMR: Track A

Evaluate frozen A0, A1 (L12), A3 (L12-L13), A16 (L16) and AM (independent frozen
SwiGLU with A3-matched dominant matrix work). All generation, data guards,
scoring, statistics, manifests and integrity checks use the shared pipeline.
Candidate decisions, protocol, data roles and evidence belong to this track.

Read `protocol/preregistration.md` and `configs/protocol.json`. They freeze
the scientific choices; effective execution also requires the shared registration
receipt and committed source. No native large-evaluation result is claimed by
the presence of this caller or its tests.

GPU execution has a separate prospective amendment,
`protocol/gpu_fp32_v2_2_amendment.md`, and configuration
`configs/protocol_gpu_v2_2.json`. It preserves the five arms, data and scientific
gates while using CUDA FP32/eager/no-cache and separate host/VRAM admission.
After freezing and committing this GPU registration, run the programme's
`scripts/qualify_native_gpu.py` CPU/reference comparison before science. A
compatible completed qualification receipt is mandatory for GPU preflight,
execution and aggregation. Then pass
`--config experiments/supa_adaptive_compute_v2/native_armr/configs/protocol_gpu_v2_2.json`
to every caller and auditor command below. All paired runs must belong to that
same GPU registration. The commands without `--config` retain the original CPU
protocol and its 3 GiB host-memory requirement; no CPU failure receipt is altered.

From the repository root, use the qualified environment:

```powershell
$py = 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe'
& $py experiments/supa_adaptive_compute_v2/native_armr/scripts/run_large_eval.py --help
& $py -m pytest -q experiments/supa_adaptive_compute_v2/native_armr/tests
```

After generating/freezing data and committing registration, assign the frozen
Track A manifest path to `$manifest`. `--preflight` loads no model and writes
no evidence. Its output reports the effective item token cap, currently 32;
48 is only the protocol-wide upper bound. Run the architecture matrix first:

```powershell
$manifest = 'experiments/supa_adaptive_compute_v2/data/generated/v2/native_armr/manifest.json'
$caller = 'experiments/supa_adaptive_compute_v2/native_armr/scripts/run_large_eval.py'
& $py $caller --manifest $manifest --role architecture_dev --arm A0 --shard-count 1 --preflight
foreach ($arm in @('A0', 'A1', 'A3', 'A16', 'AM')) {
    & $py $caller --manifest $manifest --role architecture_dev --arm $arm --shard-count 1
    if ($LASTEXITCODE -ne 0) { throw "Architecture stage failed for $arm; preserve receipt and stop." }
}
```

All 128 architecture cases and all five arms must pass the mechanical gate
before confirmation. The shared caller enforces this prerequisite automatically.
Save the five printed `RUN_DIR` values in a JSON array `native-architecture-runs.json`
and run the independent audit:

```powershell
& $py experiments/supa_adaptive_compute_v2/native_armr/scripts/audit_results.py --manifest $manifest --role architecture_dev --run-list native-architecture-runs.json
```

After a
mechanical PASS, start confirmation with one worker and 32 deterministic shards:

```powershell
& $py $caller --manifest $manifest --role confirmatory_dev --arm A0 --shard-index 0 --shard-count 32 --preflight
& $py $caller --manifest $manifest --role confirmatory_dev --arm A0 --shard-index 0 --shard-count 32
& $py $caller --manifest $manifest --role confirmatory_dev --arm A3 --shard-index 0 --shard-count 32
```

Schedule A0 then A1/A3/A16/AM for every shard index 0–31, one worker at a time.
Run the separate 128 linked diagnostic variants with
`--role confirmatory_diagnostics --shard-count 1`, again A0 then all candidates.
Each attempt gets new immutable evidence. Resume by choosing only the missing
arm/shard combinations; do not rerun completed shards or mix failed attempts.

Use `scripts/aggregate.py` or its `scripts/audit_results.py` entry point with
explicit successful run directories after the entire role completes. To avoid
the Windows command-length limit with 160 confirmatory run directories, create
a JSON array containing their full paths, then run:

```powershell
& $py experiments/supa_adaptive_compute_v2/native_armr/scripts/aggregate.py --manifest $manifest --role confirmatory_dev --run-list native-confirmatory-runs.json
& $py experiments/supa_adaptive_compute_v2/native_armr/scripts/aggregate.py --manifest $manifest --role confirmatory_diagnostics --run-list native-diagnostic-runs.json --primary-run-list native-confirmatory-runs.json
```

Both entry points independently reconcile per-case and aggregate raw evidence,
recompute the frozen scores and validity, check model/runtime/control identity,
then save a new immutable `evidence/04_independent_audit/` receipt. The analysis
includes A3 vs A0 followed by A3 vs AM, secondary A1/A16 Holm tests, the registered
pooled preservation bound, family and held-out-template effects, NLL, output
identity and costs. Missing arms, partial banks, failures, duplicate attempts
or mismatched evidence cannot yield GO. Architecture and diagnostic analyses
never license an efficacy claim. The auditor does not repeat model inference.
Diagnostic auditing additionally requires the complete original-bank evidence;
it reports whether each arm's paired advantage survives each linked variant,
using the original scenario as the resampling unit. The quality/cost frontier
retains accuracy, latency, matrix work, added distinct parameters and peak
process memory separately; it does not invent a scalar utility score.

The reused ARMR wrappers come from
`experiments/anchor_routed_middle_recurrence_v1/src/armr/` and remain unchanged.
AM is a new untrained generic compute control; it does not demonstrate what a
learned same-scale transformation could do. Sealed data and later mechanisms
remain closed.
