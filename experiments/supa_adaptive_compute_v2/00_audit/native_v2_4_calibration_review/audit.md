# Independent v2.4 calibration scalar and provenance audit

Result: **PASS, 1,996 checks**, run on 2026-09-06 UTC. The standalone
`recompute.py` imports only Python standard-library modules; it does not import
programme implementations, Torch or NumPy, load a model, read checkpoint tensor
bytes, or use the GPU. Execution took 0.3494 seconds.

Audited source run:
`native_armr/evidence/00_learned_control_calibration/20260906T014443_419834Z_b790e5e7`.
The immutable registration digest is
`76914b192a4d880857170caa14cde8a6f1439737c37ab2a2c49fceff09390bab`.

| Reconstructed quantity | Fit | Holdout |
|---|---:|---:|
| Independent original scenarios | 96 | 32 |
| Recorded ordinary tokens | 7,069 | 2,338 |
| Equal-scenario mean native residual energy | 0.09559580669801247 | 0.09606493426847566 |
| Equal-scenario mean unscaled control energy | 0.1636214976718842 | 0.165333606512526 |
| Scaled control energy | 0.09559580669801247 | 0.09659610573025923 |

Independent `math.fsum` aggregation of per-scenario ordinary-token scalar
records, followed by the frozen square-root ratio, reproduced
**alpha = 0.764362266301519** exactly. No supplied summary value was used to
derive the scalar. The saved fit-scale artifact contains precisely the fit
summary and fixed formula; holdout values do not enter it. File modification
times are consistent with all fit case records preceding that artifact and all
holdout residual records following it. These timestamps are consistency evidence,
not an independently authenticated chronology.

The audit checked all 128 case hashes and exact IDs, unique scenarios, the fixed
96/32 allocation, frozen prompt hashes, family/template counts, full-token
coverage, stratum token counts and finite scalar metrics. It rehashed 149 bound
source files and the actual calibration config, native manifest and calibration
data, reconciling them with the saved registration and start/final manifests.
Only prompt/identity fields from the calibration JSONL enter the computation.

All three source/copy MLP and learned-norm digest pairs agree across provenance
and module receipts. Ordinary L0–L13, replayed L12/L13, L14 source-only, rotary
and composite state receipts have matching before/after metadata and zero
trainable/gradient parameters. Ordinary and replayed decoder hashes agree.
Geometry independently gives 28,311,552 matrix plus 3,072 norm coefficients,
28,314,624 resident/active copied coefficients, 113,258,496 FP32 parameter bytes,
56,623,104 dominant-linear FLOPs per token and zero newly learned coefficients.
Checkpoint/config/HF identities match the registration and preserved historical
checkpoint metadata.

Composite control-state digest:
`776e3a6471dc3825f16e36edf2c0b064939a9b17a59e064fcf42c206658b9d6a`.

The machine-readable `receipt.json` records every check and input digest. Its
SHA256 is `cdeec44ea31f9c3cd0dc4bdae8384777a4062b55aaac28a09b2f0225215cbc19`.

This audit reconstructs recorded scalar evidence. Raw hidden and residual arrays
are represented only by saved hashes, so it does not reproduce per-token energy
calculations or model forwards. Tokenization is not rerun. Module/checkpoint
digests are reconciled as execution metadata, not independently rehashed from
weights. Git commitment is left to the execution admission verifier. This is
calibration evidence and provides no efficacy or architecture-admission claim.

Exact command from the repository root:

```powershell
& 'C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe' 'experiments/supa_adaptive_compute_v2/00_audit/native_v2_4_calibration_review/recompute.py'
```

The script writes its receipt exclusively; an existing receipt is never replaced.
