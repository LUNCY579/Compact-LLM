# Independent B XM calibration audit — 2026-09-13

**PASS within the evidence scope below.** The independent auditor uses Python's
standard library, imports no experiment implementation and executes no models.
The main receipt records 3,373 checks and 310 input hashes. A separate commitment
receipt verifies all 132 completed artifacts in commit
`2ae253ff0e16170b9958ddbe5f6e6f4c67c03f22` (265 checks). No source, configuration,
dataset, checkpoint or historical evidence was edited. No Git mutation occurred.

Audited report:
`cross_scale/generation_stage/v1/evidence/00_control_calibration/20260913T171252_566611Z_669e5d42/report.json`
under the programme root. Report SHA-256:
`ac87b318d56345aa5ce4719781ea92c4be8b57afcc69b532402abae9a1e8145d`.
Preregistration:
`bf99e713104573b7aafb8f5511278fd473bcd406ff1d6fece3d0ebe5787534a0`.
Actual pre-execution commit:
`452bcf5753367106c4d5397991f4ef49d1243097`.

The auditor independently recomputed all 96 fit and 32 held-out scenario
aggregates from the 128 per-case JSON records. Equal scenario weighting gives
`alpha = sqrt(mean(teacher energy) / mean(control energy)) = 1.6186783995925944`.

| Partition | Scenarios | Ordinary tokens | Teacher energy | Unscaled control energy | Scaled control energy |
|---|---:|---:|---:|---:|---:|
| Fit | 96 | 7,150 | 0.09821151170407749 | 0.03748359641967753 | 0.09821151170407749 |
| Holdout | 32 | 2,395 | 0.09855854455727829 | 0.03751560355267849 | 0.09829537422575611 |

Coverage checks independently establish unique original Track B calibration
IDs, the fixed scenario partition, no overlap, exact raw receipt inventory,
whole prompt lengths and token-stratum counts. The full immutable calibration
container is hashed; only prompt and identity fields are consulted in its rows.
No efficacy, confirmation, Track A data or sealed data is opened.

The audit checks all 169 preregistered source hashes against current files and
70 programme source/config/data/export/capture/registration files against the
actual pre-execution commit. Historical files outside this programme are bound
by their exact preregistered hashes; this does not invent a commit for the
preserved untracked historical graft implementation. Export and activation
container hashes are recomputed without tensor deserialization.

State receipts bind the same selected XG bridge, donor layer 24, post-Supa-L19
capture and XM state
`79f84b142a5708527fa7833e3df02f2939ffb21b1e62df321a0300336ad707d4`.
All recorded module parameters are frozen and gradients zero; checkpoint,
parameter-state and buffer hashes match before/after. Registered matrix cost is
106,168,320 coefficients and 212,336,640 FLOPs per token. The successful run
records 63.7219281 seconds and 2,019,373,056 bytes peak RSS.

Limits: the auditor recomputes scalar aggregates, not the tensor operations that
produced each recorded per-case energy. Module and checkpoint preservation are
checked as hash-bound run records, without an independent live tensor snapshot.
Fit-before-holdout order follows the frozen source and exclusive fit marker;
there are no independently timestamped per-case records proving chronology.
The label-free access boundary is supported by the frozen implementation and
declared receipts, not an independent operating-system file-access trace.
This scalar calibration establishes neither CPU/GPU qualification nor donor
efficacy. The next Track B action remains the frozen generation runtime
qualification and prospective dose screen.

Reproduce from the repository root, selecting a new output directory:

```powershell
& 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_adaptive_compute_v2/00_audit/cross_control_calibration_review_20260913/audit_calibration.py' --report 'experiments/supa_adaptive_compute_v2/cross_scale/generation_stage/v1/evidence/00_control_calibration/20260913T171252_566611Z_669e5d42/report.json' --output 'experiments/supa_adaptive_compute_v2/00_audit/cross_control_calibration_review_20260913/attempt_002'
```
