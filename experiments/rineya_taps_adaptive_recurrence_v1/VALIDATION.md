# Validation record

New CPU suite: **62 passed**, 0 failures,
0 errors, 0 skipped. Final pytest wall
reported by the runner: 4.201 seconds. `compileall` exited 0.
See `results/validation.json` and the full JUnit/log files in the handoff bundle.

Three unchanged inherited ARMR numerical assertion functions passed in isolation:
contracting dynamics, oscillating dynamics and gold-answer extraction. Their full
source files were matched to the exact audited Git blob SHA-1s before execution.
The isolated runner does not substitute fake native dependencies or claim the
remaining gate tests ran. A targeted inherited module collection was attempted
and exited 2 because `armr.reporting` is missing from the scoped materialization.
The complete inherited checkout and Transformers are unavailable here; the full
historical ARMR suite remains BLOCKED, not '65 passed' by reference to old results.

The new tests cover all fifteen requested categories: disabled/alpha-one identity,
analytical fixed scaling, EMA/bias correction, finite statistics, bounds, refresh,
reset, batch isolation, reproducibility, logging invariance, DEV/TEST oracle
separation, random schedules, invalid configs and memory guards. Additional checks
cover fixed-grid completeness, duplicate rows, padding, stable sequence identities,
label independence, source-host cleanup, cache rejection, exact position/mask reuse,
whole-block warmup, no training, extended-horizon refusal and exhaustive-oracle
physical compute accounting. All are CPU apparatus tests, not native efficacy.

`fixture_v1` was the earlier apparatus run before adding source/runtime manifests.
It is retained in the bundle. `fixture_v2` adds those provenance receipts. `fixture_v3` uses the final tested
source after the native-only causal diagnostic correction and two new tests. No fixture task, threshold, alpha grid or
controller parameter was tuned after inspecting the earlier outcomes. The changes
were provenance/firewall/reporting corrections, not efficacy selection.

Native preflight exited 2 and recorded BLOCKED; no checkpoint load or CUDA run was
attempted. Review `results/native_preflight_v1/preflight.json` for exact bytes,
versions and missing-dependency reasons. Its free-memory measurement is a snapshot,
not a claim about the user's RTX 4070 laptop.

Final static review identified that sequence-wide adaptive statistics make the
inherited full-gold optional NLL noncausal and could contaminate the random pool.
The native lane now disables that optional diagnostic without changing the scorer
or generation bytecode. Two regression tests cover gold exclusion and no global
mutation. A5 also receives its own smoke gate after freezing the DEV alpha pool.
No native execution occurred before or after this correction.
