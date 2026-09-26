# Phase 1 harness review and independent results audit

Reviewed the amended `scripts/phase1_run.py`, `src/slcmi/training.py`, and
`src/slcmi/provenance.py` after the mechanics agent completed its changes.
No blocking source-review finding remains. The following statements are source
checks, not claims that future training has already succeeded.

* **Per-arm provenance:** each arm runs inside its own `Run`, output directory,
  manifest and resource receipt. Root aggregation stays PARTIAL until all
  required arm receipts are COMPLETE and compatible. Starting into a nonempty
  arm directory is refused. Legacy shared-arm output is refused.
* **Checkpoint choice:** `run_arm` passes the chosen canonical path and dtype
  to `load_model` and checks the resulting dtype. Canonical files are hashed
  before and after each arm and aggregation; frozen base parameters are hashed
  within each arm with the algorithm named. Historical algorithm variants are
  not compared across runs.
* **Pairing and rescoring:** `rescore_arm` reads typed `per_example.jsonl`,
  rejects duplicate/missing/unexpected IDs, and checks target metadata and
  preservation references. Expected target and preservation ID sets are common
  to the arms, and returned records follow the same expected ordering. Thus
  preservation aggregate comparison uses the same items even though `decide`
  itself only uses the scores.
* **Gate arithmetic:** a review found that floating subtraction could reject
  an exact threshold-boundary preservation regression. The mechanics agent
  fixed the gate to use `Fraction` comparisons. Full-run frozen baseline
  reproduction is mandatory, and smoke cannot license the later phase.
* **Training accounting:** a fixed attempt budget is distinguished from applied
  optimizer updates. Scaler skips, safety skips, and applied updates are tracked
  separately. Nonfinite or zero gradients cannot reach an unscaled optimizer.
  Only applied updates advance the schedule. Live progress persists completed
  attempts and resource counters; failed accumulation before the next completed
  attempt does not have a separate per-microbatch exposure receipt.
* **Gate/checkpointing:** gate disablement surrounds training; validation loss
  and generation use the restored gate. Context-manager cleanup restores gate,
  training mode, cache, and temporary checkpointing/input hooks on failure.
  Requested nonreentrant gradient checkpointing cannot silently fall back.
* **Resources:** aggregate wall time, loads, attempts, applied/skipped updates,
  and consumed examples are summed over arm receipts; memory peaks are maxima.
  Historical memory/resource values remain receipt measurements, not quantities
  recoverable from evaluation observations.

The independent CPU audit entry point is:

```powershell
& '../Chat-Bubble/.venv/Scripts/python.exe' 'experiments/supa_long_context_minimal_intervention_v1/scripts/audit_phase1_results.py'
```

Optional `--stages phase1_smoke`, `--stages phase1`, or
`--stages phase1 phase1_a1` restricts the audit. Default output is
`results/audit_20260905/phase1_numbers.json`, inside this lab. The script was run
successfully before new per-arm outputs existed and correctly reported
INCOMPLETE, without licensing anything. Rerun it after fresh results exist.

It independently rescores raw target predictions with the official
normalizer and original accepted answers. Raw preservation predictions
use the documented matched and exact rules against a freshly sampled reference
set. Every numerical gate input comes from per-arm `per_example.jsonl`. It
checks that legacy-named sidecar row files are identical to those observations,
validates ID/reference pairing, receipt core/dtype/gates/frozen-state checks,
matched dataset fingerprints/configuration, trajectory attempt/skip/application
partitions, full baseline validity, and exact rational decision boundaries.
Complete valid full runs receive a recomputed gate; missing or invalid inputs
cannot license the later phase. Resource sums/maxima are independently compared
with the root resource receipt and explicitly labeled as receipt-derived.

Do not modify lab source files after arm execution starts: production
aggregation requires exact current source-hash compatibility, including the
audit scripts. The report files may still be updated.
