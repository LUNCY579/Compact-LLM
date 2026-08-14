# Phase 2 Experimental Protocol

## Scientific gate

Mechanistic interpretation begins only after
`reports/PHASE2_CORNERSTONE_VERIFICATION.md` reports PASS. The gate independently
checks raw rows, condition pairing, official split hashes, config/model/adapter
hashes, scoring, step traces, EOS, the pilot archive, and paired statistics.

## Data roles

| Split | N | Required label | Permitted role |
|---|---:|---|---|
| v2.2 TRAIN | 32,400 | TRAIN MECHANISTIC / EXPLORATORY | Hypothesis construction, descriptive mechanisms, auxiliary diagnostics |
| v2.2 DEV | 4,050 | DEV MODEL-SELECTION / CALIBRATION | Temperature fitting, thresholds, comparator selection, committee promotion |
| v2.2 TEST | 4,050 | TEST CONFIRMATORY | Frozen comparisons and selected predeclared mechanism statistics |

The archived n=32 pilot sampled the same TEST split. The full-run method/config
hash is unchanged, but the paper must disclose that prior partial TEST exposure.
No new threshold, committee, or aggregator may be selected on TEST.

## Stage 0: cornerstone verification

Inputs:

- `results/sealed/v22_frozen_finalists.jsonl`
- `artifacts/frozen_configs/frozen_final_config.json`
- `artifacts/frozen_configs/sealed_finalists.yaml`
- `results/sealed/sealed_run_manifest.json`
- official reviewed TRAIN/DEV/TEST JSONL
- before/after official-source manifests
- voting, decoding, scoring, and statistics source

Outputs:

- `reports/PHASE2_CORNERSTONE_VERIFICATION.md`
- `reports/phase2_cornerstone_verification.json`

## Stage 1: existing full-TEST traces

Without new inference, compute paired rescue/harm globally and descriptively by
task, difficulty, answer class, domain, hard-output length, and hard-vote
agreement. Analyze first hard/soft divergence only while both methods share the
same prefix. Enumerate winner coalitions, expert directional-confidence
profiles, vote sizes, and pairwise top-1 agreement.

Outputs:

- `reports/PHASE2_RESCUE_HARM.md`
- `results/phase2_mechanism/existing_results_analysis.json`
- machine-readable rescue/harm, coalition, expert, and pairwise CSV files

## Stage 2: teacher-forced DEV cache

For every DEV example, append one durable row after one teacher-forced forward
per expert. Store, for every gold target token including EOS:

- expert top-64 token IDs and logits;
- union-candidate logits for every expert;
- top-1/top-2 probabilities and margin;
- entropy, NLL, multiclass Brier score, logit mean/std/norm;
- gold logit/probability;
- exact log-sum-exp values on the frozen positive-temperature grid.

Do not store full-vocabulary tensors. For probability/logit aggregators,
certify that the winning union candidate beats the maximum possible score of
any omitted token. If certification is below 100% for a promoted comparator,
increase top-k and recapture before confirmation.

The capture is append-durable, ID-deduplicated, manifest-hashed, resumable, and
reported live. TRAIN is not required for fitting scalar temperatures because
DEV is the designated calibration split.

## Stage 3: cached committee and calibration analysis

Using the DEV cache:

1. fit shared and per-expert scalar temperatures by gold-token NLL;
2. calculate top-1 accuracy, NLL, Brier, ECE-15, entropy, margin, and reliability
   bins per expert, plus confidence/entropy/margin conditioned descriptively on
   whether the expert's top-1 token is correct;
3. evaluate all (2^7-1=127) nonempty subsets under hard plurality, raw and
   calibrated probability mean, raw and normalized logit mean, top-2 approval,
   top-5 Borda, and reciprocal-rank aggregation;
4. report both best-at-(K) and mean-across-committees-at-(K);
   attach descriptive example-level intervals, while labeling best-at-K
   intervals as conditional on same-DEV selection;
5. calculate K=7 leave-one-out deltas, exact Shapley values, and selected
   pairwise interactions;
6. perturb each expert temperature separately over the frozen grid and verify
   that hard top-1 proposals remain unchanged; separately recompute the
   confidence-sum hard tie-break to measure its unequal-temperature boundary;
7. run bounded post-selection robustness controls for inverse-entropy expert
   weighting and top-1 confidence clipping at caps 0.5, 0.7, and 0.9, with
   omitted-vocabulary winner certification;
8. promote only a small scientifically motivated comparator family.

All Stage 3 results are DEV model-selection evidence.

## Stage 4: free-running DEV

Freeze from Stage 3:

- raw hard K=7;
- raw probability K=7;
- per-expert calibrated probability K=7;
- at most one rank-only comparator;
- all seven K=1 experts;
- K=7 leave-one-out committees;
- scientifically relevant best/mean committee-size representatives;
- adaptive K=3→K=7 consultation threshold fitted without TEST.
- a targeted hard-vote systems panel containing the teacher-forced best
  committees at K=1…5 and all seven K=6 leave-one-out committees; reuse the
  selected full-K=7 hard result as K=7.

Run full DEV with ordinary EOS, the no-cache reference decoder, durable
checkpoints, and identical scoring. Compare teacher-forced local decisions with
free-running exact-match behavior on the same DEV examples.

## Stage 5: frozen confirmation

Before computing a new TEST statistic, write a confirmation manifest containing
the exact source/cache hashes, selected methods, temperatures, thresholds,
comparison family, expected outputs, and no-switching rule. Confirm only the
minimal mechanism-distinguishing subset. Apply exact McNemar tests and Holm
correction to the declared family.

The v2.2 calibration confirmation was frozen before its output existed in
`artifacts/phase2_mechanism/v22_calibrated_test_confirmation_manifest.json`.
It adds exactly one full-K=7 calibrated-probability condition under config hash
`e4f77f7c7d68dc332eadc4decfb839d3a4b526addefd5c61538d92cf58df2794`.
Its separate two-comparison family is calibrated versus the already-known hard
and raw-probability cornerstone anchors. The archived n=32 TEST exposure and
the fact that both anchors were known before this freeze are mandatory caveats;
no method, temperature, committee, or comparison may be changed after output.

## Cross-benchmark confirmation

Verify v2.1.2 general-reasoning TEST provenance and training/selection exposure.
Report it separately from v2.2; do not form a post hoc aggregate. Any prompt or
scoring differences must use the benchmark's official evaluation pathway.

## Data dependence and semantic audit

Before paper-level power claims:

- detect exact and normalized duplicates;
- group deterministic template/task families;
- evaluate per-example and family-cluster bootstrap intervals;
- run lexical near-duplicate sensitivity analysis;
- preserve exact match as primary;
- perform a secondary blinded semantic audit on a preselected error subset.

## Statistics

- paired bootstrap confidence intervals over examples;
- exact McNemar tests for paired correctness;
- Holm correction within each declared confirmatory family;
- cluster bootstrap by template/task family where applicable;
- effect sizes and practical-significance interpretation;
- descriptive subgroup intervals clearly separated from confirmatory tests.

## Detected infrastructure

The machine-readable record is
`artifacts/phase2_mechanism/environment.json`; the paper-facing rendering is
`reports/PHASE2_ENVIRONMENT.md`. It is generated by
`experiments/phase2_mechanism/capture_environment.py` and records the actual OS,
CPU, RAM, GPU/VRAM, driver, Python, CUDA, cuDNN, PyTorch, Transformers, PEFT,
Accelerate, Safetensors, NumPy, SciPy, and resolved BF16 inference dtype. Measured
latency, TTFT, VRAM, and RSS remain condition-level results rather than portable
hardware claims.

## Reproduction order

```powershell
$Python = "..\.venv\Scripts\python.exe"

& $Python experiments\phase2_mechanism\verify_cornerstone.py
& $Python experiments\phase2_mechanism\analyze_existing_results.py
& $Python experiments\phase2_mechanism\analyze_coalition_strata.py
& $Python experiments\phase2_mechanism\analyze_negative_controls.py
& $Python experiments\phase2_mechanism\capture_teacher_forced_dev.py --limit 4050 --progress-every 50
& $Python experiments\phase2_mechanism\analyze_teacher_forced_cache.py
& $Python experiments\phase2_mechanism\analyze_robust_soft_controls.py
& $Python experiments\phase2_mechanism\analyze_confidence_conditioning.py
& $Python experiments\phase2_mechanism\analyze_committee_diagnostics.py
& $Python experiments\phase2_mechanism\analyze_pairwise_diversity_extended.py
& $Python experiments\phase2_mechanism\analyze_hard_temperature_ties.py
& $Python experiments\phase2_mechanism\select_teacher_forced_cases.py
& $Python experiments\phase2_mechanism\freeze_selected_free_run.py
& $Python scripts\verify_cache_parity.py --config configs\phase2_selected_dev_free_run.yaml --dataset v22_dev --committee full_k7 --limit 8 --device auto --precision auto
& $Python experiments\phase2_mechanism\verify_rank_topk_parity.py
& $Python scripts\run_dev.py --config configs\phase2_selected_dev_free_run.yaml --dataset v22_dev --device auto --precision auto --summary-every 25
& $Python experiments\phase2_mechanism\analyze_selected_free_run.py
& $Python experiments\phase2_mechanism\analyze_tie_eos_counterfactuals.py
& $Python experiments\phase2_mechanism\fit_adaptive_teacher_forced.py
& $Python experiments\phase2_mechanism\freeze_adaptive_dev.py
& $Python scripts\run_dev.py --config configs\phase2_selected_adaptive_dev_free_run.yaml --dataset v22_dev --device auto --precision auto --summary-every 25
& $Python experiments\phase2_mechanism\analyze_adaptive_free_run.py
& $Python experiments\phase2_mechanism\freeze_selected_committee_dev.py
& $Python scripts\run_dev.py --config configs\phase2_selected_committee_dev_free_run.yaml --dataset v22_dev --device auto --precision auto --summary-every 25
& $Python experiments\phase2_mechanism\analyze_selected_committee_dev.py
& $Python experiments\phase2_mechanism\freeze_v22_calibrated_test.py
& $Python scripts\run_dev.py --config configs\phase2_v22_calibrated_test_confirmation.yaml --dataset v22_sealed --device auto --precision auto --summary-every 25
& $Python experiments\phase2_mechanism\analyze_v22_calibrated_test.py
& $Python experiments\phase2_mechanism\verify_cross_benchmark.py
& $Python experiments\phase2_mechanism\freeze_cross_benchmark.py
& $Python scripts\run_dev.py --config configs\phase2_cross_v212_controls.yaml --dataset v212_test --device auto --precision auto --summary-every 25
& $Python scripts\run_dev.py --config configs\phase2_cross_v212_full_k7.yaml --dataset v212_test --device auto --precision auto --summary-every 25
& $Python experiments\phase2_mechanism\analyze_cross_v212.py
& $Python experiments\phase2_mechanism\make_final_synthesis_facts.py
& $Python experiments\phase2_mechanism\make_paper_figures.py
& $Python experiments\phase2_mechanism\make_paper_tables.py
& $Python experiments\phase2_mechanism\capture_environment.py
& $Python experiments\phase2_mechanism\update_live_report.py
& $Python -m pytest -q
& $Python experiments\phase2_mechanism\verify_final_synthesis.py
& $Python experiments\phase2_mechanism\audit_phase2_completion.py
```

The free-running command is append-durable and resumable. Re-running the exact
command skips completed method-example keys after verifying the frozen config
hash. Do not substitute the optimized cache decoder: its Phase 2 parity gate is
retained as negative evidence in E-DEV-CACHE-PARITY.

### Rank-comparator memory parity and durable resume

The selected DEV sweep encountered CUDA allocator exhaustion after entering
the reciprocal-rank block. Completed JSONL rows were already append-durable,
unique, and config-hash bound. The original stable full-vocabulary sort was
replaced by a stable partial top-k that returns the same top-five token IDs and
token-ID tie ordering without materializing a vocabulary-sized permutation.
`verify_rank_topk_parity.py` establishes exact equality against the original
operation with 15,309 exhaustive vector/rank checks, 4,096 randomized
tie-stressed row/rank checks, and a 7 x 151,936 realistic-vocabulary check. Its
source-bound artifact is `rank_topk_parity_verification.json` and is a required
completion-audit gate.

After the selected run reaches 16,200 rows,
`selected_dev_resume_history.json`,
`selected_dev_runtime_boundaries.json` and
`PHASE2_SELECTED_RUNTIME_BOUNDARIES.md` bind the exact raw-line boundaries,
current runtime hashes, and observed retry totals. The strict selected analyzer
creates and hashes these artifacts before computing paper statistics.

Unused CUDA allocator blocks are also released between independent examples.
This cleanup occurs after generation latency and peak-memory measurement and
after the durable row append; it does not alter live tensors, logits, winners,
traces, or per-example measurements. Resumed rows retain the same frozen
config hash and mathematical aggregation rule.

Raw JSONL is flushed on every example. Derived summary/live-report rebuilds
use the requested cadence for small runs and a minimum interval of
`ceil(total_conditions / 200)` for large panels, bounding repeated full scans
without changing any raw row; final derived artifacts are always rebuilt.

If CUDA allocation still fails inside an example, the evaluator discards that
uncommitted attempt, releases unused blocks, and recomputes the identical
example at most twice. Only a successful generation is appended, and each new
row records `cuda_oom_retry_count`. Successful-generation latency excludes any
failed allocation attempt, so it remains a decoder measurement rather than an
end-to-end fault-recovery measurement. The selected-run supervisor also caps a
worker at 100 newly durable rows and then reloads the same frozen condition in
a fresh process; this bounds WDDM CUDA-context fragmentation without changing
row order, keys, or scores.

After the selected four-method DEV run reaches exactly 16,200 rows, the
remaining frozen run/analyze/regenerate/test sequence can be resumed as one
ordered pipeline:

```powershell
& $Python experiments\phase2_mechanism\run_remaining_pipeline.py
```

If an already-running selected DEV worker should be supervised across durable
resumes, use:

```powershell
& $Python experiments\phase2_mechanism\run_remaining_pipeline.py --wait-for-selected --poll-seconds 30
```

The supervisor reads the PID lock before launching any model process. It never
loads a second committee while that owner is alive; if the owner exits before
16,200 rows, it invokes the identical frozen resumable command. Current row and
lock snapshots plus the last 20 resume attempts are written atomically to
`state/phase2_remaining_pipeline.json`; a new supervisor inherits that restart
history rather than resetting it.

The pipeline intentionally does not run the final synthesis verifier or the
completion audit. First update the mechanism synthesis and claim ledger from
the actual long-run outputs, run `verify_final_synthesis.py` to hash-bind those
documents to the completed analyses, then run `audit_phase2_completion.py` as
the final certification step.

## Final evidence binding

The consolidated numerical packet is `final_synthesis_facts.json`. Exact rank
implementation equivalence is recorded in `rank_topk_parity_verification.json`;
attempt-level durable-resume history is recorded in
`selected_dev_resume_history.json`; and the immutable raw-line implementation
transition record is `selected_dev_runtime_boundaries.json`. These artifacts
separate scientific outputs from allocator/restart bookkeeping and are checked
again by the final audit.
