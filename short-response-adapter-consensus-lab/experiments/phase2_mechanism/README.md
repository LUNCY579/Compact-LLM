# Phase 2 Mechanism Discovery

This directory isolates the mechanism-discovery phase from the completed
phenomenon-discovery work. The official model repository remains read-only.

## Evidence stages

1. `verify_cornerstone.py` independently validates the full v2.2 TEST artifact.
2. `analyze_existing_results.py` derives rescue/harm, free-running divergence,
   coalition, expert-profile, and abstention-composition evidence without new
   model inference; `analyze_coalition_strata.py` completes difficulty/domain
   and other full-TEST coalition cuts; `analyze_negative_controls.py` audits the pre-existing
   bounded duplicate/weak/scale/fixed-pool DEV smoke packet.
3. `capture_teacher_forced_dev.py` creates the resumable top-k=64 DEV cache;
   `analyze_teacher_forced_cache.py` certifies cached soft/logit winners and
   evaluates all 127 non-empty expert committees;
   `analyze_robust_soft_controls.py` adds bounded validation-weighted hard,
   inverse-entropy, trimmed/median probability, and confidence-clipping
   controls without new model forwards or promotion;
   `analyze_confidence_conditioning.py` reports each expert's confidence,
   entropy, and margin stratified by token correctness;
   `analyze_committee_diagnostics.py` adds answer/abstain accuracy, vote
   entropy, primary-tie rates, and scaling intervals to all 1,143 cached
   committee-method records without imputing latency or VRAM;
   `analyze_pairwise_diversity_extended.py` completes all 21 pairwise JS,
   correctness-MI, and success/failure-Jaccard diagnostics from frozen data;
   `analyze_hard_temperature_ties.py` measures the confidence-tie boundary of
   the otherwise exact proposal/unique-plurality invariance claim;
   `select_teacher_forced_cases.py` deterministically samples nine local
   failure/rescue classes before narrative interpretation.
4. `freeze_selected_free_run.py` freezes the hard/raw/calibrated/rank DEV
   comparison. `analyze_selected_free_run.py` refuses incomplete or
   config-mismatched output and produces paired trajectory evidence only after
   all 16,200 rows exist.
5. `fit_adaptive_teacher_forced.py` selects a K3→K7 rule from cached DEV only;
   `freeze_adaptive_dev.py` freezes the corresponding reference-decoder policy.
   `analyze_tie_eos_counterfactuals.py` performs exact cached local ablations
   and prevents failed EOS variants from being promoted.
6. `freeze_selected_committee_dev.py` freezes the teacher-forced best K=1…5
   hard committees plus all seven K=6 leave-one-out committees;
   `analyze_selected_committee_dev.py` combines them with the existing full K=7
   hard DEV result for trajectory, answer-class, latency, VRAM, and paired
   leave-one-out analysis.
7. `freeze_v22_calibrated_test.py` freezes one DEV-selected calibrated K=7
   v2.2 TEST condition before output. `analyze_v22_calibrated_test.py` compares
   it with the already-known hard/raw anchors in a separate two-test Holm family
   and preserves the prior n=32/known-anchor caveats.
8. `verify_cross_benchmark.py` defines the v2.1.2 claim boundary;
   `freeze_cross_benchmark.py` freezes six cross-corpus conditions and
   `analyze_cross_v212.py` enforces completeness and the five-comparison Holm
   family.
9. `make_final_synthesis_facts.py` consolidates the five strict long-run
   analyses into a transcription-safe fact packet; `make_paper_figures.py`,
   `make_paper_tables.py`, and `update_live_report.py` generate the remaining
   hash-tracked paper and progress artifacts.
10. `run_remaining_pipeline.py` resumes the frozen post-selection sequence after
   the 16,200-row selected DEV run. After the written synthesis and claim ledger
   are updated, `verify_final_synthesis.py` binds them to the completed analysis
   and generated-artifact hashes; `audit_phase2_completion.py` independently
   enforces that gate plus final counts, bibliography, and official-repository
   immutability.
11. `verify_rank_topk_parity.py` binds the memory-safe stable partial top-k to
   the original stable full-sort prefix with exhaustive, randomized, and
   realistic-vocabulary exact checks. This evidence is required because the
   reciprocal-rank DEV block resumed across that implementation boundary after
   CUDA allocator failures.
12. New model-facing experiments must fit temperatures, thresholds, and
   comparator choices on DEV. Any later TEST statistic requires a frozen
   procedure and must disclose the earlier n=32 TEST exposure.

## Split labels

- TRAIN: `TRAIN MECHANISTIC / EXPLORATORY`
- DEV: `DEV MODEL-SELECTION / CALIBRATION`
- TEST: `TEST CONFIRMATORY`, with the prior n=32 exposure caveat

## Reproduction

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
& $Python experiments\phase2_mechanism\verify_rank_topk_parity.py
& $Python scripts\run_dev.py --config configs\phase2_selected_dev_free_run.yaml --dataset v22_dev --device auto --precision auto --summary-every 25
& $Python experiments\phase2_mechanism\analyze_selected_free_run.py
& $Python experiments\phase2_mechanism\fit_adaptive_teacher_forced.py
& $Python experiments\phase2_mechanism\freeze_adaptive_dev.py
& $Python experiments\phase2_mechanism\analyze_tie_eos_counterfactuals.py
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
& $Python experiments\phase2_mechanism\update_live_report.py
& $Python -m pytest -q
& $Python experiments\phase2_mechanism\verify_final_synthesis.py
& $Python experiments\phase2_mechanism\audit_phase2_completion.py
```

Long free-running commands are append-durable. Re-run the exact command to
resume; config hashes and method-example keys are validated before appending.
The supervisor persists row/owner snapshots and a bounded restart history in
`state/phase2_remaining_pipeline.json`.
After the first selected DEV command completes, the adaptive/cross/analyze/
regenerate/test suffix above can instead be run with
`run_remaining_pipeline.py`; final synthesis and `verify_final_synthesis.py`
still precede the completion audit.
