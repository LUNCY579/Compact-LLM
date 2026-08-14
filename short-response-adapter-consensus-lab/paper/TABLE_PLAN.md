# Table Plan

## Generated artifact status

`experiments/phase2_mechanism/make_paper_tables.py` now generates the populated
paper-facing tables in `paper/TABLES.md`, machine-readable CSVs in
`paper/tables/`, and `paper/tables/table_metadata.json` with exact generator,
source, and output hashes.

| Planned table | Status |
|---|---|
| 1–9 | GENERATED from frozen TEST or DEV analysis artifacts |
| 10 | GENERATED from the complete same-split free-running DEV run |
| 11 | GENERATED from DEV teacher-forced selection and frozen free-running adaptive measurement |
| 12 | GENERATED from fixed-method v2.1.2 transfer; provenance caveats retained |
| 4b | GENERATED from the frozen calibrated v2.2 TEST confirmation |

The generated package also includes all 127 committee rows and the pairwise
expert-diversity appendix. A second all-127 diagnostics appendix adds
answer/abstain all-token accuracy, vote entropy, tie rate, and forward proxy
for every aggregator; counterfactual latency/VRAM remain null. Unmeasured fields
are labeled not captured; they are not retrospectively imputed.

## Table 1 — Dataset and exposure provenance

Columns: dataset, version, absolute/archival path, split, N, SHA256, source,
training exposure, selection exposure, evaluation role, task categories, prior
access. Separate v2.2 and v2.1.2; never aggregate their accuracies post hoc.

## Table 2 — Model and adapter inventory

Columns: backbone/model, role, parameters, adapter type, rank, alpha, target
modules, tokenizer hash, declared base, adapter config hash, weights hash,
training lineage. D_greedy is labeled unmatched lineage.

## Table 3 — Primary full-TEST benchmark

Columns:

| Method | N | Exact acc. (95% CI) | Answerable | Abstention | False abstention | Mean tokens | Latency | Expert forwards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Source E-CORNERSTONE-RAW. Bold only the best exact accuracy within the frozen
comparison. Cost leaders are labeled separately.

## Table 4 — Paired rescue/harm

Columns: candidate, baseline, both correct, rescue, harm, both wrong, net gain,
paired CI, exact McNemar p, Holm p. Global rows are confirmatory; subgroup rows
appear in appendix/descriptive tables.

## Table 4b — Selected calibrated probability on v2.2 TEST

Rows: calibrated K=7 versus the already-known hard K=7 and raw-probability K=7
anchors. Columns: both accuracies, paired difference and bootstrap interval,
calibrated-only/baseline-only correctness, exact McNemar p, and the separate
two-comparison Holm p. The prior n=32 exposure and known-anchor caveats are
printed beside the table; this family is not merged with Table 4.

## Table 5 — Expert calibration

Columns: expert, fitted DEV (T_i), top-1 token accuracy, mean confidence,
confidence minus accuracy, ECE-15, NLL, Brier, entropy, margin. Include token and
example counts and label DEV selection.

## Table 6 — Aggregator comparison under teacher forcing

Columns: hard plurality, raw probability, calibrated probability, shared-T
probability, raw logit, normalized logit, top-2 approval, Borda, reciprocal
rank, validation-weighted hard, inverse-entropy weighting, trimmed and median
probability pooling, and confidence-clipped means; report token, first-token,
answer-token, EOS, and all-gold-token example accuracy plus top-k certificate
rate. Mark all added robust rows as post-DEV exploratory and not promoted;
median's two uncertified token positions must remain explicit.

## Table 7 — Committee-size scaling

For each K and selected method: number of committees, mean/min/max performance,
best performance, descriptive 95% intervals, best members, forward proxy. Mean
intervals treat examples as the resampling unit after averaging the exhaustive
committees at K; best-at-K Wilson intervals are conditional on same-DEV
selection and are not selection-adjusted. The distinction between average and
selected best is mandatory.

After E-COMMITTEE-FREERUN completes, append the teacher-forced-selected hard
best-at-K free-running exact accuracy, answer/abstain cuts, latency, forwards,
and accuracy-per-forward. Keep local all-token and free-running exact accuracy
visually distinct because they are different estimands.

## Table 8 — Leave-one-out and exact Shapley

Columns: expert, standalone performance, K7−K6 performance delta, answerable
delta, abstention delta, token delta, Shapley value, selected pair interactions.
Category-specific rows move to appendix.

Append the seven-comparison free-running K7-versus-K6 family after the frozen
panel completes, with paired bootstrap intervals, exact McNemar tests, Holm
correction, measured latency/VRAM, and descriptive category/answer-class rows.

## Table 9 — Temperature interventions

Rows: perturbed expert × temperature. Columns: raw/calibrated probability local
accuracy, NLL, selected-token change rate, top-k certificate. Hard voting is a
raw reference plus the separate tie-boundary grid: proposals and unique
pluralities are invariant, while the confidence tie-break can change under
unequal expert temperatures.

## Table 10 — Teacher-forced versus free-running

Same DEV examples only. Columns: method, local token accuracy, local all-token
example accuracy, free-running exact accuracy, gap relative to hard, first-
divergence statistics. This table adjudicates trajectory correction.

## Table 11 — Adaptive consultation and efficiency

Columns: policy/threshold, calibration fold, escalation rate, false non-
escalation, exact accuracy retention, mean forwards, latency, TTFT, VRAM, RSS.
Report full K=7 and fixed K=3 anchors.

## Table 12 — Cross-benchmark confirmation

One block per benchmark. Columns match the primary table where defined, but no
combined accuracy. Include provenance and exposure status in each block.

## Appendix tables

- all 127 committee rows;
- full task/category/domain/difficulty rescue/harm strata;
- pairwise top-1 agreement, double-fault, Q-statistic, and error overlap;
- complete coalition inventory;
- reliability-bin values;
- template-family and near-duplicate sensitivity;
- hyperparameters, software, hardware, tie/EOS pseudocode;
- all qualitative selection IDs and hashes;
- timing distributions and warmup policy.
