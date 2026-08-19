# Statistical Analysis

## Unit of analysis

The end-to-end unit is the original source problem, not an expanded transition row. FULL and SPLIT outcomes are paired by `problem_id`.

## Primary effect

```text
Delta = mean(candidate_correct - baseline_correct)
```

The comparison report includes a paired percentile bootstrap confidence interval. Pairing preserves problem difficulty and normally yields a tighter, more relevant estimate than an unpaired comparison.

## McNemar test

For binary correctness, exact McNemar testing uses only discordant pairs:

- baseline correct / candidate wrong;
- candidate correct / baseline wrong.

The report retains both counts rather than presenting only a p-value.

## Sign-flip randomization

A paired sign-flip test provides a distribution-free sensitivity check for the mean paired effect.

## Decision rules

Given non-inferiority margin `delta`:

- `SUPERIOR`: lower bound of 95% CI > 0;
- `EQUIVALENT`: 90% CI lies entirely within `[-delta, +delta]`;
- `NONINFERIOR`: lower bound of 95% CI > `-delta`;
- `INFERIOR`: upper bound of 95% CI < `-delta`;
- otherwise `INCONCLUSIVE`.

The margin must be justified and frozen before TEST. A p-value alone does not establish practical equivalence.

## Multiplicity

Category and ablation results are secondary/exploratory unless a correction plan is preregistered. Treat isolated subgroup wins cautiously, especially with small `n`.

## Multiple seeds

For repeated training seeds, report:

- per-seed paired effects;
- mean and dispersion across seeds;
- problem-level aggregate under a declared rule;
- whether conclusions are stable to seed exclusion.
