# Frozen paired factorial analysis plan

Experimental unit: the exact CHECK world, not a role, token, generation or checkpoint load. The old and new tables must contain precisely the same complete set of IDs and matching family/scenario/split metadata. A partial intersection is forbidden. Source, role, selector, parser, bank and protocol hashes bind the comparison; incompatible policy or runtime requires a new experiment.

Let A/B be old ordinary/multi correctness and C/D new ordinary/multi correctness. For each world compute:

| Label | Boolean definition |
|---|---|
| Training only repair | not A and C |
| Multi only repair | not A and B |
| Both available | not A and B and C |
| Posttrain multi extra | not C and D |
| Multi lost after training | B and not D |
| Harm from training | A and not C |
| Harm from multi pretrain | A and not B |
| Harm from multi posttrain | C and not D |

These names preserve the requested definitions: “training only” and “multi only” are NOT mutually exclusive categories. “Both available” is their overlap. Report counts without forcing them to sum to N.

Report cell means and paired mean differences C-A, B-A, D-C and interaction (D-C)-(B-A), overall and per family. The last quantity is the mean of the per-item interaction contrasts, not subtraction across unmatched populations. All 16 Boolean A/B/C/D patterns are covered by software fixtures now.

Also report difficulty and family-by-difficulty strata, with the recorded generator difficulty unchanged. Enumerate all 16 ABCD Boolean patterns and the explicit newly-useful, unnecessary and harmful scaffold masks defined in the analysis output. Preserve old/new role-complementarity, candidate ceiling, selector utilization, repair/damage, token/call/time and memory summaries to distinguish increased candidate quality from increased selection quality. These diagnostics do not replace the four paired effects.

Use 2,000 paired world-cluster bootstrap draws at seed 1978 for descriptive percentile 95% intervals for each difference and interaction. Resample the same world across all four cells and all derivatives. No power calculation, interaction p-value, factorial significance threshold, or equivalence claim is prespecified or later inferred from the scout. Current A/B McNemar descriptions remain exploratory. Model-training seed uncertainty is not estimated from one old and one new checkpoint.

Interpret the signs and error-set overlaps descriptively: additive when both interventions contribute with interaction near zero; redundant when gains overlap and new multi gain shrinks; synergistic when interaction is positive; antagonistic when it is negative. “Near zero” is descriptive, with the interval shown, not a registered equivalence test. A dataset/training difference is part of checkpoint lineage; this observational before/after contrast cannot isolate dataset content from every other training change.

When C/D arrive, do not design new post hoc event masks, switch the primary comparator, optimize a selector, or hide damage counts. Optional analyses must be labelled new exploratory work and preserve this frozen primary analysis.
