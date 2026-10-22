# Independent reference tables review 002

PASS_REFERENCE_TABLES. Independent standard-library recomputation passed 989/989 checks against the approved report, raw outcomes, trajectory prefixes, metadata and exact hashes.

The five tables retain nine policy rows, six contrasts and96 paired example rows. Intact accuracy is12/16; all recurrent policies13/16. Only the counterfactual case gains versus intact; there are no losses. All adaptive versus unit/fixed/random score deltas are zero. Inference p-values/intervals stay null (blank CSV cells), finite-grid headroom0/capture null.

Each cost row charges32 physical DEV runs from16 unique cases, two seed repeats per case. There are13 provisional scenario IDs; independence is unestablished. Reused oracle rows are excluded. The cost table totals 1552 generation calls and 619216 prefix positions, covers DEV only, and does not replace total worker/mechanical/load costs. Feature moments independently match176 generated-prefix observations, not176 independent cases. Instrumented timing and inherited gold-coupled format scores retain their limitations.

The successor corrects both findings from001: unique_DEV_cases derives from actual IDs, clustering remains explicit, and A2 strings now join exactly across accuracy, contrasts and costs. Four unchanged tables have identical hashes to001; only the cost table/source/manifest changed. No unresolved discrepancy remains.
