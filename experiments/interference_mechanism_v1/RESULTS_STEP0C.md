# INTERFERENCE-MECH-001, step 0c: the cost is a property of the rows and the coefficient (exploratory)

**Zero forwards.** The data is recorded generations on T_BANK_V3X (864 rows, exposed after
CONFIRM-001):
- CONFIRM-001's SPARSE, AVG and ADD-100, with the reconstructed FP32 dOpallo;
- the audit lane's dev-v4 composites at (a, b) = (1, 1) and (1, 0.75), with the stored BF16 dOpallo.

The two Opallo deltas are far apart as objects: ‖difference‖ = 50.7% of ‖dO‖, cosine 0.876, and
BF16 storage erased 45.8% of the update elements. Script: `endpoint_nesting.py`. Output:
`ENDPOINT_NESTING.json`.

| question | result |
|---|---|
| Do the two full-strength composites lose the **same** Titan rows? | Yes: 56 and 53 lost, **52 shared** (Jaccard 0.91). **786/864 generations are identical.** 20 gained rows are shared. |
| Is the damage **nested in β**? | **Perfectly.** All 40 rows lost at b = 0.75 are also lost at b = 1.00. Going to 1.00 adds 13 more, and 0 rows are lost only at 0.75. |
| Overlap with the rows averaging loses | 39 of 56 (reconstructed) and 38 of 53 (stored) |

## Reading

The Titan-skill cost of adding Opallo is **almost entirely independent of how the Opallo delta is
represented**. Half of its norm and nearly half of its elements can change without changing which
rows break. The cost grows **monotonically** with the coefficient on a fixed set of fragile rows,
and those are largely the rows averaging also breaks. This fits step 0's knife-edge result: the
rows that break are low-margin rows, and a larger push flips a superset of them. It does not fit a
coordinate-level collision account: changing which elements are nonzero would then change which
rows break.

## Limits

One bank (exposed), one donor pair, two coefficients. The nesting could fail at other coefficients.
The queued batteries (ADD-O50/O60/O90 and the DARE arms) will add points.
