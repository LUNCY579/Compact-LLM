# Phase 3 — Baseline positional fingerprint

**Evidence level: complete for the calibration split, teacher-forced.**
1752 items × 5 arms = 8760 records. The analysis holds **both** nuisance factors fixed —
`local_position == "middle"` and `fmt == "natural"` — see §5. Primary metric is summed gold NLL over the answer
(SPEC §13); lower is better. Confidence intervals are BCa bootstraps with 10 000
resamples **clustered on `scenario_id`**, so positional variants of one source example
are never treated as independent.

Artifacts: `results/baseline_position_results_calibration_tf.parquet`,
`results/phase3_positional_analysis.json` (+ provenance sidecars).
Reproduce with `scripts/phase3_baseline_fingerprint.py` then `scripts/phase3_analyze.py`.

Not yet run: the greedy/accuracy pass, the validation split, and the 16K/32K buckets.
The sealed split remains sealed.

---

## 1. The headline: lost-in-the-middle is real, and its *shape depends on the task*

Summed gold NLL by position of the **decisive** evidence sentence. `mid−edge` is
(middle − mean of the two extremes); positive means the middle is worse. `*` marks a
95 % CI excluding zero.

### Exact retrieval (n = 33/cell)

| arm | 0.02 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 0.98 | mid−edge | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| qwen3 | 3.52 | 7.37 | 7.06 | 6.83 | 6.06 | 4.15 | 0.80 | **+4.14*** | 0.17 |
| halo | 0.23 | 1.05 | 2.10 | 1.44 | 2.50 | 1.04 | 0.01 | +1.43* | 0.67 |
| nimbus | 0.37 | 1.51 | 2.11 | 2.05 | 2.57 | 1.46 | 0.03 | +1.95* | 0.67 |
| supa_a11 | 0.03 | 2.31 | 0.19 | 0.08 | 0.02 | 0.05 | 0.01 | **+0.10*** | 0.17 |

### Multi-hop, hard variant (n = 39/cell, 3 positions by construction)

| arm | 0.02 | 0.50 | 0.98 | mid−edge | worst |
| --- | ---: | ---: | ---: | ---: | :-: |
| qwen3 | 23.55 | 26.31 | 24.90 | +2.09* | 0.50 |
| halo | 18.83 | 24.86 | 20.13 | **+5.38*** | 0.50 |
| nimbus | 19.21 | 25.34 | 21.06 | +5.21* | 0.50 |
| supa_a11 (ungated) | 5.20 | 12.10 | 10.39 | +4.31* | 0.50 |

### State/constraint (n = 45/cell)

| arm | 0.02 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 0.98 | mid−edge | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| qwen3 | 10.52 | 10.87 | 11.12 | 11.31 | 11.33 | 11.19 | 10.59 | +0.75* | 0.67 |
| halo | 8.75 | 9.09 | 9.17 | 9.53 | 9.24 | 9.49 | 9.16 | +0.57* | 0.50 |
| nimbus | 8.78 | 9.23 | 9.36 | 9.78 | 9.55 | 9.78 | 9.40 | +0.69* | 0.83 |
| supa_a11 (ungated) | 4.28 | 4.25 | 4.05 | 4.13 | 4.09 | 4.13 | 4.03 | −0.03 | 0.02 |

### Semantic single-hop (n = 66/cell)

| arm | 0.02 | 0.17 | 0.33 | 0.50 | 0.67 | 0.83 | 0.98 | mid−edge | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-: |
| qwen3 | 10.66 | 12.62 | 9.96 | 9.83 | 9.88 | 9.80 | 8.05 | +0.58 | 0.17 |
| halo | 13.13 | 16.99 | 13.64 | 11.46 | 10.64 | 10.94 | 9.16 | +0.49 | 0.17 |
| nimbus | 12.53 | 16.27 | 12.96 | 11.28 | 10.59 | 10.94 | 9.03 | +0.66 | 0.17 |
| supa_a11 (ungated) | 6.09 | 7.03 | 5.13 | 3.87 | 4.08 | 4.71 | 5.29 | **−1.82*** | 0.17 |

**The task family determines the shape, and it is not the same shape.**

- `multi_hop` and `state_constraint` show the textbook pattern: worst at or adjacent
  to p = 0.50, significant for every non-Supa arm. Multi-hop is where the effect is
  largest by a wide margin (up to +5.38 nats).
- `exact_retrieval` does **not** peak in the middle. Qwen3's worst position is 0.17 and
  its best is 0.98, a 6.57-nat range that is dominated by a recency gradient rather
  than a symmetric U. A quadratic fit reports β₂ > 0 *and* a large negative β₁ here;
  quoting only β₂ would hide the larger term.
- `semantic_single_hop` peaks at 0.17 for every arm, and the middle-minus-edge statistic
  is **not significant** for qwen3, halo or nimbus — because the deficit is not in the
  middle. The `range` column tells a different story from the `mid−edge` column (7.8
  nats vs +0.32 for halo), and reporting only the gap statistic would call this family
  "position-invariant" when it varies by 8 nats.

**Consequence for the rest of the study:** "the edge-middle gap" is not one number per
model. Any intervention tuned on `exact_retrieval` is being tuned on a recency
gradient; any intervention tuned on `multi_hop` is being tuned on a genuine middle
trough. Phase 8 onward must state which it is targeting.

## 2. Brief questions 1–3, answered

**1. Does each model exhibit lost-in-the-middle?** Yes for Qwen3, Halo and Nimbus, on
`multi_hop` and `state_constraint`, with CIs excluding zero. For `exact_retrieval` all
four models show a significant positional deficit but its location is not the middle.
For `semantic_single_hop` the three non-Supa models show no significant *middle* deficit
despite a large positional range.

**2. Which model has the best worst-position robustness?** Supa-A1.1, decisively.

**3. Which has the smallest edge-middle gap?** Supa-A1.1, on every family:
+0.10 vs Qwen3's +4.14 on `exact_retrieval` (a 97.5 % reduction), −0.03 (n.s.) vs
+0.75 on `state_constraint`, and −1.82 on `semantic_single_hop` — where its middle is
*better* than its edges, the only reversal in the panel. The exception is `multi_hop`,
where Supa retains a +4.31 middle deficit despite being 15 nats better overall.

## 3. Paired against Qwen3 on identical items

Mean NLL difference vs Qwen3, clustered bootstrap, negative = better than Qwen3:

| arm | exact_retr. | semantic | state | multi-hop hard | multi-hop guided |
| --- | ---: | ---: | ---: | ---: | ---: |
| halo | **−3.77** | **+2.12** | −1.79 | −3.66 | −5.17 |
| nimbus | **−3.53** | **+1.79** | −1.58 | −2.97 | −4.20 |
| supa_a11 (ungated) | −4.53 | −4.89 | −6.85 | −15.05 | −11.25 |

Every cell's CI excludes zero. Cohen's *dz* ranges from 0.84 to 5.49.

**The dissociation is confirmed at full n.** Halo and Nimbus are substantially better
than Qwen3 on three families and reliably **worse** on `semantic_single_hop`
(+2.12 and +1.79 nats, dz ≈ 1.1). Supa-A1.1 is better on all five. Whatever the
Halo/Nimbus branch bought in verbatim and structured retrieval, it cost something in
semantic single-hop retrieval — and since Phase 1 showed Nimbus is just Halo at
α ≈ 0.80, this is one trade-off, seen twice, not two independent findings.

This matters for Phase 9's model selection, which must be stated over all families
before the numbers are consulted rather than after.

## 4. The abstention gate is a real confound for log-prob work

Supa-A1.1 gated vs ungated, same items. On `exact_retrieval` and `semantic_single_hop`
the two are identical to three decimals. On the harder families they diverge:
`state_constraint` 4.60–5.09 gated vs 4.03–4.28 ungated, `multi_hop_guided` +2.33 vs
+2.72 mid−edge, and the gated arm's worst position moves from 0.50 to 0.98.

That is the expected direction. The gate adds +5.156 to one vocabulary row, which
inflates the softmax normaliser and therefore every other token's NLL, and it does so
most where probability mass is most spread — i.e. where the model is least certain.
The gate is measuring uncertainty, not position. **All mechanistic phases use the
ungated arm**, as SPEC §1 requires; the gated arm is retained for accuracy and
preservation, where it is the published model's real behaviour.

## 5. Two analysis errors found and corrected before publication

Both produced complete, plausible tables. Neither raised.

**The position axis was wrong for multi-evidence families.** `state_constraint` and
`multi_hop` carry three or four evidence sentences; the non-decisive ones are pinned by
construction and only the decisive one is swept. Grouping on `target_positions[0]`
collapsed `state_constraint` onto the single constant 0.25 — the family contributed
*one* position to a positional study — and read `multi_hop` off its wrong endpoint. The
axis is now `meta.primary_target`, the decisive sentence. Everything in §1 for those two
families changed as a result; `exact_retrieval` and `semantic_single_hop` have one
evidence sentence and were unaffected. This is the same error class as the Phase 2b §7
correction, which is why it was looked for.

**A silent sign inversion.** `paired_bootstrap(a, b)` returns `b − a`, as its docstring
states. It was called as `(other, base)` while the output field was labelled
`mean_diff_vs_qwen3`, so every difference in §3 and every `better_than_base` verdict was
inverted — Halo was reported as *worse* than Qwen3 on exact retrieval when it is 3.77
nats better. It was caught by cross-checking §3 against the §1 curves, which disagreed.
The call now passes `(base, other)` with a comment naming the convention.

### A third instance of the same confound, caught after publication

The first version of this report filtered `local_position` and **not** `fmt`. The
`format_sweep` grid emits only p ∈ {0.02, 0.50, 0.98}, so each of those three cells
carried 18 extra non-natural items (6 json + 6 list + 6 shuffled) on top of 33 natural,
while the other four positions were natural-only — surface format loaded onto exactly the
three positions the edge-middle gap is computed from. Same error class as the
`local_position` confound this report already documents, and I looked for that one and
then failed to check its sibling.

Re-run with both factors fixed: **every qualitative conclusion survives.** No cell changes
significance, no worst position moves, and the ordering of models is unchanged. The
magnitudes move by up to ~13 %: Qwen3's `exact_retrieval` gap +4.67 → **+4.14**, Supa's
+0.07 → **+0.10**, Halo's +1.33 → +1.43, and the `semantic_single_hop` gaps rise slightly
(all still non-significant for the three non-Supa arms). The tables above are the
corrected numbers. `multi_hop` and `state_constraint` are unaffected, because neither is
emitted by the `format_sweep` grid.

## 6. What this run does not establish

- **Teacher-forced only.** No accuracy, no generation. Phase 2b showed accuracy is at
  floor for most of this panel, so NLL is the informative metric, but the greedy pass is
  still owed.
- **Calibration split only**, at 2K/4K/8K. Validation is untouched; sealed is sealed.
- **Halo runs in bfloat16**, not its stored float32 — a recorded approximation.
- **Multi-hop cells are unbalanced** (39–117 items) because the six position-pair
  combinations map unevenly onto three anchors. The bootstrap clusters on scenario, so
  this affects precision, not validity.
- **No mechanism.** This report says *where* each model degrades, not *why*. Layers,
  heads and the causal significance matrix are Phases 4–6.
