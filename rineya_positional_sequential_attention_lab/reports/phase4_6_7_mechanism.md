# Phases 4, 6, 7 — Significance matrices, probes, and low-rank structure

**Evidence level: complete for the calibration split.** Phase 4 ran all four arms
(149 items at 2K, `exact_retrieval` + `multi_hop_guided`, objective `gold_nll`,
ε = 0.25, 8 absolute bins). Phase 6 ran two arms at 4K on 594 items / 85 scenarios.
Phase 7 is CPU-only over the resulting artifacts.

Artifacts: `results/phase4_significance/<arm>/kv_head_position_significance.{npz,csv}`
and `tier2_ablation_significance.{npz,csv}`; `results/activation_probe_results.json`,
`activation_patch_results.parquet`, `head_ablation_results.parquet`;
`results/significance_svd.json`, `significance_low_rank_models.npz`,
`phase7_weight_delta_correlation.csv`. All with provenance sidecars.

---

## 1. The cheap estimator is validated (SPEC §11.2's gate)

Brute-force causal significance is 458 k forwards. The lab uses attribution — one
forward + one backward yields every `(layer, KV-head, bin)` cell — and then confirms it
against real ε = 0.25 ablation on the top-64 cells **plus an equal-size random control
sample** from the remainder. The control is what stops the agreement statistic from
being computed only where attribution already said "big".

| arm | Spearman(Tier 1, Tier 2) | 95 % CI | verdict |
| --- | ---: | --- | --- |
| qwen3 | +0.901 | [0.760, 0.967] | attribution_usable |
| halo | +0.945 | [0.903, 0.971] | attribution_usable |
| nimbus | +0.965 | [0.926, 0.983] | attribution_usable |
| supa_a11 (ungated) | +0.963 | [0.897, 0.989] | attribution_usable |

128 cells each, CI clustered on layer (cells within a layer share that layer's `v_proj`
gradients and are not independent). All four clear `accept_rho = 0.5` by a wide margin.
**The 13× speedup is earned, not assumed.**

## 2. Qwen3's edge attention is actively counterproductive

Mean causal importance by source-position bin. Sign convention: **positive = removing it
hurts** (the cell matters); **negative = removing it helps**.

| arm | bin 0 (start) | bins 1–6 | bin 7 (end) |
| --- | ---: | ---: | ---: |
| qwen3 | **−0.0074** | ≈ 0 | **−0.0102** |
| halo | +0.0042 | ≈ 0 | −0.0114 |
| nimbus | +0.0031 | ≈ 0 | −0.0128 |
| supa_a11 | **+0.0068** | ≈ 0 | **+0.0004** |

For Qwen3, attenuating attention to **either edge** — and especially the end — *lowers*
gold NLL. Its edge attention is not merely unhelpful, it is a net cost on these tasks.
Halo and Nimbus keep a useful start but retain the harmful end. Supa-A1.1 is the only arm
where start-attention genuinely matters and end-attention is roughly neutral.

This is the third independent line pointing the same way: Phase 3 found Qwen3's steep
recency gradient *behaviourally*, Phase 4b found Supa's lowest edge bias *structurally on
contentless contexts*, and Phase 4 now finds the same ordering *causally*. It also names
a concrete intervention for Phase 8 — suppress the terminal bin — rather than leaving the
matrix as description.

Individual top cells are much larger than these means (0.30–0.53 nats), and they
concentrate at the extremes: for Qwen3 `(14, kv6, bin7)`, `(18, kv7, bin7)`,
`(1, kv2, bin7)`; for Halo and Nimbus `(0, kv6, bin0)`.

**Causally important layers differ by model:** Qwen3 {14, 4, 1, 15}, Halo {15, 4, 5, 19},
Nimbus {15, 5, 4, 19}, Supa {0, 2, 19, 4}. Halo and Nimbus agree closely, as they must —
Phase 1 showed Nimbus is Halo rescaled by 0.8. Supa's mass sits markedly earlier.

## 3. Brief question 8: the matrices are strongly low-rank

Cumulative explained variance of the `[layer × KV-head, bin]` unfolding (centred; the
reconstruction error is normalised on the same centred matrix, so
`relerr(k) = sqrt(1 − cumulative(k))` holds exactly and is asserted in the script):

| matrix | rank 1 | rank 2 | rank 4 |
| --- | ---: | ---: | ---: |
| causal significance `I` | 0.614 | **0.902** | 0.967 |
| positional bias `B` (all four models) | **0.87–0.88** | **0.994** | 0.999 |

**Rank 2 captures 90 % of causal significance and 99 % of positional bias.** The brief
(§21) asks for this to be reported prominently if it holds, and it does: positional
behaviour in this family is controllable with a very small number of parameters, which is
the premise the compact `S[h,p] = a_h·f(p) + c_h` parameterisation rests on. `B` being
even lower-rank than `I` is expected — it is a smoother object, measured without content.

## 4. Brief question 10: weight deltas predict *bias*, not *significance* — with opposite signs

Spearman between per-head relative weight change and the change in edge bias, clustered
on layer, restricted to the 19 souped layers (`*` = CI excludes zero):

| model | scope | q_proj | **k_proj** | v_proj | o_proj | mlp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| halo | soup layers | +0.109 | **+0.254\*** | +0.198\* | +0.229\* | +0.237\* |
| nimbus | soup layers | +0.112 | **+0.269\*** | +0.174 | +0.215 | +0.244\* |
| supa_a11 | soup layers | +0.108 | **−0.361\*** | −0.231\* | −0.326\* | −0.384\* |
| supa_a11 | adapter only | −0.080 | **−0.273\*** | −0.201 | −0.233 | +0.004 |

Two things stand out.

**The sign is opposite between the soups and the adapter.** In Halo and Nimbus, heads
whose weights moved more have *higher* edge bias. In Supa, heads whose weights moved more
have *lower* edge bias. The same measurement, run on the pure adapter delta (Phase 1 §3's
natural experiment) rather than the soup, keeps the negative sign. So the LoRA adapter and
the soup did opposite things to positional bias — which is exactly the dissociation Phase 3
saw behaviourally, now visible in the weights.

**`k_proj` is the strongest attention-family correlate in every single row.** Phase 1
found `k_proj` was the most-changed attention projection in all three cores; this is the
first evidence that the change was *functional* rather than incidental, since K is the
projection that decides where a head looks.

**But the same correlations against the causal significance matrix are all near zero**
(+0.002 to +0.094, none significant). Weight movement predicts where a head *looks*, and
does not predict whether looking there *helps*. That is a clean separation, and it argues
that `B` and `I` are genuinely different objects rather than two views of one.

Caveat: 160 of 600 rows have an undefined correlation because the weight delta is constant
over those heads — the nine frozen layers. Those are excluded, not imputed.

## 5. Phase 6: position is strongly decodable, evidence location much less so

Ridge probes, train/test split **by scenario**, with a 20-permutation baseline:

| target | site | qwen3 best layer | R² | supa best layer | R² |
| --- | --- | :-: | ---: | :-: | ---: |
| normalized position | attention out | 17 | **0.965** | 12 | **0.964** |
| normalized position | residual | 17 | 0.941 | 10 | 0.948 |
| normalized position | MLP out | 10 | 0.910 | 10 | 0.906 |
| evidence location | residual | 17 | 0.594 | 15 | 0.606 |

Absolute position is almost perfectly linearly decodable by the middle of the network.
*Where the evidence is* is decodable at R² ≈ 0.6 — far above chance but far from position
itself. **Supa's positional information peaks earlier** (layers 10–12) than Qwen3's
(layer 17), matching Phase 4's finding that Supa's causally important layers sit earlier.

Two honest caveats. The permutation p-values are all 0.0476 = 1/21, which is the
*resolution floor* of a 20-permutation test — the probes clear chance, but the test cannot
report a smaller p than that. And probes nominate; they do not establish causation. The
causal claims in this report come from §2 and from patching, not from here.

## 6. What these three phases do not establish

- **No intervention has been tested yet.** Everything here is measurement. Phase 8 is the
  first test of whether any of it improves anything, and it runs with 48 matched negative
  controls against 20 treatments precisely because a significance matrix that merely
  *describes* the deficit is not evidence that acting on it helps.
- **2K, two families, `gold_nll`.** Phase 4 ran at 2K because the Tier-1 backward needs
  4.2 GB there and 7.2 GB at 4K. The absolute-binned grid is, per Phase 4b, blind to about
  half the positional structure.
- **Phase 6 covers two arms**, qwen3 and supa_a11 ungated, not the full panel.
