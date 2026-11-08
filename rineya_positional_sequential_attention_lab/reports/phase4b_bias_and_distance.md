# Phase 4b — Positional bias `B` and distance significance `D`

**Evidence level: complete for what it measures.** Four models × 8 control contexts ×
2 context kinds at T = 8192, capture-based, inference only. 17 minutes of GPU.

Artifacts: `results/kv_head_bias_matrix.npz`, `results/distance_significance.npz`,
`results/phase4b_bias_distance_summary.json` (+ provenance sidecars).
Reproduce: `scripts/phase4b_bias_and_distance.py --split calibration`.

This phase measures what a head does **in the absence of content** — where it looks when
there is nothing worth looking at. That separates *content-independent positional bias*
`B` from the *causal significance* `I` that Phase 4 measures, which is the distinction
brief §13 asks for. A head that always stares at BOS is not necessarily a broken head;
it may be an attention sink doing useful anchoring, and suppressing it would be a
mistake. `B` is how we tell those apart.

## 1. The measurement is robust to the choice of control context

Two independent control kinds were used: **random token ids** drawn from the real
unigram distribution, and **shuffled filler** built from the benchmark's own prose. If
the two disagreed, `B` would be measuring the control rather than the model.

| agreement between the two context kinds | value |
| --- | ---: |
| attention-by-bin, Pearson | **0.9984** |
| attention-by-bin, Spearman | **0.9981** |
| qk-logit-by-bin, Spearman | **0.9994** |
| mean absolute difference in attention mass | 0.0029 |

They agree almost exactly. `B` is a property of the model, not of the probe.

## 2. Brief question 7, answered with a number

*Is head significance primarily absolute-position dependent or distance dependent?*

**Both, in near-equal measure, and they are largely complementary.** Variance explained
by each model of the attention structure (log transform, sum-to-zero identifiability
constraint):

| model | absolute-only | distance-only | **additive** | residual |
| --- | ---: | ---: | ---: | ---: |
| qwen3 | 0.4745 | 0.5009 | **0.9223** | 0.0777 |
| halo | 0.4726 | 0.5020 | **0.9222** | 0.0778 |
| nimbus | 0.4730 | 0.5018 | **0.9222** | 0.0778 |
| supa_a11 | 0.4658 | 0.5047 | **0.9188** | 0.0812 |

Neither factor alone explains half the structure. Together they explain **92 %**, leaving
an 8 % interaction residual. Distance is very slightly the stronger of the two in every
model, but the gap (≈0.03) is small next to what each contributes uniquely.

The practical consequence is a warning about the intervention design: a Significance
Matrix indexed on **absolute position only** — which is the primary grid in SPEC §5.3,
chosen because a distance-binned Mode-1 bias is a query-dependent `[1,H,T,T]` mask and
unaffordable above 2K — is by construction blind to roughly half the positional
structure. That is a real limitation of the cheap intervention, not a defect in the
measurement, and Phase 8 must report absolute-binned results knowing it.

## 3. The four models have nearly identical absolute-vs-distance balance

The numbers above agree across the panel to within 0.9 %. Whatever the Rineya merges
changed, **they did not change how attention divides between absolute position and
relative distance.** Given Phase 1 showed the deltas are concentrated in `k_proj` — the
projection that decides where a head looks — this is a genuine and slightly surprising
null: the key pathway moved, and this particular decomposition did not.

## 4. Supa-A1.1 attends more to the middle even on contexts with no content

Pooled over layers and heads, on control contexts:

| model | edge bias score | **middle mass** | KV-head dispersion |
| --- | ---: | ---: | ---: |
| qwen3 | 0.0788 | 0.1850 | 0.0222 |
| halo | 0.0794 | 0.1823 | 0.0222 |
| nimbus | 0.0793 | 0.1826 | 0.0221 |
| **supa_a11** | **0.0751** | **0.1998** | 0.0224 |

Supa-A1.1 has the **lowest edge bias** and the **highest middle mass** in the panel — it
puts about 8 % more of its attention mass in the middle of a meaningless context than
Qwen3 does, and correspondingly less at the edges.

This converges with Phase 3 from a completely different direction. Phase 3 measured,
*behaviourally on real tasks*, that Supa has the smallest edge-middle gap of the four
(+0.07 nats vs Qwen3's +4.67 on exact retrieval, and a *reversed* gap on semantic
single-hop). Phase 4b measures, *structurally on contexts with no task in them at all*,
that Supa is the model whose attention is least edge-biased. Two independent
measurements, one behavioural and one architectural, pointing the same way.

Halo and Nimbus are indistinguishable from Qwen3 on both statistics (edge bias 0.0794 /
0.0793 vs 0.0788; middle mass 0.1823 / 0.1826 vs 0.1850) — and marginally *worse* on
middle mass, which is the direction Phase 3 found them worse on semantic single-hop.

**This is correlational.** Two measurements agreeing is not a mechanism, and neither
establishes that middle mass *causes* the smaller gap. Phase 4's causal significance
matrix and Phase 6's patching are what would turn this into a mechanism. What it does
do is make Supa the obvious candidate for the "second model" of the brief's §29 funnel,
on evidence rather than on preselection.

## 5. Limitations, stated

- **The KV-head rollup is an approximation.** Attention probabilities are per *query*
  head; Modes 1 and 2 act at *KV*-head granularity. `B[l, kv, b]` is the mean over the
  two query heads sharing that KV head. Both the per-query-head and rolled-up forms are
  saved, and the summary carries `kv_rollup_is_an_approximation: true`.
- **8 control contexts per kind.** Enough for a stable pooled statistic (the two kinds
  agree at ρ = 0.998), not enough for per-head confidence intervals. Per-head claims in
  Phase 8 must come from Phase 4's causal matrix, not from `B`.
- `B` says where a head looks with nothing to look at. It does **not** say whether
  looking there helps. That is exactly the separation this phase exists to create, and
  the "helps" half is Phase 4.
