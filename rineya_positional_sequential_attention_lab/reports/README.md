# Reports

Written in the order the work happened. Each one states its own evidence level in its
opening paragraph; none of them is the final report, which does not exist yet because
the results it would summarise do not exist yet.

| report | phase | evidence level | what it settles |
| --- | --- | --- | --- |
| [`phase1_checkpoint_audit.md`](phase1_checkpoint_audit.md) | 1 | **complete** — exhaustive, CPU-side, reproducible from the checkpoints | the panel's structure, what the merges changed, and the geometry of the family |
| [`phase2_instrument_calibration.md`](phase2_instrument_calibration.md) | 2b | **complete for its purpose** — a go/no-go check on the instrument, n=162 | that the benchmark has dynamic range on NLL and not on accuracy, and three protocol defects that would have wrecked Phase 3 |
| [`phase3_preview_positional_fingerprint.md`](phase3_preview_positional_fingerprint.md) | 3 | **preview — RETRACTED IN PART** (see its §6) and superseded by the row below; retained because it is where the metric decision was made and argued | the choice of primary metric (summed gold NLL) |
| [`phase3_positional_fingerprint.md`](phase3_positional_fingerprint.md) | 3 | **complete** for the calibration split, teacher-forced — 7830 records, clustered BCa CIs | brief questions 1-3: who loses the middle, who is most robust, who has the smallest gap |
| [`phase4_6_7_mechanism.md`](phase4_6_7_mechanism.md) | 4, 6, 7 | **complete** for calibration — 4 arms, attribution/ablation gate passed on all four | the Significance Matrix itself, probes, low-rank structure, and the weight-delta correlation |
| [`phase4b_bias_and_distance.md`](phase4b_bias_and_distance.md) | 4b | **complete** — 4 models x 8 control contexts x 2 kinds at 8K, the two kinds agreeing at rho = 0.998 | brief question 7 (absolute vs distance), and the content-independent bias `B` that separates "looks there" from "helps" |

## How to read the evidence levels

**complete** — the numbers are final for what they measure, and rerunning the
artifact-producing command reproduces them.

**complete for its purpose** — the run was sized to answer a specific yes/no question
and answers it; it is not sized for the effect estimates it happens to contain.

**preview** — the numbers are directional. They exist to make a design decision, and
they carry no confidence intervals, no paired bootstrap, and small per-cell n. Nothing
in a preview may be quoted as a result.

## The seven findings that changed the plan

1. **The Halo/Nimbus branch lies on one ray through Qwen3, exactly** (Phase 1 §4).
   `Δnimbus = 0.8 · Δhalo` holds *exactly* in the merge arithmetic — the apparent 10.3 %
   residual is bfloat16 store rounding to six significant figures — and
   `Δhalo_uncensored = 2.000000 · Δhalo` with cosine 1.00000000, measured directly against
   the donor checkpoint rather than inferred from recipes. **A uniform-α soup of the two is
   therefore 1-D and already contains Nimbus at α = 0.80.** Blockwise α is *not* 1-D: it is
   a 19-parameter search over per-block rescalings in a narrow [0.8, 1.0] band, and a
   blockwise model can sit 50 % of the segment length off the uniform-α line. An earlier
   draft claimed blockwise α "cannot produce a new model"; that was wrong and is corrected
   in place.

2. **Accuracy cannot resolve position on this panel** (Phase 2b §2, Phase 3-preview §3).
   Qwen3's margin at the first answer token is negative at six of seven positions: it is
   below the decision threshold almost everywhere, so exact-match sees a flat floor while
   summed gold NLL sees a nine-nat range. Every mechanistic phase is therefore selected
   on NLL, with accuracy reported beside it.

3. **`multi_hop` was unmeasurable** (Phase 2b §4). 0/36 on both accuracy scorers at
   NLL 4.4–5.1. A `multi_hop_guided` variant now exists and works: it lowers NLL by
   2–6 nats and, for Qwen3, flattens the positional curve to non-significance.

4. **The positional deficit has a different shape in every task family** (Phase 3 §1).
   `multi_hop` and `state_constraint` peak in the true middle; `exact_retrieval` is a
   recency gradient with its worst point at p=0.17; `semantic_single_hop` varies by
   8 nats while its *middle-minus-edge* statistic is not significant at all. "The
   edge-middle gap" is therefore not one number per model, and any intervention has to
   say which shape it is targeting.

5. **Attention position is half absolute and half distance, and the two are
   complementary** (Phase 4b §2). Absolute-only explains 0.47 of the structure,
   distance-only 0.50, the additive model **0.92** — identical to within 0.9 % across all
   four models. So the absolute-binned Significance Matrix that SPEC §5.3 makes the
   primary grid is, by construction, blind to about half the positional structure. That
   is a limitation of the cheap intervention to report, not a defect to hide. Separately,
   Supa-A1.1 has the lowest edge bias and highest middle mass **on contexts with no
   content at all** — a structural correlate of its behavioural Phase 3 advantage,
   measured a completely different way.

6. **The matrices are strongly low-rank** (Phase 4/7). Rank 2 explains **90 %** of causal
   significance and **99 %** of positional bias. Positional behaviour in this family is
   controllable with very few parameters, which is what the compact
   `S[h,p] = a_h f(p) + c_h` parameterisation needs to be true.

7. **Qwen3's edge attention is a net cost, and weight deltas predict bias but not
   significance** (Phase 4 §2, Phase 7 §4). Attenuating Qwen3's terminal bin *lowers*
   gold NLL. Separately, per-head weight change correlates with edge bias (k_proj
   strongest in every row) but not with causal significance — and the sign is **positive
   for the Halo/Nimbus soups and negative for the Supa adapter**, so the two lineages did
   opposite things to positional bias.

## A metric artefact that reached a published claim

Every `semantic_single_hop` answer begins with the article *the*, and `multi_hop` answers
have the same "the &lt;role&gt;" form. Between-model differences in summed NLL on those
families are therefore inflated by confidence about a token that carries no retrieval
information. Decomposed on 154 paired scenario x position observations, Supa's apparent
−3.34 nat advantage over Qwen3 on `semantic_single_hop` is **−3.37 from the article and
−0.02 ± 0.14 from the content** — i.e. entirely artefactual. The claim "Supa-A1.1 is
better on all five families" is **withdrawn**; Halo's and Nimbus's *deficit* on the same
family survives the decomposition (+0.89, +0.65 content-only, both significant), as does
Supa's positional reversal (−0.97 ± 0.33 content-only, down from −2.97).

`multi_hop` has not yet been decomposed and its between-model numbers should be read as
upper bounds. `exact_retrieval` is unaffected — bare nonce codes, no leading article.

## Corrections recorded in place

Four analysis errors were caught and are documented where they happened, because all
four produced results that looked entirely real:

- **Phase 2b §7** — a position-accuracy check compared the *decisive* evidence sentence's
  achieved position against the *first* sentence's target. For the multi-evidence
  families those are different sentences, and the mismatch read as "position control is
  completely broken" (apparent mean error 0.17, max 0.97). Like-for-like it is 0.007.
  The generator was correct; the check was not.
- **Phase 3-preview §5** — pooling all `natural`/4K items per position mixes local
  rhetorical position into the global-position axis, because the
  `local_position_sweep` grid only emits p ∈ {0.02, 0.50, 0.98}. That puts a spurious
  step at exactly the three positions defining the edge-middle gap. All Phase-3 analysis
  holds local position fixed or models it explicitly.
- **Phase 3 §5, the position axis** — `state_constraint` and `multi_hop` carry several
  evidence sentences, only one of which is swept. Grouping on `target_positions[0]`
  collapsed `state_constraint` onto a single constant position. Same error class as the
  first correction above, which is why it was looked for.
- **Phase 3 §5, a sign inversion** — `paired_bootstrap(a, b)` returns `b - a`; it was
  called with the arguments reversed, so every model-vs-Qwen3 difference and every
  better/worse verdict was inverted. Caught by cross-checking the paired table against
  the per-position curves, which disagreed.

## Still to come

Phases 4–15, plus the greedy/accuracy pass and the validation split for Phase 3. See
the STATUS section of the lab [`README.md`](../README.md) for what is implemented versus
what has been run. The short version: the library exists and is tested, Phases 1–3 have
produced results on real checkpoints, and Phases 4 onward have not.
