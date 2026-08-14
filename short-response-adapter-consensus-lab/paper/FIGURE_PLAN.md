# Figure Plan

Every plotted value must be generated from a machine artifact listed in
`EVIDENCE_LEDGER.md`. Draft figures use colorblind-safe palettes, direct labels,
vector output (PDF/SVG), and a grayscale legibility check.

## Rendered artifact status

The current renderer is
`experiments/phase2_mechanism/make_paper_figures.py`. It emits PNG review copies,
SVG publication copies, and `paper/figures/figure_metadata.json`, which records
the generator hash, source-artifact hashes, output hashes, and evidence role for
every rendered figure.

| Planned figure | Current artifact | Status |
|---|---|---|
| 1 | `paper/figures/figure01_architecture.{png,svg}` | RENDERED / VISUALLY VERIFIED |
| 2 | `paper/figures/figure02_hard_vs_soft.{png,svg}` | RENDERED / VISUALLY VERIFIED; synthetic only |
| 3 | `paper/figures/figure03_committee_scaling.{png,svg}` | RENDERED; cached all-committee and selected free-running DEV panels |
| 4 | `paper/figures/figure04_calibration.{png,svg}` | RENDERED / VISUALLY VERIFIED; DEV calibration summary |
| 5 | `paper/figures/figure05_rescue_harm.{png,svg}` | RENDERED / paired TEST |
| 6 | `paper/figures/figure06_coalition_heatmap.{png,svg}` | RENDERED / VISUALLY VERIFIED; descriptive, noncausal |
| 7 | `paper/figures/figure07_quality_compute.{png,svg}` | RENDERED / TEST hardware-descriptive |
| 8 | `paper/figures/figure08_pilot_vs_full.{png,svg}` | RENDERED / VISUALLY VERIFIED |
| 9 | `paper/figures/figure09_teacher_vs_free.{png,svg}` | RENDERED; paired same-split DEV |
| 10 | `paper/figures/figure10_divergence_timing.{png,svg}` | RENDERED / VISUALLY VERIFIED; descriptive |

Additional rendered artifacts are `figure_primary_test_accuracy`,
`appendix_shapley`, `appendix_temperature_interventions`, and
`appendix_diversity`. The Figure 6 seven-bit masks use the fixed expert order
Track A, Track B, General, Grounding, Procedure, Abstention, Social; `1` means
the expert supported the winning token at that step.

Superseded duplicate-name renders are quarantined under
`paper/figures/_legacy_aliases/` for recoverability. They are not referenced by
`figure_metadata.json` and are not evidence artifacts.

## Figure 1 — Hard adapter consensus architecture

**Purpose:** define decision-space composition and the shared-prefix recurrence.

**Layout:** prompt and accepted prefix enter seven parallel lanes. Each lane is
the same frozen backbone plus one active adapter and emits a top-1 token. A vote
counter selects one token, appends it to the common prefix, and loops.

**Required annotations:** one resident backbone; independently activated
adapters; one expert/one vote; deterministic tie rule; ordinary EOS; all experts
see the newly accepted shared prefix.

**No numeric source required.** Architecture must match
`paper/METHODS_AND_MATH.md` and verified implementation hashes.

## Figure 2 — Hard versus soft fusion

**Purpose:** isolate directional choice from confidence magnitude.

**Panels:**

- hard: logits → per-expert argmax → equal votes → plurality;
- soft: logits → per-expert softmax → arithmetic probability mean → argmax;
- calibration intervention: expert-specific (T_i) leaves hard proposals
  unchanged but changes soft distributions.

Include a small synthetic example with no empirical performance number.

## Figure 3 — Committee-size scaling

**Source:** E-TF-ANALYSIS and frozen E-COMMITTEE-FREERUN when complete.

**Axes:** K=1…7 versus teacher-forced all-token example accuracy. Show:

- mean across all committees at K with committee-distribution interval;
- best committee at K;
- fixed full K=7 point;
- separate panels/lines for hard, raw probability, calibrated probability, and
  selected rank-only aggregation.

Do not infer monotonicity or a threshold from the best-at-K envelope alone.
The selected free-running systems curve uses a different estimand (exact
generated-sequence accuracy), so it occupies a separate third panel rather
than sharing the local all-gold-token axis. That panel also shows balanced
answer/abstention sequence accuracy. The 12-condition panel is frozen; Figure
3 is regenerated only after its strict analyzer completes.

## Figure 4 — Calibration robustness

**Source:** E-TF-ANALYSIS temperature interventions.

**Panel A:** mean top-1 confidence versus gold-token top-1 accuracy for each
expert on DEV, with the identity line and fitted temperature labels.

**Panel B:** NLL versus temperature for each expert on DEV, with fitted
temperatures marked by stars.

**Panel C:** K=7 soft-fusion all-gold-token example accuracy when all experts
share temperature, with the hard raw reference.

**Panel D:** K=7 soft-fusion all-gold-token example accuracy when one expert at
a time is perturbed, with the hard raw reference.

**Reference line:** raw hard-vote local decisions. Label top-1 proposals and
unique-plurality winners as invariant under positive scaling, while pointing to
the separate confidence-tie-boundary audit for unequal expert temperatures.
Mark fitted shared/per-expert temperatures without implying TEST confirmation.

## Figure 5 — Rescue/harm flow

**Source:** E-RESCUE.

Two compact 2×2 paired-state mosaics (the accessible fallback to a Sankey):

- Track A correct/wrong → Hard K=7 correct/wrong;
- Probability K=7 correct/wrong → Hard K=7 correct/wrong.

Rows show baseline correct/wrong and columns show Hard K=7 correct/wrong. Every
cell reports count and percentage, including both-correct, rescue, harm, and
both-wrong states. Rescue and harm receive distinct accessible colors; the
numerical labels carry the interpretation without relying on color alone.

## Figure 6 — Coalition heatmap

**Source:** E-RESCUE coalition detail and later counterfactual subset analysis.

Panel A uses experts as rows and reasoning families as columns; cell color is
the fraction of tokens in that family for which the expert supported the
winning token. Panel B uses the ten most frequent winner-support masks as rows
with the same columns; cell color is within-family coalition frequency, not
accuracy. Both are explicitly descriptive and noncausal. Free-running
leave-one-out category deltas are reported in the tables rather than mixed into
this TEST-path coalition figure.

## Figure 7 — Quality/compute frontier

**Source:** full TEST raw for frozen conditions; later free-running DEV/adaptive
artifact.

Plot exact accuracy against:

- measured latency;
- expert-token forwards;
- adapter storage;
- optional energy estimate only if measured.

Use separate visual encodings for TEST-confirmatory points and DEV-selected
points. Connect adaptive policies only when they share a threshold family.

## Figure 8 — Pilot versus full TEST

**Source:** E-PILOT-RAW and E-CORNERSTONE-RAW.

Paired slope or grouped-dot plot for backbone, Track A, abstention, probability
K=7, and hard K=7. Annotate n=32 and N=4,050 prominently. The caption frames the
pilot as feasibility evidence and the reversal as small-sample instability.

## Figure 9 — Teacher-forced versus free-running gain

**Source:** E-TF-ANALYSIS and E-FREERUN-DEV.

For each frozen method pair, compare local gold-prefix advantage with
free-running sequence advantage. Do not mix TEST free-running and DEV teacher-
forced values. This is the decisive trajectory-correction figure; the preserved
ordering does not support a hard-specific corrective trajectory effect.

## Figure 10 — Hard/soft first-divergence timing

**Source:** E-RESCUE.

Histogram of first divergence (t), stratified by hard-only, probability-only,
and both-wrong outcomes. State that both methods share the same prefix at the
divergence position and that outcome association is descriptive.
