# Corrected R7 v2 report

## Outcome

R7 v2 repairs the design invalidity in the frozen R7 v1 benchmark, but Halo
does not pass the corrected full-cache capability gate.  Compression testing
on R7 v2 therefore stops before any Significance Matrix condition is run.

The eight-cluster B0 screen is exactly balanced (`4 YES`, `4 NO`).  Halo
returned `YES` for all eight cases, scoring `4/8`: `4/4` on true-YES cases and
`0/4` on true-NO cases.  This equals the always-YES baseline and fails the
prospective gate requiring at least four correct overall and at least one
correct case on each answer side.

## Frozen v1 provenance

The original ladder remains unchanged at
[`results/benchmarks/halo_r1_r8_ladder_v1`](results/benchmarks/halo_r1_r8_ladder_v1).
It remains classified `INVALID_UNINFORMATIVE`: six of seven seed clusters had
the answer `NO`, and the rendered solver branched directly on one lock-fact
sentence.  R7 v1's raw model score is not capability evidence.

The corrected dataset's
[`design_audit.json`](results/benchmarks/halo_r7_v2/design_audit.json) records
the immutable v1 hashes:

- `manifest.json`: `d437db4bde56fe8ffde77f8df8a7e4ebf28c076e6c09ba80c1951219186c5bc0`
- `inference.jsonl`: `9cfc7250990b895be221441f2da32f0207036b2df8e2f2d65c4eaadbdc47e751`
- `evaluation.jsonl`: `905ca8d3200716078953d52e2a4547d338d0eb9442f03fd65033cc65926aa7e3`

Those hashes were checked both before and after building v2.

## Corrected design

The versioned `r7-v2.0.0` generator creates 80 rendered cases from eight
independent semantic clusters.  It crosses three clustered evidence depths
with seven fixed layout controls, covering eight layouts overall.  The full
dataset is `40 YES / 40 NO`; the frozen screen is `4 YES / 4 NO` and has two
clusters in every answer-by-active-branch cell.

Each prompt contains:

- a two-rule base forward chain and a base decision;
- two competing two-rule override branches, one active for the queried unit;
- one `READY` and one `NOT READY` override sentence;
- exactly two `READY` and two `NOT READY` decision conclusions overall;
- opaque segment IDs that do not expose evidence/distractor roles.

The independent rendered solver accepts only question text and segment text.
It parses the rule language, computes a fixed point, requires a named lower
decision before an override can fire, and resolves the highest-priority status.
It never reads the seed, oracle, answer, segment role, or segment ID.  A
counterfactual branch-fact swap changes the fired override and flips the answer.

Each case has ten outcome-specific exact evidence segments.  The proof alone
resolves the answer, and deleting any one annotated segment either makes the
program unresolved or changes its answer.  The inactive override branch is not
included in the exact evidence annotation.

## Artifacts

- Dataset manifest:
  [`results/benchmarks/halo_r7_v2/manifest.json`](results/benchmarks/halo_r7_v2/manifest.json)
- Design audit:
  [`results/benchmarks/halo_r7_v2/design_audit.json`](results/benchmarks/halo_r7_v2/design_audit.json)
- Inference-only view:
  [`results/benchmarks/halo_r7_v2/inference.jsonl`](results/benchmarks/halo_r7_v2/inference.jsonl)
- Answer/evidence view:
  [`results/benchmarks/halo_r7_v2/evaluation.jsonl`](results/benchmarks/halo_r7_v2/evaluation.jsonl)
- B0 screen manifest:
  [`results/runs/halo_r7_v2_capability_screen/manifest.json`](results/runs/halo_r7_v2_capability_screen/manifest.json)
- B0 summary:
  [`results/runs/halo_r7_v2_capability_screen/summary.json`](results/runs/halo_r7_v2_capability_screen/summary.json)
- Response audit:
  [`results/runs/halo_r7_v2_capability_screen/result_audit.json`](results/runs/halo_r7_v2_capability_screen/result_audit.json)

## Scope and limitations

This is an eight-cluster exploratory capability screen, not an inferential or
population result.  It used only the published Halo Qwen3-0.6B core on CPU in
bf16 with a full cache.  Rineya/R2 and Aegis checkpoints were excluded.  No
compressed condition ran, so the result says nothing about Significance Matrix
quality on R7 v2.  The correct action is to retain R7 v2 as a valid benchmark,
record Halo's base-capability stop, and avoid selecting only the four B0-correct
YES cases for a compression comparison.
