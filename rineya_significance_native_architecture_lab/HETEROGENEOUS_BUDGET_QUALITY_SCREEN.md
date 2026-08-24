# Halo heterogeneous-budget quality screen

Status: **complete exploratory screen; not registered and not confirmatory**.

This bounded Experiment B screen reused the 12 case IDs and frozen v1 evidence
endpoint from `halo_endpoint_validation_v2_registered`. The new allocation
conditions and contrasts were not part of that registration. The run therefore
remains exploratory even though the source cases and scoring rule were frozen.

## Design

Each of the 12 independent clusters ran exactly three query-conditioned cells:

- `B3_UNIFORM`: the existing rectangular B3 cache;
- `B3_ENTROPY`: live entropy-proportional layer/KV-head allocation;
- `B3_SHUFFLED`: the same live statistic shuffled across units as a specificity
  control.

All cells used Halo, CPU bf16, 128-token chunks, `S/C`, 4 sinks, 32 recent
tokens, identical query probes, and an exact logical boundary budget of
`28 layers × 8 KV heads × 128 = 28,672` entries. Adaptive units were bounded to
96–160 entries. The first boundary is necessarily uniform because every unit
has only 128 entries; each adaptive condition produced heterogeneous allocation
at both later boundaries in every case (24 heterogeneous boundaries per
condition).

Before any cell or outcome access, the same process ran rectangular B3 against
the padded runtime with a uniform 128-entry allocation. Logits, K/V/S/C,
positions, entry IDs, counters, and allocation reports were bitwise exact;
maximum logit delta was zero. The evaluation view was neither hashed nor parsed
until all 36 predictions and valid-only resident-position artifacts were
durable.

## Results

| Condition | Correct | Mean frozen endpoint | Mean prefill logical bytes | Mean prefill physical bytes | Mean post-decode padding overhead |
|---|---:|---:|---:|---:|---:|
| B3 uniform | 5/12 | 0.21131 | 23,732,651 | 23,732,651 | 0 |
| B3 entropy | 8/12 | 0.22321 | 23,732,651 | 26,134,358 | 2,403,854 |
| B3 shuffled | 5/12 | 0.23810 | 23,732,651 | 26,852,148 | 3,121,756 |

Logical prefill payload was exactly matched. Physical bytes were not: padding
made entropy storage about 9.7% larger than its post-decode logical payload and
shuffled storage about 12.5% larger. Generated lengths differed, so post-decode
logical totals are not used as the matched-budget proof.

Cluster-aware paired estimates (candidate minus baseline) were:

| Contrast | Outcome | Mean difference | 95% cluster bootstrap interval | Exact sign-flip p |
|---|---|---:|---:|---:|
| Entropy − uniform | frozen endpoint | +0.01190 | [−0.01637, +0.03869] | 0.4834 |
| Entropy − uniform | answer correctness | +0.25000 | [0.00000, +0.50000] | 0.2500 |
| Shuffled − uniform | frozen endpoint | +0.02679 | [approximately 0, +0.05208] | 0.1074 |
| Shuffled − uniform | answer correctness | 0.00000 | [0.00000, 0.00000] | 1.0000 |
| Entropy − shuffled | frozen endpoint | −0.01488 | [−0.03720, +0.00744] | 0.2988 |
| Entropy − shuffled | answer correctness | +0.25000 | [0.00000, +0.50000] | 0.2500 |

Strict all-token evidence survival remained 0/12 in every condition.

## Interpretation

Entropy allocation produced three additional correct answers in this small
screen, but the paired answer test is not significant and the contrast was not
registered. The frozen retention endpoint does not separate entropy from
uniform. More importantly for specificity, shuffled allocation had the highest
mean retention and was numerically above entropy. The current evidence therefore
does **not** support a claim that the entropy statistic is the operative cause
of improved retention or answers.

The prior registered B3 execution used Torch 2.13.0 CPU and Transformers 5.15.0;
this screen used Torch 2.12.1 and Transformers 5.14.1. Its uniform cell is the
within-run comparator backed by the same-process identity gate. No cross-runtime
reproduction claim is made against the earlier B3 artifact.

## Claim limits

- This is 36 exploratory cells over 12 frozen clusters, not a newly registered
  test or population estimate.
- Padded storage is not a memory-saving implementation of head raggedness.
- Runtime timings are CPU research measurements, not deployment results.
- The answer-key view was used only after prediction persistence; no oracle
  evidence or answer information influenced allocation.

Authoritative artifacts are under
`results/runs/halo_heterogeneous_budget_quality_screen_v1/`.
