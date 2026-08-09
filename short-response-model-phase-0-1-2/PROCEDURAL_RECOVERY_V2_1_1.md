# v2.1.1 — Preservation-First Procedural Transfer Recovery

## Why v2.1.1 exists

v2.1 successfully isolated the original Phase 9 procedural collapse. The unchanged selected student went from **0.000** Real-RAG procedure correctness under legacy retrieval to **0.9533** on original DEV under `query_aware_v21`, showing that retrieval collision handling—not a total absence of procedural capability—was the dominant historical fix.

However, the v2.1 recovery training path was not acceptable as a final solution. On the observed DEV run:

- unchanged weights + query-aware retrieval: **0.9533** original procedure correctness;
- recovered weights + query-aware retrieval: **0.2267**;
- training-only original-path change: **−0.7267**;
- challenge procedure correctness: **0.0015 → 0.1543**;
- supported correctness: **0.6340 → 0.4887**;
- evidence faithfulness: **0.9794 → 0.8393**;
- relative training opportunity margin: approximately **+0.00018**, effectively neutral while hiding a very large regression in the strongest original path.

That is useful research evidence, but it is not a production recovery. v2.1.1 changes the goal:

> **Treat query-aware retrieval as candidate zero and accept weight changes only when residual procedural gain is larger than the quality they overwrite.**

The two observed v2.1 reports used as the design input are preserved under `reports/PHASE_9_V21_*`.

---

## Main design changes

### 1. Query-aware retrieval becomes the baseline to preserve

v2.1.1 does not try to retrain away a problem that the retrieval selector already solved. Every trained checkpoint is compared against:

```text
candidate zero = original student weights + query_aware_v21 retrieval
```

If none of the trained checkpoints earns its preservation cost, the official result is intentionally:

```text
QUERY_AWARE_ONLY_SELECTED
```

That is not a failed experiment. It means the safest and strongest system is the original student plus the retrieval-policy correction.

### 2. Stage A is a short residual-binding burst

v2.1 trained on thousands of collision-heavy procedure rows. v2.1.1 restricts Stage A to a stratified **768-row** hard-collision view and defaults to:

```text
60 optimizer steps
LR 1e-5
LoRA rank 16
```

Stage A is evaluated immediately on both:

1. original Phase 9 DEV + query-aware retrieval;
2. v2.1.1 procedural challenge DEV.

Its adapter is therefore a first-class candidate; it is no longer hidden behind Stage B.

### 3. Stage B is preservation-first rather than collision-first

The new Stage B view contains **4,820 rows**:

| Component | Rows |
|---|---:|
| Original factual answer preservation | 1,200 |
| Original factual abstain/conflict preservation | 700 |
| Verified original procedure-answer preservation | 800 |
| Original procedure-boundary preservation | 300 |
| Social preservation/replay | 120 |
| New clean/moderate v2.1.1 procedure | 900 |
| New hard residual collision | 350 |
| New procedure boundary cases | 450 |
| **Total** | **4,820** |

So **3,120 / 4,820 = 64.7%** of Stage B is preservation replay from original Phase 9 **train only**.

This directly corrects the v2.1 imbalance, where approximately 4.6k new procedural examples were mixed with only about 1.2k non-procedure preservation rows.

Importantly, v2.1.1 also replays **verified original procedure answers**. v2.1 excluded all historical procedure rows from preservation because of a known latency-tie defect. v2.1.1 instead excludes only the **18 known inconsistent rows** and preserves the remaining clean original procedure supervision.

### 4. Stage B is split into two micro-consolidation checkpoints

Instead of one long consolidation run, v2.1.1 creates two bounded checkpoints:

```text
Stage B1: 60 steps @ 5e-6
Stage B2: 60 steps @ 3e-6
```

The workflow evaluates after Stage A, B1, and B2. This gives the research harness an explicit early-stopping ladder:

```text
original/query-aware baseline
        ↓
Stage A
        ↓ evaluate both DEV sets
Stage B1
        ↓ evaluate both DEV sets
Stage B2
        ↓ evaluate both DEV sets
DEV-only candidate selection
```

This is intentionally **not** another broad LoRA hyperparameter sweep.

### 5. Aggressive schedules require an explicit override

The dedicated v2.1.1 runner rejects by default:

- any recovery LR above `3e-5`;
- more than `400` total recovery optimizer steps.

To exceed either, the user must explicitly pass:

```powershell
-AllowAggressiveTraining
```

This is a guardrail against accidentally repeating a high-LR / long-step recovery run when the current research objective is preservation.

---

## New procedural corpus

The canonical v2.1.1 corpus contains:

```text
6,560 total rows
4,976 train
792 DEV
792 sealed TEST
5,040 factual knowledge documents
2,320 trusted procedure records
24 procedure families
```

### Evidence difficulty is now mixed

v2.1 intentionally made nearly every answerable case collision-heavy. v2.1.1 uses a more realistic distribution:

| Profile | Canonical rows |
|---|---:|
| Clean binding | 1,440 |
| Moderate distractor | 1,440 |
| Hard sibling collision | 2,880 |
| Missing/conflicting boundary | 800 |

The model therefore learns ordinary evidence use and progressively harder binding instead of seeing an overwhelmingly adversarial distribution.

### 24 procedure families

The new corpus retains the original deterministic families and adds real-world operational cases such as:

- incident severity routing;
- API rate-limit arithmetic;
- inventory reorder decisions;
- shipping SLA/cost selection;
- freshness/source selection;
- dependency/major-version compatibility;
- role + MFA access policy;
- health/latency failover selection;
- inclusive maintenance windows;
- backup retention rules.

Other retained families cover semantic versions, ISO dates, latency, priority, capacity, quality/price filtering, thresholds, ranges, approval conjunctions, retry backoff, deployment prerequisites, tie-breaking, tolerance, and quota arithmetic.

The cases are still synthetic enough to have deterministic, auditable gold answers, but their prompt/evidence shapes are closer to real operational RAG than generic `audit record N` arithmetic alone.

---

## New candidate-selection policy

`select_procedural_recovery_v211.py` compares:

```text
query_aware_only
stage_a
stage_b1
stage_b2
```

on **DEV only**.

The protected original metrics are:

- original Real-RAG procedure correctness;
- supported-answer correctness;
- evidence faithfulness;
- correct abstention;
- conflict-safe handling;
- social usability.

For each trained candidate:

```text
positive residual challenge gain
-
weighted relative preservation cost
=
opportunity margin
```

Quality improvements reduce a metric's loss to zero; they cannot create negative loss that hides another regression.

Two policies are reported:

1. `balanced` — equal weights across protected metrics;
2. `preservation_guarded` — gives extra weight to original procedure correctness, supported correctness, and evidence faithfulness.

The official v2.1.1 policy is `preservation_guarded`.

If the two policies choose different checkpoints, the report marks:

```text
POLICY_SENSITIVITY_REVIEW_REQUIRED
```

instead of silently promoting a fragile result.

---

## Recommended command

For the same selected warm-344M student lineage used in the v2.1 experiment:

```powershell
$StudentBase = '.\artifacts\phase_9_grounded_recovery_v1_7_5_qwen\warm_344m_initialization'
$StudentAdapter = '.\artifacts\phase_9_grounded_recovery_v1_7_5_qwen\warm_sft_sweep\lr1e4_s1200\training\adapter'
$Out = '.\artifacts\phase_9_procedural_recovery_v211_lr1e4_s1200'

.\scripts\phase_9_student_distillation\run_phase_9_procedural_recovery_v211.ps1 `
  -StudentModel $StudentBase `
  -StudentAdapter $StudentAdapter `
  -OutputRoot $Out `
  -RequireCuda
```

Do **not** pass `-RunSealedTest` on the first run. Inspect:

```text
$Out\PROCEDURAL_RECOVERY_V211_SELECTION.md
$Out\procedural_recovery_v211_selection.json
```

The report will tell you whether DEV selected:

```text
query_aware_only
stage_a
stage_b1
stage_b2
```

### Optional legacy reference

The historical legacy/query-aware difference is already established, so v2.1.1 does not spend that evaluation cost by default. To reproduce it in the new root:

```powershell
... -RunLegacyReference
```

### Sealed TEST

Once you accept the DEV-selected candidate and want its frozen test snapshot:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_procedural_recovery_v211.ps1 `
  -StudentModel $StudentBase `
  -StudentAdapter $StudentAdapter `
  -OutputRoot $Out `
  -ReuseExisting `
  -RunSealedTest `
  -RequireCuda
```

The runner reads the DEV selection file and evaluates **that exact selected candidate**, including the unchanged original adapter if `query_aware_only` won. TEST does not recalibrate the evidence gate or train anything.

---

## What result would count as progress?

The most useful v2.1.1 result is not necessarily the checkpoint with the largest challenge score. A strong result is one where:

```text
original query-aware procedure quality remains near the unchanged baseline
+
supported/evidence-faithfulness quality remains stable
+
hard procedural challenge correctness improves
```

If Stage A produces that and later Stage B checkpoints regress, select Stage A.

If all weight-updated checkpoints regress more than they gain, select `query_aware_only` and treat the training intervention as rejected. At that point the next experiment should focus on retrieval/reranking and targeted residual data rather than forcing weight changes.

---

## Should other students receive this training?

Not as a mass rerun yet. First determine whether v2.1.1 can produce a preservation-positive trained candidate on the primary student.

If it can, use the **same fixed recipe** on one or two replication students. If it cannot and `query_aware_only` wins again, that is evidence that the original problem is primarily a RAG evidence-selection problem rather than a weight-training problem, and broad procedural retraining should not become mandatory for every student.

---

## Downstream architecture work

Student Optimization v2 remains downstream. Do not use a Stage A/B checkpoint merely because it has nonzero challenge accuracy. The accepted Phase 9 student for structural optimization should be the DEV-selected v2.1.1 path after human review.
