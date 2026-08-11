# v2.3 — Preservation-Constrained Residual Capability Distillation

## Why v2.3 exists

v2.2 produced two unusually useful observations. First, ordinary broad fine-tuning could unlock much more performance on the v2.2 multi-domain distribution, but it catastrophically damaged the student's already-strong Real-RAG procedure, supported-answer, and faithfulness behavior. Second, the DEV-selected and sealed `greedy_soup` avoided that collapse with an extremely small composition: the frozen original student plus **0.5 × the v2.2 abstention LoRA delta**. That candidate kept the historical procedure path strong and improved supported answering/abstention, but its apparent multi-domain score was dominated by abstention cases rather than broad answerable reasoning.

v2.3 therefore does **not** broadly fine-tune the sealed v2.2 winner again. It treats the v2.2 greedy checkpoint as an immutable optimization anchor and asks a narrower question:

> Can the student acquire genuinely **answerable** reasoning and hard procedural capabilities that stronger-but-destructive teachers already demonstrate, while preserving the greedy anchor's grounded behavior?

This is a governed Phase 9 research branch. It does not replace the accepted release lineage automatically and it must finish behavioral selection before Student Optimization v2 / structural compression proceeds.

## Source-of-truth anchors

v2.3 keeps two immutable references:

1. **Optimization anchor — v2.2 greedy soup.** This is the checkpoint used as the student initialization and KL reference during residual training. By default the runner resolves it from `track_d_greedy_soup/final_model` under the supplied v2.2 run root.
2. **Historical anchor — v2.2 frozen original student.** This is evaluated for long-horizon regression visibility but is not silently substituted for the greedy baseline during v2.3 selection.

The old v2.2 sealed TEST is considered consumed. It is never assigned to a v2.3 tuning variable. v2.3 contains a fresh sealed generation and touches it only after one DEV/OOD-DEV candidate is frozen.

## End-to-end workflow

```text
v2.2 greedy_soup (immutable)
          |
          +------------------------+
          |                        |
          v                        v
fresh structural-OOD          preservation TRAIN rows
reasoning + procedural RAG    greedy already solves
          |                        |
          v                        |
run discovery teachers             |
Track A / Track B / procedure      |
          |                        |
          v                        |
gold-verified residual mining      |
          |                        |
     +----+----+                   |
     |         |                   |
     v         v                   |
Reasoning R  Procedure P           |
     |         |                   |
     +----+----+-------------------+
          |
          v
independent LoRA training
+ preservation replay
+ greedy-anchor KL
          |
          v
coarse + locally refined
capability-dose sweep
          |
          v
safe R / safe P / small joint family
          |
          v
DEV + structural OOD capability matrix
+ original Phase 9 preservation
+ historical procedure challenge
          |
          v
Pareto / preservation-guarded selection
          |
          +--> adapter-interference diagnostics
          +--> oracle-routing upper bound
          |
          v
optional NEW v2.3 sealed evaluation
```

## Fresh v2.3 data

### Direct structural-OOD reasoning corpus

Packaged at:

`data/phase_9_student_distillation/residual_capability_v23/direct_reasoning/`

The deterministic corpus contains **24,300** cases across **27 domains** and **12 reusable operations**. It is explicitly separated into answerable versus insufficient/conflicting-evidence cases so answerable reasoning cannot be inflated by abstention success.

| Split | Rows | Representation families |
|---|---:|---|
| `discovery` | 11,340 | prose, bullets, compact table, key-value |
| `dev` | 4,860 | email, chat, ticket |
| `ood_dev` | 4,860 | JSON, YAML, CLI, log |
| `fresh_sealed` | 3,240 | XML-like, config, mixed-document |

Overall: 20,250 answerable rows and 4,050 abstention rows. Targets remain compact final answers; no chain-of-thought target is published.

The 12 operations are:

`compare_metric`, `filter_then_rank`, `aggregate_delta`, `capacity_fit`, `state_transition`, `dependency_order`, `evidence_binding`, `rule_exception`, `multi_hop_relation`, `diagnostic_priority`, `insufficient_information`, and `conflicting_evidence`.

### Fresh procedural Real-RAG corpus

Packaged at:

`data/phase_9_student_distillation/residual_capability_v23/procedural_rag/`

It contains **7,200** cases across **1,200 topic families**, split by topic rather than by individual row. Eight procedure families are represented: latency, capacity, threshold, semantic-version ordering, priority, approval routing, backoff, and quota. Each topic contains four supported records, one explicitly missing record, and one structured conflict case.

| Split | Rows |
|---|---:|
| `discovery` | 4,320 |
| `dev` | 1,080 |
| `ood_dev` | 1,080 |
| `fresh_sealed` | 720 |

The new corpus uses retrieval selection policy **`query_aware_v23`**. Historical Phase 9 and v2.1.1 evaluations continue to use `query_aware_v21` for reproducibility.

`query_aware_v23` adds one narrow behavior for fresh cases: when the query contains an explicit record/item/case locator and retrieval contains only conflicting locator values, the selector fails closed instead of treating a same-topic sibling as sufficient support. It does not rewrite historical v2.1 selection behavior.

## Oracle capability audit and residual mining

The discovery stage evaluates the greedy anchor plus three v2.2 teachers:

- Track A broad reasoning;
- Track B anchored reasoning;
- the procedure specialist.

A teacher prediction is never accepted merely because it differs from the anchor. A row is eligible for residual learning only when:

```text
anchor is wrong
AND
at least one teacher is verified correct by deterministic packaged gold
```

Rows where all candidate models fail go to an unresolved bucket rather than becoming teacher-labelled SFT data.

The miner writes:

- `reasoning_residual.reviewed.jsonl` — answerable direct residuals only;
- `procedure_residual.reviewed.jsonl` — answerable fresh procedural residuals only;
- unresolved buckets;
- `preservation_replay.reviewed.jsonl` — TRAIN-only examples that the greedy anchor itself demonstrably solves;
- `RESIDUAL_MINING_AUDIT.json/.md` — model accuracies, recoverable residual counts, and the **oracle-union ceiling** (fraction solved by at least one discovery model).

The oracle-union ceiling is a pre-training sanity check. It shows whether the teachers actually contain useful information that the greedy anchor lacks before GPU time is spent on residual training.

## Independent residual adapters

v2.3 intentionally separates two skills:

- **Residual R:** answerable cross-domain reasoning;
- **Residual P:** fresh hard procedure/evidence application.

They are trained independently from the same greedy anchor so attribution remains clean.

Default mixes:

| Adapter | Total rows | Residual fraction | Preservation fraction |
|---|---:|---:|---:|
| Reasoning R | 6,000 | 60% | 40% |
| Procedure P | 5,000 | 48% | 52% |

Training only proceeds when mining discovers enough gold-verified residual rows. Defaults require at least 250 reasoning residuals and 60 procedure residuals. Repetition is bounded by a maximum repeat factor of 8.

Default training:

| Adapter | LR | Steps | LoRA rank | Greedy-anchor KL | Residual-row KL multiplier |
|---|---:|---:|---:|---:|---:|
| R | `6e-6` | 180 | 16 | 0.30 | 0.35 |
| P | `5e-6` | 140 | 16 | 0.35 | 0.30 |

Preservation rows keep KL multiplier 1.0. Residual rows use a smaller per-row multiplier so the student is allowed to move more on known failures while staying close to the greedy anchor on behavior it already performs correctly.

`train_student.py` therefore supports the per-example metadata field:

`metadata.v23_anchor_kl_multiplier`

The effective KL strength for a batch is the configured global anchor weight multiplied by the batch mean of that field.

## Capability-dose calibration

v2.2 established that adapter strength is not binary: the abstention delta was useful at 0.5 but destructive at 1.0. v2.3 treats this explicitly as **capability-dose calibration**.

For each residual adapter the default sweep tests:

`0.10, 0.25, 0.50, 0.75, 1.00`

Then it automatically performs a local refinement around the best coarse dose (±0.15 in 0.05 increments). Zero dose / the unchanged greedy anchor remains a valid fallback. Non-winning materialized dose checkpoints are deleted by default to avoid filling disk; use `-KeepDoseModels` to retain them.

Dose selection uses DEV probes plus protected original Phase 9 Real-RAG and historical procedural evidence. It never uses the old v2.2 sealed TEST or the fresh v2.3 sealed split.

## Candidate composition

After R and P are dose-calibrated, v2.3 evaluates:

- unchanged `greedy_anchor`;
- best reasoning dose;
- best procedure dose;
- best-R + best-P;
- half-strength joint;
- R full / P half;
- R half / P full.

These are a small preregistered family rather than an open-ended coefficient search. The exact composition is materialized as:

`W = W_greedy + alpha_R * Delta_R + alpha_P * Delta_P`

The unchanged greedy anchor remains a first-class selectable candidate.

## Evaluation changes that matter

v2.3 separates answerable capability from abstention. The direct evaluator reports:

- `overall_correctness`;
- `answerable_correctness`;
- `answerable_hard_correctness`;
- `macro_answerable_domain_correctness`;
- `abstention_correctness`;
- `false_abstention_rate`;
- `answer_coverage`;
- `short_usable_rate`.

This prevents a candidate from looking like a general-reasoning improvement merely because it learned to abstain on all insufficient-evidence rows.

Every serious candidate is also evaluated on:

- original Phase 9 DEV Real-RAG under `query_aware_v21`;
- historical v2.1.1 procedural challenge DEV under `query_aware_v21`;
- fresh v2.3 procedural RAG DEV + OOD-DEV under `query_aware_v23`;
- v2.3 direct DEV + structural OOD-DEV;
- v2.1.2 and v2.2 reasoning sets as diagnostics.

Protected original metrics remain procedure correctness, supported-answer correctness, evidence faithfulness, correct abstention, conflict-safe handling, and social usability. Missing primary/protected metrics technically invalidate a candidate; null evidence is never treated as zero regression.

The official selection policy is `preservation_guarded`, with a balanced policy also reported and policy-sensitivity disagreement surfaced explicitly. Candidate comparison remains relative to the greedy anchor and reports a Pareto frontier.

## Diagnostics

### Adapter-interference map

For the learned R/P adapters and selected v2.2 adapters, v2.3 computes effective LoRA delta statistics including:

- Frobenius norm;
- cosine similarity;
- sign agreement;
- top-magnitude overlap;
- per-layer cosine similarity.

This is descriptive/mechanistic evidence, not a promotion score.

### Oracle-routing upper bound

v2.3 also reports a non-production upper bound that selects the correct candidate per example if any evaluated specialist gets the case right. It answers: are remaining limitations mainly in the individual specialists, or in composition/routing?

The oracle router is diagnostic only; v2.3 does not change production inference into a multi-model ensemble.

## Sealed-data policy

The v2.2 sealed TEST has already been consumed. v2.3 records that fact in:

`data/phase_9_student_distillation/residual_capability_v23/SEALED_REGISTRY.json`

The first v2.3 run is DEV/OOD only. After review, re-run the same root with `-ReuseExisting -RunFreshSealedTest`. The sealed stage evaluates **only**:

1. v2.3 direct `fresh_sealed`;
2. v2.3 procedural Real-RAG `fresh_sealed` using the already-frozen discovery calibration.

No sealed recalibration, training, merging, coefficient change, or candidate switch is allowed.

## Recommended command

Run against the completed v2.2 artifact root that contains the frozen anchor, greedy model, and teacher adapters:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_residual_capability_v23.ps1 `
  -V22RunRoot ".\artifacts\phase_9_capability_composition_v22_lr1e4_s1200" `
  -OutputRoot ".\artifacts\phase_9_residual_capability_v23" `
  -RequireCuda
```

The script automatically resolves the v2.2 artifacts and runs data validation, discovery evaluations, residual mining, two residual trainings, dose calibration, DEV/OOD evaluation, composition, selection, interference diagnostics, and the oracle upper bound without requiring manual stage launches.

Do **not** pass `-RunFreshSealedTest` on the first run.

After reviewing `RESIDUAL_MINING_AUDIT.md`, the two `DOSE_SWEEP_RESULT.md` reports, `RESIDUAL_CAPABILITY_V23_SELECTION.md`, and diagnostics, freeze the chosen DEV/OOD decision and run:

```powershell
.\scripts\phase_9_student_distillation\run_phase_9_residual_capability_v23.ps1 `
  -V22RunRoot ".\artifacts\phase_9_capability_composition_v22_lr1e4_s1200" `
  -OutputRoot ".\artifacts\phase_9_residual_capability_v23" `
  -ReuseExisting `
  -RunFreshSealedTest `
  -RequireCuda
```

If the anchor/reference forward pass creates too much GPU pressure, use `-AnchorDevice cpu`. This lowers GPU pressure but can be much slower.

## Stop conditions

A v2.3 run is informative even if it retains the greedy anchor. Important outcomes include:

- oracle-union ceiling too small -> teachers do not contain enough recoverable residual skill;
- enough residual data but every dose loses preservation -> parameter direction remains incompatible;
- R works but P fails -> procedural residual is a distinct interference problem;
- P works but R fails -> broad answerable transfer remains the bottleneck;
- R and P work alone but joint composition fails -> composition/interference is the bottleneck;
- oracle routing greatly exceeds merged candidates -> routing/composition is the bottleneck;
- no oracle advantage -> specialist quality itself is the bottleneck.

A positive v2.3 result should improve answerable DEV and structural OOD capability while preserving the greedy anchor strongly enough to earn a positive relative-opportunity margin. Only then should a new fresh sealed readout be considered.
