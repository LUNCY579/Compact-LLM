# MRS-1 Phase II continuation — draft pending freeze

This document specifies the future continuation of the same open-development experiment. It is not an executed dataset acceptance, training permit, trained-model receipt, or scheduled job. Freeze this document with the Phase I protocol before observing new-candidate outcomes. Preserve the exact Phase I A/B evidence, old comparator, bank, prompts, role projections, parser, selector, inference policy and budgets.

The user confirmed `Rineya_LCR_Full_v1_6000` as the intended dataset. Its [completed P0 audit](evidence/dataset-acceptance-001/DATASET_ACCEPTANCE.md) binds manifest `65995575f13187ef78464e4120d1768696dd9114c7dfd8a379ed3ca6e28b266d` and records STOP_NOT_ACCEPTED. The future trained-checkpoint identity remains unbound. The findings require a repaired immutable dataset revision and a new P0 audit; do not equate folder presence or a producer report with acceptance. No automatic monitoring, training, checkpoint export or campaign expansion is installed. Training remains closed until binding P0 checks pass, P1 is frozen, appropriate authority exists and resource gates pass; the current host-memory gate blocks even inference.

Normative companions are [MULTI_REASONING_POSTTRAIN_CONTRACT.md](MULTI_REASONING_POSTTRAIN_CONTRACT.md), [MULTI_REASONING_PROTOCOL.md](MULTI_REASONING_PROTOCOL.md), [MULTI_REASONING_PREREG.md](MULTI_REASONING_PREREG.md), and [FUTURE_2X2_ANALYSIS_PLAN.md](FUTURE_2X2_ANALYSIS_PLAN.md). The eventual verified freeze digest binds their exact bytes. Where a necessary recipe choice remains open, record and freeze it before its downstream stage; never infer a passing status from a completed template.

## P0 — Accept the identified dataset before training

Use a new create-only acceptance directory. Bind the exact archive and unpacked-file SHA-256 inventory, producer/version identity, acceptance source repository/branch/SHA, runtime and verifier versions. Preserve the archive and original files. Identify authorized train/development/holdout roles from their contracts before reading payloads. Do not open sealed, ANY_READ, or confirmation material to perform the audit; use permitted manifests, hashes, ID/group inventories and existing contamination controls. If a required comparison cannot be completed without forbidden access, record it as blocked rather than claiming zero overlap.

Recompute the following locally, preserving per-check results and failures:

- Archive/file hashes, row counts, schema, stable case IDs and declared split roles; missing, duplicate, malformed and rejected rows.
- Model-visible text and gold separation; literal `UNKNOWN` contract; tokenizer/config/template compatibility; rendered prompt and supervised lengths; complete length distributions; zero silent truncation or dropped overlength rows.
- Answers with independent deterministic or algorithmic solvers, including contradiction, exception and missing-evidence cases. Producer explanations are not an answer oracle.
- Scenario/world/group structural isolation across splits; template and entity/name isolation wherever required; structural or canonicalized duplicates beyond renamed IDs.
- Answerability, answer priors, family, difficulty and evidence-position balances, with actual counts and declared acceptable ranges.
- Question-only answerability classifier; simple surface-feature attacks; majority, copy, first, last and obvious task-specific heuristic baselines. Use the registered audit split; never select a training recipe using protected holdout outcomes.
- Contamination against all quarantined Rineya evaluation material using permitted inventories and controls. Explicitly inventory inaccessible comparisons and whether they block acceptance.
- Overlap against the permanently open MRS-1 scout bank, including canonicalized scenarios/groups and exact/normalized questions. For this paired transfer study, exclude scout items/worlds from training and recipe selection. If overlap cannot be ruled out, repair the dataset before training; do not regenerate the frozen scout.

Freeze binding acceptance thresholds, split semantics, verifier versions and normalization/shortcut definitions before computing pass/fail for a dataset version. Report exploratory checks separately. A failed binding requirement or unresolved required check stops training. Preserve the failure receipt, identify the smallest dataset repair, assign repaired bytes a new version/hash, and rerun the affected checks plus integrity/split/contamination checks. Do not train “to see what happens.”

The acceptance receipt must distinguish PASS, FAIL and BLOCKED and contain the exact requirement inventory, measured values, failure reasons, source/runtime bindings and artifact hashes. No producer assertion substitutes for local recomputation.

## P1 — Freeze the training contract before optimization

Only after P0 passes, create a separate immutable training contract. It must bind all of the following, with concrete values or explicit non-applicability rather than placeholders:

- Source model identity, exact checkpoint SHA-256 and bytes, storage dtype, config/tokenizer/chat-template hashes, serving/output policy and source repository/branch/SHA.
- Accepted dataset manifest SHA-256, exact train and development split IDs/hashes, presentation order, random seeds, and forbidden evaluation populations including the unchanged MRS-1 scout and all protected banks.
- Trainable parameter allowlist and counts; frozen parameters/buffers; objective; target construction; causal alignment; loss masking; supervised-token definition; permitted precision and attention backend.
- Optimizer and every parameter; learning rate; schedule; warmup/decay; clipping; weight decay; microbatch/batch/accumulation; context limit; packing policy; no-truncation behavior; epochs or exact processed-token budget; random/training seeds.
- Exact stopping and acquisition/preservation requirements; development selection rule; checkpoint cadence/retention; restart/resume semantics; source/final hashing; output/export contract; no selection on protected data or final C/D outcomes.
- Retention and acquisition instruments, allowed data roles and thresholds; model loads; host/GPU/disk ceilings; hard wall-time limit; process supervisor; resource accounting; failure/interrupt handling.

Use current evidence to choose the design. If a small exploratory recipe screen is needed, give it a separate identifier, open development population and bounded budget. Freeze the selected recipe before producing the final candidate. Do not revise that recipe after seeing C/D. Any material change creates a new version and retains earlier failures and receipts.

## P2 — Train only after acceptance, contract and authority gates

P0 PASS and a complete P1 contract are necessary, but are not themselves training authority. Check that current user authority covers the exact identified dataset, candidate, recipe, local/remote resource use and execution scope. Do not invent a permit or infer authorization from a fixture receipt. Reuse existing explicit authority when it already covers those facts; request missing authorization only for a concrete reviewed plan when required.

Create a new isolated research branch from the exact inspected parent. Preserve all historical ChatBubble, RRP, Opallo, SupaExperiment, RineyaBeta and RineyaArchitecture heads, weights, evidence and preregistrations. Do not reuse or modify the Phase I execution environment while another process relies on it. Refresh resource measurements immediately before loading and enforce the registered host/GPU/disk and hard-time limits without weakening them. Training must have its own bounded supervised executor; the inference-only MRS runner cannot authorize optimization.

Record every presentation's case/group identity and order, processed and supervised tokens, optimizer updates, learning-rate and loss trajectories, gradient norms, trainable/frozen parameter inventories, checkpoint hashes, source/final hashes, wall time, allocated/reserved GPU memory, host memory, disk usage, and all failure/restart events. Record actual resumed state and avoid presenting a restarted or changed recipe as the original uninterrupted experiment. Preserve raw outputs and failures in create-only directories.

The candidate must meet the prospectively frozen acquisition and preservation requirements on allowed open development instruments before C/D. A failed requirement produces a retained negative result. Do not silently retrain with a changed recipe under the same experiment ID. Freeze the exact final checkpoint and serving assets; qualify any distinct storage export as a separately identified artifact. Loading a BF16 export into FP32 does not recreate lost precision.

## P3 — Ordinary post-training cell C

Register the exact candidate and training provenance in a new create-only `future-registration.json` bound to the original MRS freeze digest. Preserve the fixed existing D1 export as the primary old comparator and Supa as the separately labelled source reference. Perform fresh no-update native qualification under the same guarded workflow and enforce the frozen format-smoke gate.

Run C with the same scout bytes, case assignments/order, A0 prompt, parser, deterministic decoding, template, context/output budgets, scorer and output contract as A. No prompt or parser adaptation is permitted to improve the new candidate. Preserve raw outputs, resource records and exact paired A/C case outcomes. If old A/B have not completed, complete the frozen old cells when authorized resources allow before inspecting final C/D outcomes; never manufacture missing values or choose a different old comparator.

## P4 — Multi-reasoning post-training cell D

Run the identical A3/S2 M1 treatment used for B: the same R0/R1/R2/R3 computations, role prompts, information projections, model selector, candidate budget, maximum calls, token limits, stopping rule and scoring. Retain the same A1/A2/S0/S1 controls; any secondary M2 variant is eligible only if actually frozen in Phase I. Do not add M2 after viewing new-model outcomes. No selector tuning on CHECK or C/D.

Each model-visible prompt remains label-blind. Gold and deterministic oracle answers enter scoring only after generation. Preserve per-item role outputs, malformed outputs, disagreements, selector inputs, raw final answers, costs, B/D and C/D pairings. Model/config/tokenizer/runtime/policy incompatibility must fail the paired contract; a separately named experiment may study it without rewriting MRS-1.

## P5 — Complete paired A/B/C/D analysis

Use the exact complete shared population; never compare a convenient intersection. Missing cells remain missing. Follow the frozen analysis command and masks in FUTURE_2X2_ANALYSIS_PLAN.md and report all 16 Boolean correctness patterns, family/difficulty strata and the exact paired matrix.

| Quantity | Paired expression |
|---|---|
| Training under ordinary inference | C - A |
| Multi-reasoning before training | B - A |
| Multi-reasoning after training | D - C |
| Interaction | (D - C) - (B - A) |

Report all-correct cases, training repairs/damage, pretraining multi repairs/damage, post-training extra repairs/damage, overlapping intervention repairs, and cases where training makes multi-reasoning newly useful, unnecessary or harmful. Event masks may overlap; do not force them into an exclusive partition. Use the registered paired world-cluster intervals and state their descriptive scope. A before/after checkpoint comparison does not isolate every difference in historical training or estimate uncertainty across training seeds that were not run.

## P6 — Recompute complementarity after training

For old and new checkpoints separately, report role-wise correctness, each role pair's both-correct/first-only/second-only/both-wrong counts, answer disagreement, unique candidates, correlated failures, family-specific overlap and the analysis-only oracle ceiling. Report real selector accuracy/utilization, correct-candidate rejection, repair and damage, formatting/nontermination, abstention behavior and actual token/call/latency/memory costs, including cost per recovered case and net recovered case where defined. Zero or negative net recovery must remain visible; do not invent a finite positive cost ratio.

Determine descriptively whether training raises all roles, changes complementary correctness, makes errors more correlated, improves selection/checking, removes a need for extra routes, or makes added computation more useful. The oracle ceiling remains an analysis statistic and is never a deployed treatment score.

## P7 — Apply the prospective decision rules

| Observed pattern | Interpretation and next direction |
|---|---|
| C clearly improves A, D approximately C, and B exceeds A | Training may have absorbed useful scaffold behavior; prioritize ordinary trained inference subject to observed costs/preservation. |
| B exceeds A and D exceeds C by similar amounts | Approximately additive contributions on this open scout; assess combined-system cost. |
| C exceeds A and D-C exceeds B-A | Descriptive complementarity between training and added reasoning computation; inspect role-level causes. |
| B exceeds A but D does not exceed C | Scaffold transfer is unstable; retain regressions and investigate interference. |
| Oracle ceiling rises but the real selector remains weak | Candidate generation has headroom; selection/verification is the next bottleneck. |
| Role errors remain strongly overlapping and A3 resembles A2 | Little evidence that differentiated computations add value over repeated direct attempts. |

“Clearly,” “approximately” and “similar” are descriptive summaries with raw counts and the registered intervals, not newly introduced success or equivalence thresholds. Report ordinary and repeated-direct contrasts, repairs and damage, and measured cost before choosing a direction. Do not promote open-development evidence into confirmation, release readiness or an architecture-level causal claim.

## P8 — Consider only the smallest supported internal mechanism

Do not build a new architecture merely because B or D is positive. Identify which registered computation supplies complementary correct answers or reliable correction. Evidence checking may motivate a compact verifier; alternative-solving gains may motivate query refinement or an alternative low-rank query read; sequential repair may motivate one bounded recurrent pass; selection failures may motivate a small selector; concentrated benefit may motivate a conditional compute gate.

Relate the specific supported operation to the existing query-refinement, A06 controls, recurrence, paired looped flow, matched-MLP, EAM, CED and XSG evidence. Preserve their distinct qualifications and negative results. Do not merge mechanisms because they exist. Any native/internal follow-on needs its own isolated implementation, prospectively frozen controls, scope and authority.

## Completed-study deliverable and stopping boundary

Create `FINAL_MULTI_REASONING_2X2_REPORT.md` only when the actual required stages are complete. Include dataset acceptance receipt; training contract and execution receipt; exact old/new model identities; A/B/C/D results; per-item and per-family matrix; role complementarity and selector performance before/after; repair/damage decomposition; token/call/latency/memory costs; C-A, B-A, D-C and interaction; limitations; internalization decision; and the exact smallest recommended next experiment. Link raw generations, manifests, hashes, resource/supervisor receipts and retained failures.

Until then report each unfinished stage explicitly as NOT_RUN, FAIL or BLOCKED with its smallest next action. Keep SOFTWARE PASS, NATIVE EXECUTION PASS, OPEN DEV RESULT, MECHANISM SCOUT RESULT, CONFIRMATORY CLAIM and RELEASE CLAIM separate. The final two remain false unless a separately authorized protected confirmation/release process actually earns them. Dataset arrival alone does not change those statuses or authorize automatic training.
