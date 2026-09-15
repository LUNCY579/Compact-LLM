# RRP-001 results

**Status: the native instrument is qualified and a bounded engineering microfit met its
pre-registered criterion. This is NOT a reasoning-improvement result and NOT a quality claim.**

Every number below is generated from a receipt under
`rineya-reasoning-posttraining-records/rrp001-journal/runs/`. Nothing is retyped by hand.

## What this does and does not establish

The microfit trained on 32 TRAIN worlds and was then scored on **those same 32 worlds**.
A rise from 8/32 to 31/32 therefore measures fitting, not generalization. Per the
programme's own rule, microfit success is engineering evidence only: it shows the
optimizer, label masking, EOS handling, generation path, scorer and budget accounting all
work end to end on real weights, and that these worlds are learnable at this scale.
It says nothing about held-out reasoning, preservation, or transfer.

- `quality_improvement_established`: **false**
- `preservation_or_generalization_claim`: **false**
- Human data review: **OPERATOR-ATTESTED, not per-world reviewed** (`OPERATOR_DATA_APPROVAL.json`). It covers the 85-world packet including all 32
  microfit worlds; it does NOT cover the 384-world DEV bank, which is
  machine-verified instead. See CL-029.
- DEV, preservation/abstention DEV and fresh confirmation banks: **not yet built**

## Zero-update native qualification (`qualify-001`)

- Unique parameters counted natively: **596,049,920**
- Trainable: **314,618,880** in **220** tensors (blocks 8-27)
- Source state before -> after: `6e8d322963ca6c0a...` -> `6e8d322963ca6c0a...` (IDENTICAL)
- Optimizer steps: **0**
- Cached vs uncached logits, max abs difference: **1.125e-04**
- Approved training targets reproduced by the native tokenizer: **32/32**
- Longest qualified training shape: **2048 tokens**
- Gradient probe `native_chat` at 31 tokens: 220/220 finite, 220/220 nonzero
- Gradient probe `synthetic_2048_shape_on_real_source` at 2048 tokens: 220/220 finite, 220/220 nonzero
- Checks passed: `native_hidden_logits`, `cached_logits`, `causality`, `response_eos_masks`, `loss_value`, `finite_gradient_participation`, `frozen_weights`, `tied_storage`, `shape_2048`

## Microfit (`train-001`)

- Updates: **105/105**, complete: **True**
- Processed tokens: **167,256**, supervised tokens: **1,184**
- Frozen source weights unchanged: **4e3c74c44bc99cdc...** (engine verified equality)

### Fixed 32-world TRAIN probe

| probe | canonical correct | strict text match | response CE | terminated | truncated | literal UNKNOWN |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (source) | **8/32** | 8/32 | 1.49630 | 32/32 | 0/32 | 12 |
| midpoint | **29/32** | 29/32 | 0.08554 | 32/32 | 0/32 | 5 |
| final | **31/32** | 31/32 | 0.01224 | 32/32 | 0/32 | 6 |

There are exactly **6** UNKNOWN references. The source emitted 12 literal UNKNOWNs and
the fitted candidate emitted 6: the source over-abstained on computable items, and the
fitted model matched the true abstention rate. On TRAIN rows only, so this is not an
abstention generalization result.

### Per family (n=4 each)

| family | source | midpoint | candidate final | resumed final |
| --- | --- | --- | --- | --- |
| deduction | 1/4 | 2/4 | 4/4 | 4/4 |
| dialogue_memory | 0/4 | 4/4 | 4/4 | 4/4 |
| document_reconciliation | 3/4 | 4/4 | 4/4 | 4/4 |
| program_trace | 0/4 | 4/4 | 4/4 | 4/4 |
| relational_join | 0/4 | 4/4 | 4/4 | 4/4 |
| signed_ledger | 1/4 | 3/4 | 3/4 | 3/4 |
| source_boundaries | 1/4 | 4/4 | 4/4 | 4/4 |
| story_state | 2/4 | 4/4 | 4/4 | 4/4 |

Prospective criterion was **>= 31/32** canonical training correctness with finite gradients
and unchanged frozen weights. Measured **31/32**: **MET**.

Not fitted after 8 exposures:
- `signed_ledger`: predicted `211`, reference `145` (terminated, not truncated) - an
  arithmetic/execution error, not a format failure.

## Fresh-process resume verification (`resume-001`)

- Resumed from `checkpoint-00002` in a new process with a fresh source load
- Updates this attempt: **103** of 105, reprocessed tokens: **164,528**
- Candidate final parameters: `e8c72eb4542fa4bfddd7ac03...`
- Resumed  final parameters: `e8c72eb4542fa4bfddd7ac03...`
- **Parameters match: True**
- **Predictions match: False**
- Resumed final canonical correct: **31/32**
- Comparator verdict: **PASS**
- Independent checks run: 15, all passing: True

A bit-identical resumed run is the save/resume equivalence and determinism evidence the
plan asked for. It also means the recorded step log is reproducible, not a one-off trace.

## Measured cost

| lane | spent | approved ceiling |
| --- | --- | --- |
| `adaptive_model_selection_rounds` | 1 | 1 |
| `cloud_or_teacher_spend` | 0 | 0 |
| `evaluation_and_qualification_model_calls` | 130 | 160 |
| `generated_tokens_all_candidates` | 16,384 | 20,480 |
| `model_worker_launches` | 3 | 4 |
| `model_worker_wall_seconds_and_conservative_gpu_seconds` | 1,067 | 2,400 |
| `new_disk_bytes` | 11,342,008,320 | 17,179,869,184 |
| `on_policy_rl_rollout_tokens` | 0 | 0 |
| `processed_training_tokens_all_attempts` | 333,987 | 400,000 |
| `retries` | 0 | 1 |
| `total_owned_work_wall_seconds` | 1,068 | 3,600 |

- Budget breached: **False**
- Three supervised launches, three terminal supervisor receipts, zero unsettled reservations.
- Per-update wall time held at a ~3.7 s median; the projected O(n^2) journal-replay cost
  did not materialise at this scale (see `CLAIM_LEDGER.json` CL-010).

## What comes next

The instrument is now built and the research question is measured but not yet moved. Of the
five prerequisites recorded here originally, three are done, one is done-with-a-caveat, and one
is structurally impossible:

1. ~~A repeatedly usable DEV bank and a preservation/abstention DEV~~ — **both built.** Three
   banks now exist, each with its own positive control at 100% and its own shortcut floor:

   | bank | rows | floor | domain |
   | --- | --- | --- | --- |
   | `RRP2-DEV-V1` | 384 | 0.1458 | Forge v1, plain-string answers |
   | `RRP2-DEV-V2` | 336 | 0.2857 | Forge v2, strict-JSON answers |
   | `RRP2-PRES-DEV` | 768 | 0.4375 | non-Forge ordinary language + abstention |

   Each accuracy is meaningful **only against its own floor** and never against another bank's.
   A genuinely fresh confirmation bank remains **NOT_ESTABLISHED** and is not achievable here
   (CL-017), bounded precisely by CL-035: the one sealed unconsumed surface on this machine is
   3,960 `residual_v23 fresh_sealed` rows, and they belong to another programme's
   pre-registration, so RRP-002 registers them `ANY_READ` and declines them.
2. ~~UNKNOWN and IMPOSSIBLE strata~~ — **present.** v1 carries a five-way sentinel split, with
   every non-UNKNOWN stratum confounded with a single family (CL-019); v2 carries a structured
   three-way `reference.status` that spans families, which is why the stratified abstention
   analysis belongs on v2 with v1 as the cross-grammar replication. `RRP2-PRES-DEV` adds a
   two-sentence abstention contract and separates missing-evidence from source conflict (CL-037).
3. ~~Shortcut baselines and a positive-control exact solver~~ — **both built and load-bearing.**
   Every bank reports its floor beside any model number, and two of my own baselines were caught
   reading the reference and quarantined (CL-031, and the answer-agnostic test in the
   preservation lane). The v2 verifier was adversarially audited at 4,390 attacks for 0 credit
   leaks (CL-021).
4. ~~Operator sign-off~~ — the operator has attested the data (CL-029). The attestation is an
   attestation, not a per-world review, and it does **not** reach any of the three DEV banks
   per-world; all three rest on machine verification and are labelled
   `MACHINE_VERIFIED_NOT_HUMAN_REVIEWED`.
5. **Still outstanding, and now the single blocking prerequisite:** the cluster-robust paired
   statistics layer, plus predeclared contrasts, family weights, cluster units and practical
   margins. CL-038 constrains it with a measured fact rather than a preference — sub-bank (b) of
   `RRP2-PRES-DEV` has **16 effective units for 192 rows**, so every row of that bank carries a
   `cluster_id` the statistics layer must group on. A registered successor experiment with its
   own numeric envelope is also still required: this grant is scoped to `RRP-001-MICROFIT` and
   its single adaptive round is spent.

The source-model error decomposition from the baseline probe gives the first bottleneck
candidates: abstention miscalibration, response-contract violation (prose instead of
`key=value`), and genuine execution/binding errors. Those are three different mechanisms
and should not be chased with one undifferentiated data mixture.
