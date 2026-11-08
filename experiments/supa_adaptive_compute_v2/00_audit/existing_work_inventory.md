# Existing work inventory — audited before implementation

Audit date: 2026-09-05. Repository `https://github.com/LUNCY579/ChatBubble.git`.
Paths below are repository-relative. No sealed dataset contents were opened.
The audit covered source, callers, development manifests, historical reports/tests,
and branch trees; it did not execute every old experiment or inspect every corpus row.

## Provenance

| Branch / state | Exact commit | Scope |
|---|---|---|
| original checkout `codex/supa-qwen4b-single-layer-graft` | `d8475ea3b9de3e17c5193a00ed01cc4aa6f1a136` | Native CPU replication; single-layer graft is untracked local work |
| `research/supa-a1.1-anchor-routed-middle-recurrence-v1`, local and live origin | `d8475ea3b9de3e17c5193a00ed01cc4aa6f1a136` | Native historical evidence |
| `research/rineya-cross-scale-qwen4b-8b`, local and live origin | `fde553c908db39e87a15a52ed239d1ba0a68bfd3` | Larger lab; its files have no differences from current checkout |
| `research/rineya-cross-scale-capability-transfer` | `8942d4d8936dff0e0fdbd6732bf22e12c70b92d2` | Earlier cross-scale completion |
| `research/supa-a1.1-fast-weight-attention-v1` | `d0b417632930170a08d471be2629ed44d76f225c` | SCIENTIFICALLY DIFFERENT; do not import attention mechanisms |
| `research/positional-sequential-attention` | `ee13448f11efaa7f14206dab95b277816b912ce6` | SCIENTIFICALLY DIFFERENT; do not merge |
| live remote symbolic HEAD (`origin/main`) | `a6ee824584a9e28881826ab92a11db48e124d90c` | Queried with `git ls-remote`; not the research base |
| new `codex/supa-adaptive-compute-v2` | starts at `d8475ea3b9de3e17c5193a00ed01cc4aa6f1a136` | New work only; no remote branch at audit |

Untracked `experiments/cross_scale_layer_graft_v1/` has **no source commit**.
Its manifest records the base HEAD and source hashes; claiming its implementation
is committed at that HEAD would be incorrect. Preserve its full byte inventory.

## Component decisions

`A` below means `experiments/anchor_routed_middle_recurrence_v1`;
`G` means `experiments/cross_scale_layer_graft_v1`;
`L` means `Rineya_Cross_Scale_Capability_Transfer_Lab_v1`.

| Exact path (with prefixes above) | Classification | Purpose / quality / reuse decision |
|---|---|---|
| `A/src/armr/hooks.py` | REUSE AS-IS | Scoped native repeated forward, frozen inference, cache rejection, original positions and restoration. Introduced `fb7e220a025b09577a79ea60b87a7cbf404983a0`. Reuse in audited CPU FP32 eager tensor-returning Qwen3 runtime. |
| `A/src/armr/recurrence.py` | REUSE AS-IS | Correct L12,L13,L12,L13 block order, bounded middle only; `251bb463054284e4dd03d62ec75bee5e24b5bc69`. |
| `A/scripts/run_experiment.py` | REUSE WITH REPAIR | Reuse local canonical checkpoint resolution; do not use its old dataset caller/new_run directly. Reapplication screen preserved. |
| `A/scripts/run_fixed_recurrence.py` | HISTORICAL EVIDENCE ONLY | A0/A1/A3 and old smoke selection logic. Tiny development gates do not license routing in v2. |
| `A/src/armr/evaluation.py` | REUSE WITH REPAIR | Greedy no-cache generation, NLL and latency reference. Deadline checked too late; EOS can bypass deadline and NLL is unbounded. V2 must check before/after every forward, reject safety stops, share strict scorer. |
| `A/src/armr/metrics.py` | REUSE WITH REPAIR | Reuse logit/hidden metrics and Unicode/repetition diagnostics; script assumptions restricted to explicitly English bank. |
| `A/src/armr/provenance.py` | REUSE WITH REPAIR | Reuse exclusive JSON, state/file hashing and checkpoint list. Fixed two-file data allowlist SUPERSEDED. New manifests need transitive sources, per-arm directories and shard contracts. |
| `A/src/armr/preregistration.py` | HISTORICAL EVIDENCE ONLY | Exact old budget-only freeze. Preserve source hashes and frozen scorer. |
| `A/scripts/audit_budget_replication.py`, `audit_fixed_results.py`, `audit_completed_screen.py` | REUSE WITH REPAIR | Reuse audit principles: uncensored baseline, paired coverage, raw evidence reconciliation; adapt new schema without modifying these files. |
| `A/evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z` | HISTORICAL EVIDENCE ONLY | A0 12/16, A1 12/16, A3 13/16; one paired win. `af7c49d432a652c55cbafc5155c378d87497cf24` preregistration; completed at base HEAD. |
| `A/evidence/02d_cpu_budget_v2_audit/20260905T135230_002601Z` | HISTORICAL EVIDENCE ONLY | Independent replication and identity audit. Preserve intact. |
| `A/evidence/01c_completed_candidate_atlas/20260905T042420_466033Z` | HISTORICAL EVIDENCE ONLY | L16 uniquely passed both two and three applications: cosine .9787/.9430, KL .0806/.3046. Prospective rationale for A16, not efficacy. |
| `A/tests/` | REUSE AS-IS | Mechanics, ordering, disabled/restored identity and historic receipt checks. Rerun with qualified runtime; do not rewrite expected historical outcomes. |
| `G/graft.py` | REUSE WITH REPAIR | Reuse streamed actual HF layer loading/capture, masks, residual expert, zero gate, affine-bias cancellation, ridge baseline and random controls. Extend through v2 modules; leave original untouched. Original source SHA256 `56f9c609ce43ece7579adba4fea689bdfffe3d8dca4ec7c970d1cfd1a64bf57d`. |
| `G/run.py`, `G/protocol.json` | HISTORICAL EVIDENCE ONLY | Original fixed L12→4B L16 absolute bridges, gate .1; not a v2 selection caller. No 1000-case rerun of this mechanism. |
| `G/diagnose_alignment.py` | REUSE WITH REPAIR | Token-category and delta diagnostics; post-hoc evidence must remain labeled post-hoc. |
| `G/audit.py`, `G/tests/` | REUSE WITH REPAIR | Reuse controls/identity/capture test concepts and independent scoring audit; extend prospectively. |
| `G/evidence/20260905T141942_899240Z`, `G/evidence/audit_20260905T150001_276474Z` | HISTORICAL EVIDENCE ONLY | G/A0 12/16, identical 16 greedy outputs; donor delta cosine .443202525, relative MSE 3.65759110; ordinary text up-map error 7.4157. Mechanical viability, no useful transfer established. |
| `L/src/rineya_cross_scale/streaming_forward.py` | REUSE WITH REPAIR | Reuse StreamingWeights storage reader; use actual installed HF decoder for transformation, not handwritten forward. |
| `L/src/rineya_cross_scale/checkpoint.py` | REUSE AS-IS (reader only) | Validated shard reader/discovery. Weight-writing helpers out of scope. |
| `L/cross_scale_evidence/02_architecture/qwen4b.json` | REUSE AS-IS | Pinned donor `1cfa9a7208912126459214e8b04321603b3df60c`, shard/config hashes. Use actual head_dim=128, not legacy inferred 80. |
| `L/src/rineya_cross_scale/hashing.py` | REUSE AS-IS (hashes) | Canonical JSON and file hashes. Atomic replacement writers UNSAFE for immutable evidence; use exclusive creation. |
| `L/src/rineya_cross_scale/evidence.py` | REUSE WITH REPAIR | Manifest pattern useful; existing replace writer and narrow private subtree skip insufficient for new immutable/sealed boundary. |
| `L/src/rineya_cross_scale/leakage.py` | REUSE WITH REPAIR | Existing guard covers only named TEST2/private patterns. V2 must reject all sealed/test2 paths and allowlist frozen generated bank roles, resolving symlinks first. |
| `L/src/rineya_cross_scale/evaluation/scoring.py` | REUSE WITH REPAIR | Exact/JSON normalization reusable; numeric-last-token and keyword/choice matches UNSAFE for strict efficacy because extra contradictory answers can pass. |
| `L/data/evaluation/calibration.jsonl`, `mergedev_fast.jsonl`, `supa_layer_ablation_smoke.jsonl` | HISTORICAL EVIDENCE ONLY | Previously inspected data; cannot become fresh confirmation. No bridge use of native efficacy labels. |
| `L/scripts/v3_e6_capture_token_states.py` | UNSAFE / INCORRECT for v2 | Wrong ordinary Supa loading path (`trust_remote_code=False`), possible normalized terminal state, insufficient sealed guard and scenario split. |
| `L/src/rineya_cross_scale/functional_translator.py` | SCIENTIFICALLY DIFFERENT / REUSE WITH REPAIR | Training-fit R² called held_out_proxy_r2 cannot be used as held-out evidence. Representation maps alone do not optimize actual donor delta. |
| `L/src/rineya_cross_scale/{projection,aligned_projection,spectral_projection,direction_transfer,v3_transport}.py` | SCIENTIFICALLY DIFFERENT | Weight/representation transport and earlier objectives. Retain as historical comparison; no automatic mechanism import. |
| `rineya_significance_native_architecture_lab/src/rineya_significance_lab/evaluation/statistics.py` | REUSE WITH REPAIR | Exact paired McNemar/bootstrap primitives; commit `eb8e433f68de929ed9c5eefc97e5b16b3bf261d6`. Extend ID/coverage validation, family summaries, multiplicity annotations. |
| `rineya_significance_native_architecture_lab/src/rineya_significance_lab/evaluation/clustered.py` | REUSE WITH REPAIR | Scenario cluster statistics and power planning at same commit. Preserve scenario dependence; use preregistered primary originals where possible. |
| New independent banks, v2 two callers, bridge-delta optimization, strict frozen registrations, immutable shard aggregation | MISSING | Build only these missing/repair layers around existing tested primitives. |

## Shared implementation plan

New root `experiments/supa_adaptive_compute_v2`; thin import adapter to preserved
legacy helpers. Share hashes, guards, model loading, capture, generation,
scoring, statistics, manifests and reporting. Native owns operators and large
evaluation caller. Cross-scale owns bridge objectives and calibration-only
depth-pair caller. Each track gets separate generated role partitions, protocol,
selection state, run manifests and evidence roots. No destructive reorganization.

Every immutable evidence run must bind inputs/config/scorer/source hashes,
checkpoint identities, git commit, runtime/threads/device/cache, timing/memory,
status and failure. Resumption creates new exclusive shard attempts and only
aggregates complete compatible receipts; no concurrent writers per directory.

Available machine: Ryzen 9 7940HX, 16 cores / 32 logical CPUs, about 15.2 GiB
visible RAM; only about 3.2 GiB free at audit. Use historical environment
`C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe`
(historical torch 2.12.1+cu126, transformers 5.14.1), CPU FP32/eager/no-cache.
The PATH Python has different versions and is not automatically qualified.
Schedule heavy inference sequentially. Large model runs may be deferred with
deterministic resumable commands under the session's explicit compute allowance.

## Untouched boundary

All old evidence, old source, old manifests/scorers and existing untracked graft
work remain untouched. Canonical and donor checkpoints never saved. Sealed
data, 8B execution, routing, MLA/MHA/MTP, PEFT, backbone training and distillation
remain closed. Any new scientific changes after confirmation require a new
protocol and fresh held-back data, never a rewritten result.
