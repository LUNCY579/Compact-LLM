# Cross-scale composable skills: prior-work audit

Audit date: 2026-09-20. Study: CROSS-SCALE-COMPOSABLE-001.

This is a source and historical-record audit, not a new model experiment. It was performed before implementing the successor mechanics. Existing refs were inspected with `git branch`, `git log`, `git ls-tree`, `git grep`, and `git show`; none was checked out, moved, or modified. Reads in the successor checkout used the inherited parent files. This audit changes only this new document.

Successor: `research/cross-scale-orthogonal-composable-skills-v1`, at
`C:/Users/clclc/BusinessProject/ChatBubble-cross-scale-composable-v1`.
Exact parent: `codex/dataset-retention-audit-v1@07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd`.

The useful distinction is between four different objects: a cross-scale architecture, a map between frozen representations, a transported **learned skill residual**, and a student delta whose behavioral effects compose. Prior sources implement substantial parts of the first two and substantial small-model delta machinery. They do not establish the full latter two-part claim required here.

## Scope and evidence admission

- No pretrained checkpoint was loaded or executed for this audit; no optimization, download, native qualification, or grant consumption occurred.
- No sealed behavioral prediction/score file or sealed evaluation examples were opened. Existing published README/report/agenda summaries are cited as **reported historical conclusions**, not reanalysed arm comparisons. Their statements are not newly verified scientific findings.
- Historical test receipts below describe tests run by the prior work. This audit did not rerun those branches. Current successor tests must have their own source-bound receipt.
- Source presence is directly observed at the exact Git commits below. A referenced local weight/cache/vault file is not thereby observed, intact, qualified, or available now. Names such as “skill,” “transfer,” “graft,” and “orthogonal” are not evidence of the mechanism.
- The new study's comparability gate must precede endpoint reading: views, implementation, dataset, tokenizer/model, base, delta and probe-weight digests; exact training rows; and applicable parent disjointness. A failed or unresolved binding remains `BLOCKED_ON_COMPARABILITY`.
- Historical acquisition floors vary. In particular, old DMRF code still contains a seven-row cross-draw spread. This successor must follow the user's current `|net| >= 20` cross-training-draw floor, report `(net, nd, exact p)`, and preserve the separate paired treatment of within-checkpoint deterministic transforms. Do not import historical floor constants without review.

## Ref inventory used

The names are navigation labels; the full commits are the evidence identities.

| Ref inspected | Exact commit | Principal scope |
| --- | --- | --- |
| `codex/dataset-retention-audit-v1` | `07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd` | Successor parent; MQ/RRP, retention/acquisition, inherited CED |
| `research/ced-cross-scale-e28-d36-v1` | `d82e27c48b2a75b3a9459148990127cd9b14a2e9` | XSC-001 E28/D36 causal-memory architecture |
| `research/rineya-cross-scale-capability-transfer` | `8942d4d8936dff0e0fdbd6732bf22e12c70b92d2` | Initial projected-weight transfer lab |
| `research/rineya-cross-scale-qwen4b-8b` | `fde553c908db39e87a15a52ed239d1ba0a68bfd3` | Follow-up transport diagnostics, E6, v4 design |
| `origin/research/cross-scale-stitch-graft-framework-v1` | `3059e576555afbeb735d79e129435c12caaf4359` | XSG-001 guarded capture/bridge/graft/stitch mechanics |
| `single-layer-fine-tuning` / `multi-reasoning-slft` | `954a0c2637e691b493c5ac3d488653a8c1adf675` | Titan-Sparse publication and amendments |
| `research/delta-multi-reasoning-factorial-v1` | `7f471b77116b35b52bae9091e3a96b2fe2287626` | Delta, composition, magnitude and direction diagnostics |
| `origin/research/dmrf-source-observability-v1` | `3c76199fb28fb55f0022aa25d93cf876a566f5db` | OBS-001 source/claim admission guard |
| `origin/research/sequential-complementary-skills-v1` | `3ad2769a9f8343a9697e1720fcdc9efa18c3845a` | SCP-001 sparse/rank controls, F/P/T preparation, analysis |
| `origin/research/sequential-complementary-native-v1` | `b4f010619290a562c54cdaf3bb6edd4eb9112a91` | Native integration snapshot, identified through successor boundary |
| `origin/research/sequential-complementary-native-boundary-v1` | `9551d7ba50784a5acd19e8512faaba6d0efc6aca` | Locked native integration review boundary |
| `origin/research/scp-native-qualification-boundary-v1` | `7486a1490517e9bd0ed8e29631a601d713ec20e8` | Separate offline qualification-boundary successor |
| `research/retention-dose-parameterization-v1` | `f254acae6ea6d280333b268373ca2585e6e6e583` | Scalar-dose study and pending high-dose LoRA code |
| `codex/supa-qwen4b-single-layer-graft` | `d8475ea3b9de3e17c5193a00ed01cc4aa6f1a136` | Branch-name check; head actually names ARMR recurrence work |

The inspected local/remote ref inventory contains no separately named orthogonal-skill branch. Relevant orthogonality evidence was found in DMRF, the capability-transfer follow-up, and sequential-skill sources. This is a statement about the inspected refs and paths, not proof of absence from all history, external vaults, or unpublished work. The graft-named local ref is a useful caution: its last commits concern anchor-routed middle recurrence, and the tree-name query returned no graft-specific paths. XSG-001 is the directly located stitch/graft implementation reference.

## 1. Cross-scale CED is an architectural experiment

Evidence: at `d82e27c48b2a75b3a9459148990127cd9b14a2e9`,
`experiments/cross_scale_ced_v1/{README.md,model.py,config.py,checks.py,source.py,checkpoint.py,protocol.json,sources.json,verification/LOCAL_VERIFICATION.json}`.

**Implemented.** A complete 28-layer, width-1024 Supa encoder and complete 36-layer, width-2560 Qwen3-4B decoder are joined by a memory writer and post-block memory reads. It retains both ordinary backbones, initializes read-output projections at zero, uses an aligned causal token grid, and provides zero/bypass identity, causality, cache, gradient, checkpoint and frozen-backbone checks. H/P are paired architectural controls; P is still contextual. The source identifies Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.

**Evidence.** The prior receipt says `LOCAL_MECHANICS_PASS_ONLY`, 132 passed and two native random-Qwen tests skipped because Transformers was absent. It includes tiny random-stack checks and full-shape inventories on meta; it explicitly excludes official source execution, P1/P2, efficacy, GPU fit and deployment. The complete hybrid is analytically 5,375,687,808 parameters with 757,169,792 trainable interface parameters. This is far outside the present small-model delivery budget.

**Reusable.** Exact architecture inventories; the distinction between residual width and Q/KV head geometry; tensor allowlists and source locks; causal/cache/frozen-weight checks; zero-output initialization; separation of engineering and efficacy receipts. Reuse contracts and tests as references, not the full hybrid as the deliverable.

**Question and limit.** It asks whether two native-width backbones can be coupled and qualified through causal memory. Architectural compatibility, an active interface, or passing random fixtures does not show that a learned 4B skill can be transported into a standalone 596M student.

## 2. Initial cross-scale projected-weight work and its corrections

Evidence: `Rineya_Cross_Scale_Capability_Transfer_Lab_v1/` at
`8942d4d8936dff0e0fdbd6732bf22e12c70b92d2`, with follow-ups at
`fde553c908db39e87a15a52ed239d1ba0a68bfd3`.
Relevant files include `README.md`, `src/rineya_cross_scale/{architecture.py,projection.py,aligned_projection.py,spectral_projection.py,direction_transfer.py,operations.py,checkpoint.py,diagnostics.py,delta_analysis.py}`, `scripts/{build_cross_scale_delta.py,merge_task_vector.py,merge_ties.py,merge_dare.py,merge_consensus.py}`, and the v3 report / v4 amendment under `cross_scale_evidence/`.

**Implemented.** Architecture/depth audits, projection and tensor correspondence, streamed checkpoint reads/writes, projected-donor delta extraction, task-vector and interpolation operations, TIES/DARE/Fisher/consensus variants, singular-profile and truncated-direction transfers, diagnostics, source fingerprints, and several controls. Crucially, `operations.build_delta` registers its equation as `projected_donor - official_qwen3_0.6b`. That object is not `large_skilled - large_base` for an independently trained skill.

**Reported evidence.** The README reports no confirmatory GO and an unopened sealed bank. The v3 report explains that P4 retained the target's singular vectors and transported donor singular-value profiles: its cross-donor agreement statistic was reproduced by shape-matched Gaussian donors and its earlier interpretation was withdrawn. This is a valuable null-control lesson; it does not falsify all cross-scale transfer. The v3 report also describes candidate functional near-identity. These are historical published conclusions, not fresh comparisons admitted here.

The v4 amendment explicitly corrects the historical target to **Supa-A1.1**, rather than Supa Core 1.7. It proposes fitted aligned-weight transport, ordinary-target, norm-matched random-map and shuffled-depth controls; it does not supply a qualified learned donor-skill pair for this successor. Its decision to retire 4B in favor of 8B belongs to its old target, precision and development-bank context. It is not a general claim that a 4B donor cannot contribute to the new registered skills.

**Reusable.** Shape/inventory and one-tensor-at-a-time I/O ideas; actual stored-delta-versus-intended-delta diagnostics; source/provenance vocabulary; random donor and shuffled correspondence controls; the `direction_transfer.project_weight` dimensional contract for any future secondary weight-space mechanism.

**Do not reuse blindly.** Historical operations contain overwrite/staging-cleanup options and gate-neutralization behavior specific to that study. The new namespace needs create-only evidence and exact source policy. The spectral-profile operator cannot be relabelled as directional skill transfer. Raw weight rebasing is a secondary mechanism here; residual displacement is primary.

## 3. Prior functional representation transport already exists

Evidence at `fde553c908db39e87a15a52ed239d1ba0a68bfd3`:
`scripts/v3_e6_capture_token_states.py`, `scripts/v3_e6_fit_translator.py`,
`scripts/v3_e6b_incremental_alpha_sweep.py`, `scripts/v4_fit_basis_maps.py`,
`src/rineya_cross_scale/streaming_forward.py`, and
`cross_scale_evidence/14_v4_aligned_transport/protocol/amendment_v4_aligned_transport.md`.
The introducing tooling commit is `6003c12f93961a33c6ecba7b22029b04ab2fdcbc`; the follow-up result commit is `60bc65a1821d9fb4a02d5383bed842d720e313d3`.

**Implemented.** Per-token activation capture, prompt-disjoint train/check splitting, train-only standardization, ridge and rectangular Procrustes fits, low-rank variants, token-shuffled donor controls, and an incremental test comparing prediction from the small base alone with prediction from small base plus donor. A width-matched random-feature control tests whether merely adding columns helps. The code fits maps of ordinary representation states, not isolated learned-skill residuals.

**Evidence and interpretation.** Prior sources report measurable representation predictability, followed by a donor-redundancy concern under the incremental control. The earlier v3 report's “E5/E6 blocked” row is not the latest code state: later E6 tooling and artifacts exist at the pinned follow-up commit. Conversely, existence of a later translator script is not evidence that a student adapter acquired a skill. This audit does not re-open its cached outputs or claim fresh comparability with the new Supa 1.7 study.

**Reusable.** Prompt/group holdouts, train-only normalization, wrong-pair controls, the incremental-predictability question, sequential capture, and source-visible capture positions. A simple linear transport map and normalized-depth mapping are therefore **not novel** by themselves. The new question is whether a frozen map applied to `h_large_skilled - h_large_base` supplies a useful training target beyond the ordinary student's signal, surviving magnitude/rank/budget controls and later composition.

## 4. XSG-001 is the strongest guarded bridge reference

Evidence at `3059e576555afbeb735d79e129435c12caaf4359`:
`experiments/cross_scale_stitch_graft_v1/{README.md,alignment.py,contracts.py,artifacts.py,modules.py,planning.py,evaluation.py,docs/DESIGN.md,docs/NATIVE_CONTRACT.md,evidence/LOCAL_VERIFICATION.json}`.

**Implemented.** Exact identity and token-position capture contracts; FIT/MAP_CHECK split and group isolation; CPU FP64 affine ridge with streaming centered sufficient statistics; finite and workspace guards; immutable safetensors capture/bridge artifacts; bidirectional stitching; one-FFN graft; trainable-parameter-matched ordinary adapter; paired evaluation; a supervised no-skip CPU qualifier. The optional adapter targets already loaded frozen stock Qwen3 under Transformers 4.51.3 and refuses the custom Supa-A1.1 policy.

The important APIs are:

| API | Mechanical value for this successor |
| --- | --- |
| `contracts.Identity`, `Site`, `RowKey`, `Capture` | Model/runtime/template/precision/site/token-row identity |
| `contracts.paired`, `disjoint`, `capture_digest` | Exact pair alignment, split isolation, hashed capture semantics |
| `alignment.RidgeMoments.update/solve` | FP64 centered streaming ridge; declared workspace bound; no sample-by-sample Gram allocation |
| `alignment.AffineFit.delta(change)` | Applies **only the linear part** to a residual displacement; affine bias must cancel |
| `alignment.fit_bridge` | Held-out map checks with identity/site consistency and digest receipt |
| `artifacts.save_capture/load_capture`, `save_bridge/load_bridge` | Create-only typed artifacts and payload/hash verification |
| `planning.graft_budget/stitch_budget` | Honest distinction between trainable interface size and deployed model size |

**Evidence.** Its prior local receipt records 144 tests, zero skips/failures/errors, four tiny fixture optimizer updates per demo, zero native calls and `PASS_MECHANICS_ONLY`. Native parity is false. This is directly applicable engineering precedent, not efficacy evidence.

**Question and limit.** Stitching asks whether a retained prefix and suffix communicate through a bridge. Grafting adds retained donor computation. The README gives about 2.584B parameters for a 4B midpoint stitch and about 676M for the one-FFN graft. Both violate this task's intended standalone nominal-596M deliverable. Reuse transport contracts/math without retaining large donor modules in the deployed student.

## 5. Titan, sparse deltas, task vectors and parameter geometry

Evidence at `954a0c2637e691b493c5ac3d488653a8c1adf675`:
`Rineya_Published_Cores/Rineya-Titan-{Experimental,Sparse-Experimental}-1.0.0-devfreeze/{README.md,CORE.json,AMENDMENT_001.json}` and the Sparse core's
`evidence/{build_sparse.py,SPARSE001_PREREG.md,REP001_PREREG.md}`.
The inherited parent also contains `Rineya_Published_Cores/TECHNIQUES_DELTA_CORES.md`.

**Implemented.** A source-relative learned update, FP32 sparse coordinate masking, deterministic magnitude/tie selection, published source/update identity and replication amendments. The relevant operation is `W0 + M_0.25(delta)`, retaining coordinates per tensor, not `W0 + 0.25 * delta`. FP32-trained updates and a promoted source are important: low-precision export is a distinct intervention that can erase part of a small update.

**Reported evidence and boundary.** Public amendments describe limitations in earlier precision and replication claims. Public technique summaries describe compressibility, scalar/magnitude alternatives, retention variability and failed naive composition. These summaries establish why the new study needs direction-specific controls; they do not establish independently composable skills. Neither sparse storage nor low rank is a new acquisition mechanism.

**Reusable.** Exact delta extraction arithmetic, stable coordinate masking for optional controls, source/trained allowlists, materialization receipts, actual serialized tensor identity, and a reminder to measure effective dense deltas after merge. Do not execute the historical builder's absolute paths or overwrite its published core.

## 6. DMRF composition and norm controls constrain the new claims

Evidence at `7f471b77116b35b52bae9091e3a96b2fe2287626`:
`experiments/delta_multi_reasoning_v1/{DMRF001_AUDIT.md,AGENDA.md,COMBINED_RECORD_20260918.md,analysis/paired.py,analysis/norm_audit.py,analysis/delta_norms.py,audit/measure_delta.py,audit/spectral_gaps.py,build/build_n16.py,analysis/build_scaled_arm.py,analysis/COMPOSITION_PAIRED_20260920.md}`.

**Implemented.** Factorial reasoning/weight analysis, low-rank and norm-scaled constructions, tensor/delta norm audits, subspace diagnostics, exact paired repairs/damage, complete-population checks, and interpretation of deterministic transforms versus independent training draws. Direction/orthogonality appears here as a diagnostic; it must not be inferred from a branch name.

**Reported evidence and caution.** The public agenda states that nearly orthogonal weight vectors still interfered behaviorally. It also corrects a tempting interpretation of total-norm matching: matching the composed norm attenuated **each component skill** relative to its solo setting. Thus “same total norm” does not mean “same strength of A and B.” The public paired-composition note says the old composition weights were reclaimed; the presence of a report is not proof that those exact artifacts can now be re-probed.

**Reusable.** Separately report raw-vector cosine, module/layer-weighted cosine, singular-subspace overlap, component norms, composed norm, and each component's scaling. Use a norm-matched baseline that answers the intended question rather than averaging away part of each skill. Carry acquisition and retention as paired endpoints with repairs/damage and cluster structure, not just totals. A nonsignificant regression test is not a losslessness test.

**Limit.** Existing scalar, masking and rank transforms of one trained update are post-hoc transformations. They are valid controls, not the new cross-scale mechanism. Historical near-orthogonality makes a fixture showing dot-product zero particularly weak evidence; the new study must obtain residual/activation and behavioral measures.

The old `analysis/paired.py` embeds `REP001_SINGLE_DRAW_SPREAD['acquisition'] = 7`. Reuse its complete-population/discordance ideas, not that superseded program floor. The new floor is 20 for cross-draw acquisition claims, and exact pairing remains appropriate for a deterministic transform of one checkpoint.

## 7. Sequential-skill successors already supply guarded preparation

Evidence at `3ad2769a9f8343a9697e1720fcdc9efa18c3845a`:
`experiments/sequential_complementary_v1/{README.md,delta.py,data.py,training.py,analysis.py,reasoning.py,integrity.py,workflow.py,SOURCE_PINS.json,provenance/TITAN_SPARSE_CORE.json}`.

**Implemented.** Exact magnitude masking, FP64 SVD/FP32 reconstruction, per-tensor norm controls, source-locked tensor-at-a-time checkpoint/factor construction; open corpus identity/length/overlap/shortcut checks; matched final/process/verified-teacher presentations; an RRP serialization/compiler handoff; frozen reasoning and paired/factorial analysis; create-only receipts.

`delta.truncate` returns the represented low-rank delta, factors, norm-scaled full-direction control and size/rank metadata. `delta.build_checkpoint` verifies source/trained hashes and exact trained names before construction. `training.compile_handoff` reuses `serialization.compile_row`, rather than implementing a different optimizer or loss.

**Evidence.** The README explicitly says CPU mechanics/preparation only, no accepted production corpus, no pretrained SCP execution, and no opened confirmation bank. It identifies a missing exact 4B teacher recipe/checkpoint. That missing teacher binding cannot be filled with a checkpoint name, user headline, or a synthetic hash.

**Native successors.** At `9551d7ba50784a5acd19e8512faaba6d0efc6aca`, `experiments/sequential_complementary_native_v1/BOUNDARY_REVIEW.md` documents an unconditional review lock around the preceding `b4f010...` integration snapshot. Its receipt review is necessary-only and cannot produce a native PASS. The separate `7486a149...` successor has `experiments/scp_native_qualification_v1/README.md`, asset/header audit and fixture-host boundary code, but explicitly unimplemented/unqualified live integration and a refusing launch. These guards are not authority to run the new study.

**Reusable.** Dataset-role and group-isolation contracts, exact teacher verification (no success filtering), byte-identical retention anchors across presentations, compiler handoff, tensor construction and refusal behavior. None is a trained two-skill functional transport demonstration. The successor must add independently defined A/B skill deltas and fixed A-alone/B-alone/composed endpoints rather than call a sequential plan completed composability.

## 8. Retention/acquisition machinery and DOSELORA remain separate

Evidence at the parent `07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd`:
`experiments/model_quality_gpu_v1/{native.py,integrity.py,resources.py,supervisor.py,run.py}`,
`experiments/reasoning_posttraining_v1/{serialization.py,controller.py,native_runtime.py,evaluation_stream.py,instrument.py,statistics.py,sources.py,overlap.py}`,
and root `DATASET_PROGRAMME_PREREG.md`, `BROAD_RETENTION.md`, `LONG_CONTEXT_RETENTION_PREREG.md`, `REHEARSAL_COMPOSITION.md`.

**Reusable APIs.** `native.audit_assets`, exact source loading contracts, `resources.guard/lease`, `supervisor.claim_worker/bounded_child`, controller binding, EOS-set handling, `serialization.compile_row/compile_dialogue`, `evaluation_stream.context_for`, and `sources.safetensor_inventory`. `statistics.paired_clusters` and `resolution_floor` make endpoint clustering and statistical resolution explicit. Some native qualifiers include backward work; they cannot be reused as inference-only authorization.

Source/dataset bindings, sidecar identity, precision, raw outputs, terminal child status, and reserved resources are part of the experiment. A hash-shaped declaration is not independently observed custody or authority. Retention probes and sealed banks must not become transport fitting rows. Reuse existing eligible open data only after the new split and rights/source binding passes.

Evidence at `f254acae6ea6d280333b268373ca2585e6e6e583`:
`research/retention_highdose_lora_v1/{adapters.py,materialize.py,test_adapters.py,launch.py,RUNBOOK.md}` and `research/retention_dose_parameterization_v1/{build_arm.py,paired.py,PREREG.md}`.

**Implemented.** FP32 `LoRALinear`, exact target module inventory, frozen base/allowed trainable names, canonical effective dense-update/merge machinery, norm-comparable export preparation and supervision. Its exact production scope is tail layers 8..27, rank 16, alpha 32, seven projections and existing norms. Base parameter count is 596,049,920; the unmerged adapter-augmented count is 603,258,880. A same-size merged export is therefore distinct from an unmerged small-trainable-parameter model. The present study must state both trainable budget and delivered total size; “596M” is the repository's rounded architecture label.

**Current boundary.** The runbook and user brief say no DOSELORA native model has run, the grant is unspent, wait-002 is terminal, and the old deadline `2026-09-20T09:39:05Z` expired. This audit does not inspect/alter its grants, journal, lease or evidence. No new wait is launched. Any later dose continuation needs the genuine external memory recovery and fresh bounded registration specified by the user, preserving the 7 GiB launcher and separate 6 GiB child guard. Cross-scale work does not acquire dose authority by reading this code.

## 9. Source-observability corrections should be retained

Evidence at `3c76199fb28fb55f0022aa25d93cf876a566f5db`:
`experiments/delta_multi_reasoning_v1/SOURCE_OBSERVABILITY_AUDIT.md`,
`audit/observability.py`, and `tests/test_observability.py`.

This additive standard-library layer distinguishes directly observed source properties from reported/synthetic evidence, local inventories from remote custody, file bytes from tensor identity, and partial row coverage from a whole-bank claim. It supplies nonempty row, prediction, pair and rule-sweep preflights. Its own document explicitly says that collector attestations are not signatures or proof of external execution.

These distinctions apply to this audit itself. A historical file-hash mismatch need not mean model tensors changed: safetensors metadata ordering can change serialization while preserving tensor content. Preserve both the actual file identity used by the experiment and canonical tensor identity where appropriate. Conversely, never attach a newly computed digest to an old unbound run and imply prospective provenance.

## What is new, and what should be reused

| Proposed successor component | Prior art in this repository | New requirement |
| --- | --- | --- |
| Normalized-depth correspondence | Cross-scale architecture and projection labs | Freeze mapping before endpoints; measure correspondence separately |
| Linear/Procrustes representation mapping | E6; XSG affine bridge | Use open disjoint data and apply a frozen map to a paired **skill-minus-base** donor residual |
| Capture identity and immutable artifacts | XSG contracts/artifacts | Bind donor base/skilled pair, student base, rows, token positions, map and receipt chain together |
| Delta extraction, LoRA and norm controls | Titan/SCP/DOSELORA | Keep direction, actual merged norm, rank, parameter budget, rows and steps comparable across S/X/O/R arms |
| Parameter/subspace diagnostics | DMRF, rank/spectral studies | Add layer/module weighting and functional activation overlap; avoid treating vector cosine as behavior |
| Orthogonal projection algebra | Projection/geometry precedents | Protect a small source-bound measured skill/retention basis; preserve removed component and reconstruction receipts |
| Two-delta addition | Prior composition studies | Independently qualify skills A/B, then prospectively measure each composition loss and matched component-strength controls |
| Paired behavioral analysis | RRP/SCP/DMRF | Enforce current 20-row cross-draw rule and comparability before score access; preserve endpoint clustering |
| Donor computation in deployment | CED/graft/stitch | Exclude it: donor informs a merged nominal-596M-or-smaller deliverable |

The distinct scientific contribution sought is a donor-derived, learned functional residual that improves the student's acquisition/interference profile **after** the matched controls, followed by a separately qualified test of whether protected-space projection improves independent two-skill composition. Linear algebra alone, a smaller update, an architectural bridge, or a good representation fit is not that contribution. This is a repository-specific novelty audit, not a claim of novelty against all external literature.

The smallest useful progression is first a frozen, sequential, open-data representation qualification once separately authorized and fully bound; then one bounded learned-skill transfer pilot against S0/S1 and the matched X/O controls. Representation qualification cannot decide whether skill transfer improves behavior. The smallest behavioral test requires an eligible **paired donor base/skilled checkpoint**, one declared skill, a fixed student update budget and the same qualified evaluation instrument across arms. Missing donor-skilled custody or native authority remains a blocker, not an invitation to replace the donor pair with an ordinary 4B response.

**Audit outcome:** prior source and reuse boundaries are established; functional learned-skill transfer, improved retention, and independent composability remain scientifically untested by this successor. No scientific improvement claim is opened.
