# CSC-001 Stage D: bounded behavioral test after representation qualification

Prospective design, 2026-09-20. Audited source: `f3d7a14d5090abbfc891e56d2188816b491417c4` on `research/cross-scale-orthogonal-composable-skills-v1`. This document is an amendment proposal to be bound into the Stage D execution registration before its first native call. It is not a completed run, model-quality result, or executable launcher. The user has authorized continued empirical testing and publication of progress; host/runtime/source and bounded execution gates still have to pass.

The first behavioral question is concrete: **does the donor residual help this Supa student learn held-out single-step state transitions, compared with the same student training without the donor residual, and does the difference survive update-magnitude controls?** This is a test of one form of reasoning acquisition. It cannot establish a general improvement in intelligence from one favorable benchmark, representation fit, training loss, or donor response.

No model was run to prepare this design. Only source code, published protocol/erratum documents, open training data, bank identity/partition metadata, and checkpoint bytes/header metadata were inspected. No historical prediction file or new behavioral result was read. The spent long-context answer-key file was hashed and its population fields counted; it was not used to choose a training row, target, alpha, threshold, or arm.

## Corrections required before spending Stage D compute

1. **There is no inherited alpha selector to invoke.** `alpha_sweep.py` explicitly records no selection rule, and `research/retention_dose_parameterization_v1/PREREG.md` describes curve characterization without selection. SELECT-VIEWS-001 supplies family competence floors, not a best-alpha algorithm. This successor prospectively fixes **S1 = 0.5 S0**. The rest of the fixed alpha grid is diagnostic. No behavioral winner is selected.
2. **The 384 rows are the donor/source pool, not the student exposure count.** At batch one, one row per optimizer step, 64 steps expose 64 rows. Freeze the exact shared 64-row order; capture targets only for these rows. Do not report all 384 as student training or spend three unnecessary 384-row capture passes.
3. **Final-prompt residuals must be matched before the answer.** Use `prompt_ids` and the exact last prompt position, which is `answer_start - 1` in the inherited compiler. Do not pool over teacher-forced completion tokens. The student can compute completion CE and read this earlier causal hidden state in the same forward, but the capture and student hidden-state position must match exactly.
4. **Equal interpolation alpha and equal effective norm are different comparisons.** Publish both; do not describe a reciprocal norm-scaled export as simultaneously alpha matched unless measured equalities happen to hold.
5. **Stage C does not qualify training or cached generation.** Its frozen prompt forwards do not validate adapter gradients, completion masking, merge parity, or 512/4096-token cached decoding. Qualify those exact operations before training/probes under the new runtime binding.
6. **X2/O0 remain conditional.** An old tensor file is not an already qualified protected skill, and projecting against a dense old direction may increase each module's rank. Do not let the two-run transfer pilot silently claim orthogonal composability.
7. **Spent banks cannot become selection inputs.** Long-context and its UNKNOWN subset are report-only. The retired HSKILL instrument is not an admissible replacement.

## Frozen source and 64-row exposure proposal

Use the exact existing open prepared source:

`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/rrp015-20260917/control-prepared/microfit-train.jsonl`

Raw SHA-256: `0881c027d166f7cdd0286518aff20e5966f92120f4f3877cbca6855fbee2d9e1`.

The whole file has 384 rows, with compiled token IDs, labels, prompts, parent IDs and family labels. The donor adapter used this full source. The student's bounded pilot uses the following deterministic order, proposed and calculated without model results:

```text
salt = "cross-scale-composable-001/stage-d/seed-1729"
key(row) = (SHA256(UTF8(salt + ":" + row.id)), row.id)
ordered_source = sorted(all_384_rows, key=key)
student_training_rows = ordered_source[:64]
one exposure; no accumulation; one optimizer update per row
```

No post-length-filter reselection, score filtering, family balancing or replacement is permitted. An invalid row stops preparation. The exact source and IDs should be placed in a create-only Stage D row manifest; the following independent metadata calculation supplies a reproducibility check:

| Quantity | Proposed D64 value |
| --- | --- |
| Unique rows / parents actually trained | 64 / 64 |
| Families | ledger_replay 22; register_machine 23; prerequisite_plan 19 |
| Processed tokens per training run | 9,132 |
| Supervised tokens per training run | 910 |
| Maximum complete row / prompt token count | 161 / 143 |
| All 64 `prompt_ids` equal `ids[:answer_start]` | yes |
| First row ID | `RRFR2-e443266e9449f11ea13c7b2f-step56-valid` |
| Last row ID | `RRFR2-0963e4f1060f3273bb8f80cc-step8-valid` |
| Ordered ID-list digest | `8b6f332821ff5f10f4b13b61577930eb5320c205be4c0005d46b01ed22550612` |
| Selected metadata-manifest digest | `38942750da1650f96cad940697e51d86c601365f7514229234835babd2913ad8` |

Both digests use UTF-8 canonical JSON: sorted object keys, separators `(',', ':')`, `ensure_ascii=False`, no NaN. The metadata manifest is the ordered list of objects with `id`, `parent_id`, `family`, `prompt_ids_sha256`, `ids_sha256`, `labels_sha256`, `processed_tokens=len(ids)`, and `supervised_tokens=count(labels != -100)`. The three nested token/label hashes use that same canonical encoding. These are token-array identity checks, not a claim of fresh native tokenizer qualification.

The observed selected-parent intersection with both SELECT-VIEWS-001 and the older report-only process-transition bank is zero. Recompute these exclusions, calibration parent exclusions, and the inherited text-contact safeguards during final preparation. Training uses only the frozen prepared targets; no evaluation answers enter the prefix. Record the unused 320 source rows explicitly. Donor training on 384 while the student sees 64 is intentional use of an existing skilled donor, not a claim of equal total teacher/student learning compute.

This is a very small 910-supervised-token pilot. A null or weak effect may be a dose/capacity limit. It must not be rescued by silently increasing steps, repeating the prefix, expanding adapter modules, or changing the map after reading endpoints. A larger dose would be a separately registered follow-up.

## Two training runs and target capture

Require Stage C's final registered pair `(student block 27, donor block 35)` to pass its frozen mechanical continuation rule. Preserve all three Stage C pair results; do not substitute whichever pair scores best downstream. Bind the exact transport weight/bias receipt, calibration rows, capture dtype/runtime, and source commit into Stage D.

For only the 64 training prompts, capture student base, matched donor base, and donor skilled sequentially: **192 prompt forwards** at batch one, one target pair, zero generated tokens. Persist raw last-prompt-token FP32 vectors and exact token positions. Matched donor base/skilled uses the same qualified NF4 realization, numerical dtype, implementation and adapter off/on policy; no BF16-versus-NF4 subtraction. Recheck the row token prefix against the student forward. The three persisted matrices contain `64*(1024+2560+2560)*4 = 1,572,864` bytes (1.5 MiB), excluding receipts. There is no need to retain donor modules while student training runs.

Then train exactly **S0 and X0**, from identical immutable student base and identical initial LoRA factors, each with the same 64 rows/order and 64 steps. Keep the registered recipe: final four blocks 24..27, q_proj/v_proj only, rank 8, LoRA alpha 16, 163,840 trainable scalars; AdamW LR `1e-4`, betas `(.9,.999)`, epsilon `1e-8`, weight decay zero, gradient clip 1, seed 1729, batch one, no accumulation, token cap 256, completion-only CE. Declare constant LR, zero LoRA dropout and no schedule unless an amended executable registration changes those before either run. The qualified precision and optimizer placement must be identical across the pair.

S0 minimizes completion CE. X0 adds the registered coefficient-one residual MSE:

```text
target_i = (donor_skilled_i - donor_base_i) @ frozen_T
actual_i = student_current_i - frozen_student_base_i
X0 loss = completion_CE + mean((actual_i - target_i)^2) / frozen_calibration_RMS_squared
```

Bind the RMS denominator to the prescribed student-base calibration data and state whether it is scalar/global (proposed) rather than fitted per training batch. Reject zero/nonfinite denominators. Do not recompute it as the student changes. Record both CE and residual-loss components and their gradient contributions at a fixed predeclared diagnostic step, without tuning the coefficient from them. S0 must use the same adapter initialization, kernel/precision policy and row-level loss convention. It may omit donor matching arithmetic, but unequal compute is reported rather than claimed matched.

Save both learned deltas, original factors, effective merged FP32 deltas, and actual optimizer counters. Verify frozen base/other modules unchanged and unmerged-versus-merged forward parity. Record training parameter count separately from dense changed-parameter count. Do not serialize BF16 deltas or infer magnitude from LoRA factors: norm controls use the actual FP32 merged update.

## Prespecified derived arms; no behavioral selection

Let `s = ||delta_S0||_2` and `x = ||delta_X0||_2`, computed from actual dense merged updates. Both must be positive and finite. All norm-scaled constructions require measured tolerance, module inventory and effective-rank agreement before behavioral interpretation.

| Arm | Construction | Role |
| --- | --- | --- |
| B | immutable student base | fresh same-runtime acquisition/retention reference |
| S0 | `B + delta_S0` | ordinary adaptation |
| S1 | `B + 0.5 delta_S0` | **fixed primary scalar control**, no selector |
| X0 | `B + delta_X0` | donor-residual adaptation |
| X1 | `B + (0.5*s/x) delta_X0` | same norm as S1 |
| S_at_X | `B + (x/s) delta_S0` | reciprocal control at X0's norm |
| X_half | `B + 0.5 delta_X0` | same scalar alpha as S1; norm may differ |
| S_quarter | `B + 0.25 delta_S0` | fixed diagnostic grid member |
| S_three_quarter | `B + 0.75 delta_S0` | fixed diagnostic grid member |

The `alpha=0` and `alpha=1` grid endpoints are B and S0; do not duplicate them under different names. Deduplicate other exports only by verified actual weight identity and compatible run/bank bindings, never by an equal aggregate score. Register no alpha search, no selected winner and no adaptive extension. An extreme reciprocal coefficient is an observed geometry issue: preserve it, apply the preregistered finite/resource guards, and mark the contrast blocked if it cannot be safely qualified; do not substitute a more favorable alpha.

The two central transfer contrasts are **X0 vs S0** and **X1 vs S1**. `X0 vs S_at_X` tests the reciprocal norm point. `X_half vs S1` supplies the equal-alpha companion. The full grid is diagnostic; it cannot supply an optimized replacement baseline after results.

This prospectively replaces the earlier unresolved phrase “selected S1 norm” with `0.5*s` wherever later X2/O0 controls use it. At a common norm, X2 compares against X1, O0 against S1, and X2 against O0. Retained, unrescaled projections remain separate diagnostics. No later favorable projection order or norm variant may become the primary arm.

## Concrete existing endpoint battery

All generations must use a fresh source B baseline in the **same new qualified runtime**. Historical totals or raw outputs are not automatically comparable. Hash complete ordered populations, raw outputs, actual probe weights and scorer code before reading any new scored comparison. Every baseline/arm generation must finish; skip/intersection-based comparisons are blocked.

| Endpoint | Frozen population and identity | Generation/scoring and scientific meaning |
| --- | --- | --- |
| SELECT acquisition (primary) | `rineya-dataset-retention-records/select-views-001/SELECT_VIEWS.jsonl`; 384 unique parents, 128/family; SHA `7dd53f2a751b54c349c2ef571e8f9c5153737f4a1ed3df5eafe25f95980e78a3` | Greedy, thinking off, full official EOS set, **explicit 128 new tokens**; canonical JSON correctness primary, exact-string correctness and termination secondary. One-step state transitions, not multi-step planning/general intelligence. SELECTION/DEV admitted; never training or confirmation. |
| Arithmetic retention | `experiments/architecture_benchmark_001/data/monitor_probe.jsonl`; 150 unique rows, 50 each percentage/time/ratio; SHA `5f6df1c14aa082eb2f61024a2652809077dbc91a6d7a4e19933d55f7890ae266` | Greedy 24 new tokens, 512 total context, typed scorer and both strict/permissive bare policies. Report all-150 correctness and each policy's fixed B-correct retained/lost IDs separately. Do not allow new gains to offset forgotten source answers invisibly. |
| Long-context retention | `Rineya_Long_Context_Demo_v1/data/all_prompts.jsonl`; 256 rows; SHA `2619bd2c4a503e8d65e1d22d5090738b91229365b400332ade29c110dece0af5`; answer-key SHA `b95cf52c29a44b3f4eb18d2979c18e3b941d93709a06d908c30295ac6ed66398` | Greedy 24 new tokens, 4096 total context, no truncation; fixed first-line acceptance rule. Report 256 total, 64 Showcase/192 Stability and 64 each base/paraphrase/counterfactual/unanswerable. **SPENT, report-only**; no selection, training or confirmation. |
| Literal abstention (same generations) | 64 unanswerable rows inside the same 256-row long-context bank | Report the existing literal-UNKNOWN readout and unanswerable accepted-answer accuracy separately; neither proves semantic abstention. Also report inappropriate UNKNOWN on 192 answerable rows so a constant abstainer cannot look improved. These are overlapping endpoints, not independent replications. |

The process probe's current default is **96**, so a launcher relying on defaults would change this declared 128-token envelope. Pass limits explicitly and bind them. Tokenizer-only audit must establish every complete acquisition prompt plus output reserve fits the qualified window; no implicit truncation. The 512-token acquisition/panel and 4096-token long-context cached paths need fresh native qualification; Stage C's uncached 256-token prompt check does not cover them.

SELECT floors are the published stronger of copy-before/constant-majority: ledger 30/128, register 9/128, prerequisite 10/128. Report each family's canonical count and the two-sided exact 95% binomial lower bound versus that fixed floor. Do not use the old report-only bank's different floor values merely because both have 384 rows.

The inherited arithmetic “47” and strict “41” reference counts are historical runtime/scorer observations, not declarations to force the fresh baseline to satisfy. If the new B run reproduces the exact old source/scorer/runtime custody, state that separately. Otherwise freeze the new B-correct IDs before arm interpretation and report the new reference size. Never use the arm/source union reference to manufacture a larger retention denominator. GATE v2's old `panel >=62 AND retained >=37/47` is not this study's success criterion: its own amendments explain that 62 requires an improvement over its old source and is not a pure preservation condition; permissive policy review remained pending.

The older `process-transition-probe-001/CANDIDATE_VIEWS.jsonl` (SHA `45674251e38e574229f4ea98d04c30fe623ec6ac58ad308ad491661c84afab7e`) is available as a later report-only secondary instrument, but is excluded from this smallest battery. Do not add it simply to search for a favorable result. The 672-row HSKILL-VIEWS-001 is explicitly **RETIRED_INVALID** by `HSKILL_ERRATUM_001.md`; its confirmation sibling was retired unread. Neither is an eligible positive/negative skill test here.

For each of nine distinct maximum exports, this battery is 790 prompt generations and at most `384*128 + 150*24 + 256*24 = 58,896` generated tokens. Full maximum: **7,110 prompt generations / 530,064 generated-token allowance**, excluding separately charged qualification. These are hard upper bounds, not predictions of runtime. A prospective execution registration may stage the fixed battery into bounded sessions, but must retain every predeclared arm and endpoint; no result-based selective completion. Each new session needs its own unexpired deadline and resource observation.

## Interpretation and inference

Report `(net, nd, exact p)`, repairs, losses, full denominators and row identities for each acquisition contrast. Cross-training-draw contrasts retain `|net| >=20`; scaling/projection of the same learned checkpoint uses its within-draw paired discordance. A matched seed/initialization improves experimental pairing but does not turn S0 and X0 into one learned checkpoint.

Freeze a pilot-specific two-contrast Holm family `{X0-S0, X1-S1}` at .05 if only H1/H4 are executed. Preserve the original five-contrast full-Stage-D family for later X2/O0 completion; do not present partial-family significance as the full registered test. Reciprocal/equal-alpha/grid comparisons are predeclared diagnostics unless a later pre-result amendment enlarges the primary family.

A bounded positive pilot requires acquisition benefit on the shared dev instrument, survival at matched norm/rank/budget, and a disclosed acquisition/retention profile. Report all retention damage, including failures of the existing <=2 percentage-point exploratory tolerance; spent endpoints may describe that profile but may not choose a new arm/alpha. A nonsignificant retention difference does not establish preservation or losslessness. Selection-admissible dev evidence is still exploratory; it is not fresh confirmation or an intelligence-wide claim. Replicate important cross-draw findings with separately registered seed pairs before promotion.

## Protected-subspace candidates for later X2/O0

The strongest directly located existing candidate is COMPOSE-001's separately trained rehearsal/epistemic-boundary H update. Its protocol and preparation code distinguish H-only training from the transition T update and mixed T+H corpus. The physical file currently exists:

`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/compose001-20260918/trained_H_fp32.safetensors`

Fresh streamed SHA-256: `a1103060591a91ee7d9ceb2ad7450621be352b087cd19188f2c383a0dbbc7b90`; size 1,258,500,512 bytes; 220 FP32-trained tensor entries. The header metadata points to `H/journal/runs/train-001/microfit/checkpoint-00028`; that checkpoint manifest was **not present at the referenced local path** during this audit. Prepared source records are present under `compose001-20260918/H/H-prepared/` and `H/H-PLAN.json`, not directly under the outer experiment directory. Header/source-path text and a rehashed consolidated file do not independently verify the absent native checkpoint or successful run custody.

All eight q/v matrices for layers 24..27 appear in the H header with compatible shapes. A source-relative H projection restricted to these modules is mechanically definable, but it is a **new restricted basis**, not the full historical H behavior. Its source hash, scope, original trained-tensor custody, zero-base check, rank, normalization and actual numerical digest must be registered. Without qualified retention relevance and recovered run custody, it remains a candidate, not an admitted protected direction.

There is a second constraint even if custody is recovered: subtracting a component of a dense H direction from a rank-8 student delta generally increases matrix rank. Recompressing to rank 8 may reintroduce overlap and is a separate treatment, while accepting increased rank defeats the rank-matched H4 comparison. Do not silently solve this by relabelling factor count or using an arbitrary random base direction. The initial two-run pilot can answer H1/H4 without that basis; it cannot answer H2/H3. Later X2/O0 need a specifically registered equal-rank control or a qualified compatible protected subspace. New independent skill-B training belongs to Stage E and is not inserted into this pilot just to fabricate a basis.

## Required launch and output contracts

- Freeze the new source commit, implementation inventory, unchanged `.gitattributes` evidence-byte policy, Stage C map receipt, D64 row manifest, tokenizer/runtime, source weights, adapter allowlist and all bank identities in a new Stage D namespace. Never edit old receipts/grants to fit this run.
- Both S0/X0 initializations and all eight target-module shapes must match; 64 actual optimizer updates and the 9,132/910 token accounting must match the frozen D64 rows. Any difference is an incomplete or incomparable run, not a replacement training opportunity.
- Hash actual model/probe bytes and receipt custody before and after every leg. Export digest and the full ordered scored population must be bound to a qualified output receipt; self-consistent metadata supplied beside an arbitrary score file is insufficient.
- Use create-only external output directories. Legacy probes hardcode inherited evidence locations and some expect a source receipt at the current historical implementation digest. Reuse the qualified generation/scoring logic through a source-bound successor collector; do not invoke old writers unchanged or claim an absent frozen reference exists.
- Enforce current shared GPU lease, real free-RAM/resource guards, finite tensors/loss/gradients, unchanged base, valid masking/EOS, source cleanliness and terminal child status. Record every failed/partial attempt. No wrapper reports COMPLETE after a failed child.
- DOSELORA remains a separate lane; none of this borrows its expired session, reserved grant, wait ID, resource thresholds or evidence namespace.

This design makes the next empirical claim assessable while keeping its scope narrow: a successful matched student experiment would be evidence for transfer of this one transition skill. The full orthogonal-composability and broader intelligence claims remain separate questions.
