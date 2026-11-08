# Reviewed code and evidence — 2026-09-14

## Parent selection and read scope

The live `research/ced-rab001-hotfix001-qualification-v1` head was `a36de96b30eb03265db3a18e464a2e2a23ee67bf` before branch creation. It already contains Opallo registration, the repair-mixture design and the exploratory training machinery. The new branch is directly descended from that head, not from old chat-mentioned subjects or main. The sibling behavior screen remains at `d59729433fae8632d0e2bbe6bf99e185fb4c2694`; the XSC subject remains `d82e27c48b2a75b3a9459148990127cd9b14a2e9`.

The review covered the relevant training entrypoint, native model reference and cache/checkpointing behavior, tokenizer template, curated-data builder, repair registration and current published-source evidence available in the conversation and fresh connector reads. This was NOT a full review of every unrelated repository file, and the local workspace is not a complete historical checkout. No model weights were materialized. No pre-existing branch, evidence file or workflow was changed.

## Specific observations and design responses

| Reviewed path at the parent | Observation | MQ-001 response |
|---|---|---|
| `experiments/rab001_exploration/arm_dose.py` | Parses argv at import, imports another CLI by replacing argv, and mutates optimizer module globals | Do not import it. Use isolated functional modules and independent native qualification. |
| Same file, loss/update loop | Full-sequence CE, one row per step; chunked path deliberately preserves this objective | Add explicit response/EOS labels, token-sum loss and matched full-row accumulation groups. New lane, not historical replay. |
| Same file, saved-state handling | Improved endpoint/exhaustion bookkeeping and create-only trainable saves exist | Preserve them. New runner separately saves optimizer moments, RNG and clean group cursors for resumability. |
| `experiments/architecture_benchmark_001/model.py` (`99df00eacf616f69b6e29d1361b34f499b8eda85`) | Has hidden-only output, bound checkpoint closures, causal/segment/cache protections | Adopt the hidden-only loss approach conceptually; do not fork or patch the research model. Native backend must earn a separate receipt. |
| `Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/tokenizer_config.json` (`417d038a63fa3de29cfde265caedae14d1a58d92`) | Last assistant render has an empty thinking stub; plain text-boundary guessing is unsafe | Verify exact generation-prefix and token-prefix equality, then include answer and EOS only. |
| `experiments/rab001_exploration/build_splits_curated.py` (`829f95db67d500134d46d4dc17fe6b4317367f59`) | Separates corpus-intrinsic selection from outcome-based selection; old rows carry IDs/families/messages | Accept explicit bounded JSONL with those fields; freeze selection and order separately and independently audit splits. |
| `REPAIR_PREREG.md` (`8b79fe7dbfee935364e58d2c2cf24c3b1a53dd0e`) | Existing three-seed design, approximately matched budgets, natural-language abstention targets and literal UNKNOWN endpoint | Leave unchanged. Treat it as a separate experiment; do not reinterpret its nulls or reuse its thresholds as a general release gate. |

## Evidence that motivates the plan, not new measurements

All paths below are under ChatBubble at the parent unless noted. They are primary committed records, not independent replications performed by this branch.

* `Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/CORE.json`: recorded BF16 panel 73/150, 41/47 retained; FP32 candidate 74; long-context strict regression 135 vs source 161. Registration is not a general no-regression certificate.
* `LONG_CONTEXT_RETENTION.md`: strict result and post-hoc formatting/answerability decomposition. Containment is not a pre-registered semantic evaluator and does not establish equivalence.
* `experiments/rab001_exploration/outputs/B28full_wsd_seed17_20260914.log`: full WSD endpoint 76/150, 37/47 retained. Compare the right optimizer recipe rather than using the earlier cosine 73 endpoint as if it were WSD.
* `experiments/architecture_benchmark_001/evidence/retention-design-v1/WSD_REPLICATED.md`: four favorable recipe contrasts, but schedule and beta2 change together, and repeated test questions require dependence-aware analysis.
* `experiments/rab001_exploration/AUDIT_2026-09-13.md`: existing multiplicity, Hlast4, gain-strata and checkpoint-provenance corrections. Do not revive broader claims that this audit narrowed.
* `experiments/opallo_behavior_screen_v1/README.md` at sibling `d5972943...`: design and offline replay, not a qualified native collector or executed repair. The historical UNKNOWN metric has substring/contract limitations.

Public method documentation was checked for chat rendering and non-reentrant checkpoint semantics. Documentation links in PLAN are technical references only; exact behavior is gated against the pinned assets and runtime.

## Deliberately not implemented

No independent 2,000-row quality corpus or gold review; no native capture for the sibling screen; no automatic execution of legacy repair arms; no Opallo continuation source; no 4B loader, LoRA/DoRA/QLoRA, preference/RL training, pruning, CED migration, semantic summary judge, sealed benchmark or model release. They are explicitly staged in PLAN rather than represented by placeholder PASS claims.

## Review corrections during implementation

The pre-publication review tightened qualification metric validation (missing source digests cannot pass), recorded CUDA visibility/CUBLAS/Torch-build identity, used write-capable checkpoint file descriptors for durability flushing, and added adversarial review/plan/checkpoint tests. Loss tests compare both values and gradients; optimizer tests cover two beta2 values and weight decay; replay tests compare complete model state after interruption; all six frozen cells execute with random CPU fixtures.

These checks reduce implementation risk. They do not eliminate the outstanding native integration/resource uncertainty. The recorded local receipt distinguishes tests actually executed from skipped native checks and unexecuted training.
