# REP-001 — does the k=0.25 pruning effect replicate across independent training draws?

**Written before any corpus is drawn, any model is trained, or any arm is measured. This is a
REPLICATION screen: it selects nothing and promotes nothing. Long context and abstention are
report-only throughout.**

## Why this and not something else

Everything INT-001 and SPARSE-001 established descends from **one** training draw —
RRP-015's control at selection seed 17. Neither interpolation nor magnitude pruning can
manufacture replication: both reuse the same update direction. The promoted
`Rineya-Titan-Sparse-Experimental` core therefore rests on a single seed, and its headline —
that keeping the top 25% of the update by magnitude improves preservation at no acquisition
cost — could be a property of that one draw rather than of the recipe.

This screen trains **two new independent selection draws** and asks whether the pruning
effect appears in each, paired within draw.

## Design

For each new selection seed **s ∈ {23, 29}**:

1. Draw a fresh 384-row transition corpus, 128 per family, exposures = 1, excluding the 768
   scored parents — both the historical probe's 384 and the SELECT-VIEWS-001 bank — exactly
   as RRP-015 did via `also_exclude_views=[selection]`.
2. Train from **exact Supa 1.7** (`962abc5b…`) on the unchanged RRP-015 recipe: blocks 8–27,
   `assistant_response_plus_eos`, CpuAdamW, peak LR 1e-5, WSD 20/60/20 to 0.1×peak, betas
   (0.9, 0.95), eps 1e-8, weight decay 0, clip 1.0, FP32, max_sequence 2048, 1 exposure.
   **No rehearsal** — RRP-015 measured the rehearsal arm as harmful and it is not repeated.
3. Consolidate the checkpoint's FP32 `parameter` tensors, manifest-verified per entry.
4. Build the **k = 0.25** per-tensor magnitude-pruned variant with the same deterministic
   builder used for the promoted core (`argsort(|δ|, descending, stable)`), FP32.
5. Measure **both** the unpruned control and the pruned variant on the full battery.

Four models measured. Seed 17's control and k=0.25 are already measured and serve as the
third pair; they are **not** re-run.

**Nothing here is a new intervention.** The recipe, the mask and the battery are all fixed
by prior registrations. Only the selection seed varies, which is the point.

## Instruments and permissions — unchanged, and the bars are unchanged

| instrument | role | permission |
|---|---|---|
| SELECT-VIEWS-001, 384 views | acquisition | **SELECTION-ADMISSIBLE**; floors published pre-hoc: `ledger_replay` 0.2344, `prerequisite_plan` 0.0781, `register_machine` 0.0703 |
| 150-row arithmetic panel + 47 source answers | retention | the gate's own instrument |
| Long-Context Demonstration Suite, 256 rows | long context | **SPENT**, barred TRAIN/SELECTION/CONFIRMATION — **REPORT-ONLY** |
| 64 unanswerable rows inside that bank | literal abstention | same bank, same bar — **REPORT-ONLY**, not independent of the long-context score |

**No selection rule is defined, because this screen selects nothing.** It reports paired
within-draw differences and a count over draws. Long context and abstention are reported for
every arm so the damage the gate cannot see stays visible, and they may not be used to
prefer any arm — which is exactly why no arm is preferred here.

## Registered predictions

The unit of replication is the **draw**, and every comparison is **paired within draw**
(pruned vs its own control), because the draws differ in training data.

1. **Each new draw's control clears all three family floors on SELECT-VIEWS-001.** Seed 17's
   control did, and D1 cleared 6/6 historically on the other bank.
2. **In each draw, `references_retained` for k=0.25 ≥ its own control's.** Seed 17: 41 vs 39.
3. **In each draw, long context for k=0.25 ≥ its own control's.** Seed 17: 164 vs 162 against
   the BF16 export, 164 vs 157 against the unrounded FP32 control — the honest paired
   comparison here is against the **FP32** control, since that is what pruning is applied to.
4. **In each draw, k=0.25 acquisition is within 10 of its own control's.** Seed 17: 232 vs 235.
5. **Predictions 2 and 3 both hold in 2 of 2 new draws.** This is the screen's headline. If
   they hold in 2/2, the pruning effect replicates across independent draws. **If either
   fails in either draw, the effect is draw-dependent and the promoted core's headline is
   materially weakened** — and that will be recorded in the core's own amendment file, not
   only here.
6. **Control acquisition across the three draws spans ≤ 40 of 384.** Seed 17's control scored
   235; D1's six historical seeds spanned 217–239 on the other bank.
7. **No arm reaches `retained` ≥ 43 while clearing all three floors.** This failed on both
   the α line and the k line already; predicting it again tests whether that ceiling is a
   property of seed 17 or of the recipe.

A prediction that fails is recorded as failed.

## Execution and limits

Training runs from the **clean** pilot clone `ChatBubble-retention-pilot-20260917`, because
the native lane refuses a dirty worktree and the primary checkout currently carries another
session's uncommitted edits. All outputs go to `rineya-dataset-retention-records/rep001-20260917`.
Nothing is written inside any git worktree.

Per-seed grant limits, matching RRP-015's shape and well above what one 54k-token run needs:
`processed_training_tokens_all_attempts` 600,000; `model_worker_wall_seconds` 1,800;
`evaluation_and_qualification_model_calls` 240; `model_worker_launches` 2; `retries` 1;
`new_disk_bytes` 8 GiB; `cloud_or_teacher_spend` 0; `on_policy_rl_rollout_tokens` 0.

Matched-condition stop rules, identical to the prior screens: implementation digest
`a329eb7bb0`, transformers 4.51.3, `max_context` 4096, TF32 off,
`float32_matmul_precision=highest`, greedy, 0 rows skipped for overflow. Any arm whose
metadata differs is reported non-comparable rather than compared.

One model job at a time against the 15.22 GiB host RAM limit and the 8 GiB GPU. Each FP32 arm
is ~2.38 GiB; arms are built, measured, and their weights then reclaimed **after** their
hashes are recorded and verified — the operational mistake SPARSE-001's Erratum 1 records is
not repeated. In-run native zero-update qualification on the actual source is required before
any optimizer step, as RRP-015 required.

A failed run is reported, never silently re-rolled at a different seed or setting. If a draw
fails to train, this screen reports one new draw instead of two and says so.

## What this can and cannot establish

**Can.** Whether the pruning effect is a property of the recipe or of a single draw — the
one thing neither prior screen could address, and the binding weakness in the promoted core.

**Cannot.** Confirmation of anything. Long context and abstention remain report-only on a
spent suite; three draws cannot pin a rate to better than a wide interval; and no fresh
confirmation instrument exists. `quality_improvement_established` stays false and
`release_permitted` stays false regardless of outcome.
