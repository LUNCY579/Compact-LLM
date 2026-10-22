# ChatBubble branch consolidation plan

**Plan date: 2026-10-22 (America/Chicago).** Inventory captured on October 21; the 149 source heads were rechecked unchanged on October 22 before execution began.

## Decision and scope

Use **`research/post-training-mimo-centralized-v1`** as the enduring research integration branch. Its audited starting head is [`bf2567da46a310aa39f590cc6c73c99961d2ae90`](https://github.com/LUNCY579/ChatBubble/commit/bf2567da46a310aa39f590cc6c73c99961d2ae90). Create the temporary staging branch **`research/rineya-consolidation-20261022`** from that exact commit, integrate the seven validated source branches below, and promote the result through a PR into the enduring central branch after verification.

The objective is to preserve all source, history, experiment identity and evidence while reducing redundant branch names. Keep historical work reachable through verified tags and a searchable registry. A branch's age, name, negative result or absence of current activity does not prove that its contents are obsolete.

This document records the intended order. **Candidate, validated and executed are separate states.** The immutable source SHAs in [branch_registry.json](branch_registry.json) describe the original 149 branches. Update its execution fields only from actual remote receipts; the presence of a plan or a passing local test does not establish that a merge or deletion occurred.

At this document's preparation checkpoint, the staging branch has been created; the 38 deletions have **not** been executed. The live count is therefore 150 at that checkpoint. Subsequent outcomes belong in [execution_receipt.json](execution_receipt.json), with exact incoming scope in [merge_plan.json](merge_plan.json) and test evidence/limits in [validation.json](validation.json). The bounded cleanup is defined by [cleanup_manifest.json](cleanup_manifest.json) and its guarded executor [prune_branches.py](prune_branches.py).

### Audit results

| Finding | Verified amount | Consequence |
|---|---:|---|
| Original branch names | 149 | Every name and exact head appears once in the registry |
| Distinct head commits | 116 | There are 33 excess names pointing to identical heads |
| Initial cleanup candidates | 38 | Bounded manifest; independently preserved by 33 tag references and 5 parent-history references |
| Validated first integration cohort | 7 branches / 393 added source files | Additive package integration into temporary staging |
| Scoped validation | 654 passed / 40 skip records | 39 individual test-case skips plus one RGI module-level skip |
| Later systems ancestry candidates | 14 | Separate retirement batch after preservation and consumer checks |
| Additional small-vocabulary ancestry candidate | 1 | Deferred; excluded from initial cleanup |
| Untagged historical-main snapshots retained | 2 | Existing archive tag does not preserve either exact tip |
| Full worktree evidence snapshots retained | 14 | Preserve potentially unique ignored/untracked research evidence |

Current central already contains [`main`](https://github.com/LUNCY579/ChatBubble/compare/b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2...bf2567da46a310aa39f590cc6c73c99961d2ae90) and [`post-training-Mimo`](https://github.com/LUNCY579/ChatBubble/compare/03db218ceebc61a2bc04f2482cf7e094045e01c1...bf2567da46a310aa39f590cc6c73c99961d2ae90): it is 540 commits ahead of the former and 79 ahead of the latter. Neither needs a broad merge merely to establish the central destination.

## 1. Preserve the baseline and freeze the first batch

The original baseline references are:

| Role | Branch | Audited head |
|---|---|---|
| Default historical baseline | `main` | `b0ca54bdc1b69e2b19d7baeca2e3fdde3ff0a7b2` |
| Historical MiMo integration parent | `post-training-Mimo` | `03db218ceebc61a2bc04f2482cf7e094045e01c1` |
| Enduring research integration | `research/post-training-mimo-centralized-v1` | `bf2567da46a310aa39f590cc6c73c99961d2ae90` |
| Durable complementary reasoning track | `research/rineya-centralized-reasoning-system-v1` | `2b77e25035bac66b39d1fa83bcb94660a0df4568` |

Before a mutation, re-read the involved source/destination heads and compare them with the frozen SHA. A moved head requires a refreshed diff and decision. Keep source heads as parents of genuine integration commits so their provenance remains accessible. Record the starting central head and each resulting commit. Do not force-push or use an empty “ours” merge to label unintegrated work as merged.

The CRS track has an explicit [owner-directed preservation charter](https://github.com/LUNCY579/ChatBubble/blob/2b77e25035bac66b39d1fa83bcb94660a0df4568/AGENTS.md). Retain its named branch and identity. A newer central branch or reviewed-publication branch does not make CRS obsolete. Its retirement, replacement or abandonment is outside this batch.

## 2. Integrate the seven validated branches in this order

All counts below refer to source files added by the incoming branches, excluding the consolidation documentation itself. The complete trees were checked for these candidates; the GD package has 301 files, exceeding the compare API's 300-file display limit.

| Order | Source branch | Pinned head | Added files | Validation already completed |
|---:|---|---|---:|---|
| 1 | `research/deferred-diffusion-context-hci-notes-v1` | `451ef168ce9521a538b4e31c8ab46d0009b6f5f5` | 4 | Markdown-only scope; retain `DEFERRED_DOCS_ONLY` |
| 2 | `codex/confirm002-paired-correctness-audit-20260930` | `224233406c42d90ab30b6599303323a10c54fb0f` | 6 | 30 synthetic standard-library tests passed |
| 3 | `research/rineya-evidence-function-equilibrium-v1` | `c934589026f7456352d62a83bb102c453e15519a` | 9 | 40 tests passed |
| 4 | `research/rineya-verified-resampling-v1` | `ac9b79ac07ad650d8815c76eb4194a180f23e8d4` | 26 | 67 passed; 11 optional Torch tests skipped |
| 5 | `research/rineya-recursive-self-rewrite-v1` | `8d3ee140accb054273580adb4291384e0dd88e0f` | 22 | 79 passed; 7 optional Torch tests skipped |
| 6 | `research/rineya-generalization-dynamics-v1` | `7ae47a21581e15aa83e0859b880a84eb2d7dcb1d` | 301 | 310 passed; 21 optional Torch tests skipped |
| 7 | `research/rineya-relative-generalization-invariance-v1` | `53b2155b2c54137891ea98a24f9e79892e658527` | 25 | 128 passed; one Torch-dependent module skipped |

The tests cover source/reference mechanics in Python 3.12.14, with NumPy 2.3.5 and pytest 9.1.1 where needed. Torch and Transformers were not installed in the audit environment. Skipped coverage is unexecuted coverage; the RGI skip represents an entire module and should not be described as one passing or exercised native test.

### Merge and promotion checks

For each incoming source, verify its exact head and complete changed scope, then create the integration commit on staging. The resulting tree must retain every existing central blob and add only the approved package paths. Keep the original experiment directories, evidence files and qualification labels. RVS also carries its original root index and CPU workflow; preserve those additions exactly.

After all seven merges, check the aggregate tree and ancestry: all seven source heads must be reachable, all 393 approved additions must match their source blobs, and central's existing source files must be unchanged. Review the staged consolidation documentation separately. Recheck the live central head before promotion. If it moved, reconcile the new changes rather than overwriting them.

Open a PR from staging to central describing the additive source integration, the test results and the skipped coverage. Inspect applicable checks and the final PR diff. Promote only after the scope and preservation checks pass. The original RVS workflow's push filter does not cover the new central/staging names, so local validation must not be presented as an automatically completed GitHub Actions run.

The EFE source is the head of [open PR #28](https://github.com/LUNCY579/ChatBubble/pull/28), whose base is `post-training-Mimo`. Retain that source branch while the PR remains unresolved. Once its exact head is verifiably incorporated into central, record the integration destination when resolving the older PR; do not silently treat an open PR as deleted or merged.

This integration validates no new model quality claim. The paired CONFIRM-002 helper remains a descriptive audit of stored outputs on a spent instrument; its synthetic tests did not reopen that bank. Historical GD evidence remains historical. No checkpoint load, training, GPU run, paid provider call or native efficacy qualification is part of this consolidation.

## 3. Execute the bounded 38-reference cleanup

Use [cleanup_manifest.json](cleanup_manifest.json) as the complete allowlist. Its candidate entries contain the exact branch name, expected source SHA, independent preservation reference, peeled target/ancestry proof and a recovery command. Do not broaden the batch with prefix matching or “all merged branches” deletion.

| Initial deletion category | Count | Preservation mechanism |
|---|---:|---|
| Redundant `codex/archive-20261004/*` references | 30 | Existing annotated tags preserve the exact head or verified ancestor history |
| Worktree parent reference aliases | 5 | Two retained full worktree snapshots preserve their parent history |
| `codex/rvs001-inference-pilot-20260929` | 1 | Independent verified tag; named research RVS branch also remains |
| `multi-reasoning-slft` | 1 | Independent verified tag; `single-layer-fine-tuning` remains |
| `fma-finalize` | 1 | Existing exact-head tag; its commit is also contained in central |
| **Total** | **38** | No preservation keeper is itself a deletion candidate |

Immediately before deleting each ref, verify the candidate head, preservation ref object, peeled commit and required ancestry. Reject the candidate if any expected value changed. Verify it is not a protected baseline, the head/base of an unresolved PR requiring retention, or a known active dependency. Keep preservation tags and the manifest unchanged. Record deletion results individually, including failures; a successful batch request is not proof that every ref was removed.

The dependency scan is bounded: six central workflows, three metadata files and eight exact-central materialized files were checked without finding the 38 names. Five additional default-branch searches returned no matches. This is useful negative evidence, not a complete repository or local-host dependency audit. No local Windows worktree/process state was available. Preserve the full worktree snapshots and avoid replaying historical destructive cleanup scripts.

If exactly this 38-ref batch completes, the original 149 names become **111 original names**. A temporary staging branch would make the live total **112** until it is retired after central promotion and recovery verification. These are projected counts, not execution claims; record the actual post-operation inventory.

## 4. Consolidate the remaining additive work

The registry records all 13 remaining isolated Rineya package candidates and their exact heads. Use this order, refreshing the target tree and running the package's existing scoped CPU suite before each merge:

1. `research/rineya-layered-post-training-v1`, then `research/rineya-tropical-verified-composition-v1`.
2. `research/rineya-bf16-attention-gproj-v1`, `research/rineya-compute-depth-elasticity-v1`, and `research/rineya-length-generalization-regularization-v1`.
3. `research/rineya-self-distillation-axes-v1`, followed by `research/rineya-distillation-dynamics-v1`.
4. `research/rineya-triadic-working-memory-v1`, `research/rineya-useful-fixed-point-recurrence-v1`, `research/rineya-id-balancing-skill-routing-v1`, and `research/rineya-hebbian-fact-memory-v1`.
5. `research/rineya-recursive-self-distillation-v1`, then `research/rineya-spikingbrain-hybrid-memory-v1`.

These are additive source candidates, not already executed merges. The final SpikingBrain branch contains 369 files and substantial historical evidence; inspect the complete tree and preserve source qualification boundaries. RSD also carries workflow/quality-queue additions that require their own scope review. Do not infer that every package is natively qualified because its source merges cleanly.

In parallel with those independent packages, integrate systems work in dependency order:

| Workstream | Concrete next action | Dependency and validation |
|---|---|---|
| Cross-scale stitch/graft | Review 30 isolated additions under `experiments/cross_scale_stitch_graft_v1` | Run existing CPU mechanics suite; preserve native contract |
| EFE fresh validation | Review `codex/efe-func001-fresh-validation` after EFE base | Its two extra commits include native/preflight preparation; preserve blocked/unqualified distinctions |
| Cheap quality and data admission | Review the six cheap-quality branches and `codex/data-admission-001-20260930` after RVS | Compare remaining unique paths after the shared RVS base; abstention also modifies inherited files |
| Quality SFT | Review `codex/rineya-quality-sft-v1` after self-distillation axes | Compare shared package versions; avoid replacing the newer axes package with an older copied version |
| MiMo supervisor | Begin with `codex/mimo-supervisor-death-review-fix-20261002`, then port remaining unique fixes | Preserve receipt identity; Windows lifecycle behavior needs appropriate host validation |
| CRS | Review the two commits in `codex/crs-reviewed-publication-20261010` above durable CRS | Consider advancing the existing CRS branch; its owner-exception branch still has distinct commits |
| PPT and SYNTH | Preserve their distinct packages and evidence; reconcile shared CRS documents explicitly | They diverge from one another; stale shared files must not replace newer CRS work |
| CE8, corpus and Triskill | Use verified family successors as reference points | Development, scout completion, stage-a and soup heads still contain unique commits |

## 5. Retire proven ancestors in a later recorded batch

Four heads are already contained in central. Ten more are contained in named family successors. Keep these out of the initial 38-ref batch until exact-head recovery tags and branch-consumer handoffs are verified.

| Later candidate | Retained successor |
|---|---|
| `codex/dataset-retention-audit-v1` | Central |
| `research/jumplora-rineya-skill-isolation-v1` | Central |
| `research/maxrl-comp002-pretraining-qualification-v1` | Central |
| `research/titan-opallo-exact-endpoint-v3` | Central |
| `codex/comp-rs-native-q1` | `codex/ce8-reviewed-evidence-20261009` |
| `research/comp-rs-001-amendment-1` | Same CE8 successor |
| `codex/comp-rs-methods-warmstart-20260930` | Same CE8 successor |
| `codex/comp-rs-s1-dev-20260930` | Same CE8 successor |
| `codex/triskill-geometric-pilot-20261001` | `research/triskill-geo001-parentq` |
| `codex/triskill-geometric-gapfix-20261001` | Same Triskill successor |
| `research/titan-clare-sparse-composable-skills-v1-claude` | `codex/clare-phase4-scope-fail-closed-20260930` |
| `codex/mrs-phase2-dataset-selection-v1` | `research/rineya-reasoning-corpus-v1` |
| `research/mrs-format-probe-v1` | Same corpus successor |
| `codex/mimo-job-handle-fix-20261001` | `codex/mimo-supervisor-death-review-fix-20261002` |

`research/rineya-small-vocab-depth-distill-v1` is one additional proven ancestor of central and is deferred to a later preservation batch. TAPS is also contained in its adaptation-loop successor, but retain its named qualification baseline until the large campaign and native-host handoff are reviewed.

Ancestry establishes that old commits remain reachable. It does not establish that every old file remains in the successor's current tree or that the successor is scientifically superior. Exact-head tags and the registry preserve the old state for inspection and restoration.

## 6. Keep divergent architecture and historical evidence intact

Do not wholesale merge the following scope-stripped snapshots. Their comparisons begin with at least 300 removed files:

- `codex/clare-phase4-scope-fail-closed-20260930`
- `codex/mimo-artifact-stream-budget-20260930`
- `codex/mimo-capture-fixed-input-audit-20260930`
- `research/fwanda-delta-retention-v1`

Preserve their exact heads and selectively port unique, reviewed changes. GitHub's 300-file compare limit makes a capped file list insufficient for merge approval.

DMRF source-observability, pairwise-disjointness and factorial heads still diverge. The apparent older branches each retain unique commits. Retention-dose, Titan tangent, CED conversion/host qualification and TGSA architecture work require family-specific reconstruction or selective ports with their provenance. `research/tgsa-programme-ledger` has no common ancestor with central or main; keep it as a separate ledger rather than forcing an unrelated-history merge.

Retain these two historical-main references because their exact tips are not covered by the existing historical-main tag:

- `codex/archive-20261004/historical-main-snapshots-24`
- `codex/archive-20261004/historical-main-snapshots-27`

Retain all 14 full worktree snapshots listed as `KEEP_WORKTREE_EVIDENCE_SNAPSHOT` in the registry. The CED loader snapshot contains preservation evidence for previously ignored/untracked files; its older qualification tag does not cover that snapshot. The detached CRS native snapshot diverges from the named CRS and owner-exception histories. Catalogue their unique contents and preserve exact-tip references before considering any branch-name retirement.

## 7. Branch policy after consolidation

Maintain three repository baselines: `main`, the historical `post-training-Mimo` parent, and the enduring central research branch. CRS remains a durable named complementary track. Keep a small explicit set of active family branches while compatibility, native-host qualification or distinct evidence work requires them; CED, TGSA, CRS-related extensions and unresolved native campaigns are examples requiring actual handoff decisions.

For new work, create a temporary feature branch from the appropriate current baseline, record its experiment/family, owner and intended destination, and merge through a scoped review. Once its complete source is integrated and an exact recovery tag/receipt exists, retire the temporary name. Use tags for frozen research milestones and historical snapshots. Creating another `archive/*` branch does not reduce branch clutter.

A practical steady-state target is the three baselines, roughly four to eight active family heads, and a few temporary integration/feature branches. This is an operating target, not a quota that authorizes deleting unresolved work. The 149-row registry remains the searchable index of where every original line of work went.

## Completion and recovery

For every completed merge, record the source SHA, destination before/after, merge commit, PR, file-scope verification and validation limits. For every deleted branch, record its expected SHA, actual deletion result, independent preserved target and restoration command. Re-inventory the repository and reconcile the observed branch count with the successful operations.

Restore a removed branch only to the manifest's recorded commit after confirming that commit remains reachable from its preserved tag/keeper. Undo a faulty central integration through a reviewed revert of its specific merge commit; do not reset or force-push the shared central history. A stalled or failed cohort remains on staging while its discrepancy is resolved.

The first cycle is complete when the seven source heads are verifiably integrated into central, the 38 cleanup candidates have individual successful or explicitly blocked receipts, the remaining preserved tracks/snapshots are intact, and this registry reflects actual remote outcomes. Native model qualification and additional research experiments remain separately scoped work.
