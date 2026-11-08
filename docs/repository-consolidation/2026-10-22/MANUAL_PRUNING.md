# Manual branch pruning — 2026-10-22

**67 exact, pinned branch names are candidates for owner-managed deletion: 38 preserved aliases and 29 sources now merged into durable central branches.** Remove the temporary staging branch last, after the concluding documentation PR has merged and its exact head is confirmed in central. That makes **68 names in total** for this handoff.

The remote inventory at **2026-10-22T18:15:28.676Z** contained **150 branches**. All 67 pinned source heads matched, all baseline preservation references matched, and none of the candidates was protected or used by an open PR. No branch was deleted by the assistant. If the same heads remain and all 68 are removed, the count would become **82**. This is a projection, not a claim of completed pruning.

## Manual deletion order

1. Remove the 30 exact archive aliases listed below, retaining their existing archive tags.
2. Remove the five long worktree parent aliases, retaining both full snapshot keepers and every other full worktree snapshot.
3. Remove the three redundant original/duplicate names: `codex/rvs001-inference-pilot-20260929`, `fma-finalize`, and `multi-reasoning-slft`.
4. Remove the 29 actually absorbed source names in the following table, after matching each full head SHA and reviewing any local branch-name consumers. Their PRs are merged; their original heads are reachable from the recorded durable central branch.
5. Remove `research/rineya-consolidation-20261022` last, only after the concluding documentation PR is merged. Use that PR's exact head SHA and verify it is an ancestor of the retained MiMo central head. The final PR record supplies this SHA because a file cannot contain the hash of its own future commit.

Before each deletion, match the exact full branch name and current head. A moved head, missing or changed preservation reference, new open PR, or protection/default-branch restriction stops that row. Keep all listed archive tags and keepers. The 15 conditional ancestors later in this guide are **excluded** from these 67 pinned candidates.

## 29 sources absorbed by merged PRs

“MiMo” means `research/post-training-mimo-centralized-v1`, verified after source integration at `66a1c223630fee8aa1a5fdab22ac1db1125a050b`. “CRS” means `research/rineya-centralized-reasoning-system-v1`, verified at `1f5e72df4d609b3af70f60b3677d09c32a36455a`. Keep both durable branches. The original source commits are actual merge parents; historical source revisions remain accessible even where a later source supersedes one README or retains newer SDA content.

| Exact source branch to retire | Expected full source head | Retained destination | Merged PR |
| --- | --- | --- | --- |
| `research/deferred-diffusion-context-hci-notes-v1` | `451ef168ce9521a538b4e31c8ab46d0009b6f5f5` | MiMo | [#29](https://github.com/LUNCY579/ChatBubble/pull/29) |
| `codex/confirm002-paired-correctness-audit-20260930` | `224233406c42d90ab30b6599303323a10c54fb0f` | MiMo | [#29](https://github.com/LUNCY579/ChatBubble/pull/29) |
| `research/rineya-evidence-function-equilibrium-v1` | `c934589026f7456352d62a83bb102c453e15519a` | MiMo | [#29](https://github.com/LUNCY579/ChatBubble/pull/29) |
| `research/rineya-verified-resampling-v1` | `ac9b79ac07ad650d8815c76eb4194a180f23e8d4` | MiMo | [#29](https://github.com/LUNCY579/ChatBubble/pull/29) |
| `research/rineya-recursive-self-rewrite-v1` | `8d3ee140accb054273580adb4291384e0dd88e0f` | MiMo | [#29](https://github.com/LUNCY579/ChatBubble/pull/29) |
| `research/rineya-generalization-dynamics-v1` | `7ae47a21581e15aa83e0859b880a84eb2d7dcb1d` | MiMo | [#29](https://github.com/LUNCY579/ChatBubble/pull/29) |
| `research/rineya-relative-generalization-invariance-v1` | `53b2155b2c54137891ea98a24f9e79892e658527` | MiMo | [#29](https://github.com/LUNCY579/ChatBubble/pull/29) |
| `research/cross-scale-stitch-graft-framework-v1` | `3059e576555afbeb735d79e129435c12caaf4359` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-layered-post-training-v1` | `54f82501c14f3968a3bfdf4af0a657cbfd0b2421` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-tropical-verified-composition-v1` | `3fb76cb8f7027b643293752fc08c5c288ca8b01f` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-bf16-attention-gproj-v1` | `e4e30cbff2b8e6d14f20fb2a4ee61e39361110d9` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-compute-depth-elasticity-v1` | `7a7f7b60e38c18dbeda7e182daf037e993ca8afd` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-length-generalization-regularization-v1` | `8b78e49fc976b86c12a4ac4e54c98cdf74aea97b` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-self-distillation-axes-v1` | `bdc43afa26f00e84781307ea989a2dead5e33446` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-distillation-dynamics-v1` | `98badfd919bc18866571e5d40e1cf10dd5834cff` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-triadic-working-memory-v1` | `879006c14173e987012499eb8e7a1a8df8456b14` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-useful-fixed-point-recurrence-v1` | `f1861c85c6ba1f64b9aa72dc000bea7da1d9b8c1` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-hebbian-fact-memory-v1` | `a4f38244cb875e5f8691ce19563eb9517d5c24a8` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-id-balancing-skill-routing-v1` | `181e6f07a5d4c78797bfbfeea4d0abbc0a9f8e35` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-recursive-self-distillation-v1` | `54b37b17a279700ff77dec36463c5ee9625ceba5` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `research/rineya-spikingbrain-hybrid-memory-v1` | `743f62584ca1b7e4da254f5940326893fd31e8a2` | MiMo | [#31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| `codex/cheap-quality-complementarity-20260929` | `63ff0f6d5cef6c4779ac173a95f9e8c703b2b949` | MiMo | [#32](https://github.com/LUNCY579/ChatBubble/pull/32) |
| `codex/cheap-quality-shortcuts-20260929` | `97081cb9cb9af1f7d1be5ced74c788b8b5f89326` | MiMo | [#32](https://github.com/LUNCY579/ChatBubble/pull/32) |
| `codex/cheap-quality-stability-20260929` | `996cf0ebc946ceccc9a36e0e38a5303ad8326c40` | MiMo | [#32](https://github.com/LUNCY579/ChatBubble/pull/32) |
| `codex/cheap-quality-token-fit-20260929` | `1861a4859fbd3f4ced8215ece711c3d65186ea10` | MiMo | [#32](https://github.com/LUNCY579/ChatBubble/pull/32) |
| `codex/data-admission-001-20260930` | `00b1ee366207a039f344724bf0acb4b2e6e35281` | MiMo | [#32](https://github.com/LUNCY579/ChatBubble/pull/32) |
| `codex/efe-func001-fresh-validation` | `71cfb145be65e09ea6cbe8014c511e49170078c8` | MiMo | [#32](https://github.com/LUNCY579/ChatBubble/pull/32) |
| `codex/rineya-quality-sft-v1` | `3c099798253f03f8a92ea4ba96152da67b9eee3a` | MiMo | [#32](https://github.com/LUNCY579/ChatBubble/pull/32) |
| `codex/crs-reviewed-publication-20261010` | `dad11145cec7f6292a90bcc2788c5886a3b16143` | CRS | [#30](https://github.com/LUNCY579/ChatBubble/pull/30) |

The full mapping includes source-parent merge commits and recovery commands in [manual_pruning_manifest.json](manual_pruning_manifest.json). Every expanded-manifest restoration command uses the explicit repository URL and an empty expected lease; it creates the branch only if the remote name is absent and cannot update an existing branch. PR #28 was closed as superseded only after its exact EFE source was preserved through PR #29; it was not merged into the historical `post-training-Mimo` baseline.

## Keep the two blocked sources

- **`codex/cheap-quality-abstention-20260929` — [draft PR #33](https://github.com/LUNCY579/ChatBubble/pull/33):** two inherited JSON/JSONL replacements contain verified damaged bytes. A third replacement remains unresolved. Recover intended data and review the replacements before integration. Its separate `codex/archive-20261004/...` alias is still in the preserved 38-ref group; these are different branch names.
- **`codex/cheap-quality-retrieval-20260929` — [draft PR #34](https://github.com/LUNCY579/ChatBubble/pull/34):** the mandatory exact-tokenizer fixture is unavailable. Ten tests passed and one required test errored. Obtain and hash-check the original tokenizer, then rerun the unchanged suite. Its separate archive alias can be retired independently.

See [review_blockers.md](review_blockers.md) for the exact files, hashes and next actions.

## Original 38 preserved aliases

These entries retain the immutable baseline manifest. The expanded manual guide does not expand the existing cleanup executor's fixed 38-ref allowlist. The manual manifest's source hashes identify its original input snapshots; the branch-registry input hash is not a checksum of the updated registry published beside this guide. The assistant has not run that executor for deletion. The detailed preflight is [manual_pruning_preflight.json](manual_pruning_preflight.json); the original inventory remains in [branch_registry.json](branch_registry.json).

## Reading the preservation columns

Thirty-three candidates use **27 existing tags**: 26 annotated tags and one lightweight tag. For an annotated tag, verify both its pinned tag-object SHA and its peeled commit against the JSON manifest. The lightweight `refs/tags/archive/Titan-Sparse/single-layer-fine-tuning` directly names the commit. An “exact tag” preserves the row’s expected head; the single FWANDA ancestor row is explicitly marked with its newer preserved commit. The five worktree aliases are direct parents of retained snapshots. No preservation ref in this baseline is itself slated for deletion.

An archive alias and its named research branch are different refs. The “named equivalent” column identifies names outside the original 38-ref group, even when they have the same commit. Retire an equivalent source name only if it appears in this guide’s verified 29-source table; deleting an archive alias alone does not make that decision.

## 1. Archive references — 30 rows

| Manifest row | Delete only this exact branch | Expected head — full SHA | Independent preservation ref | Reason / named equivalent outside the 38-ref group |
| --- | --- | --- | --- | --- |
| 1 | `codex/archive-20261004/codex-cheap-quality-abstention-20260929` | `351e93120c814d719b340bc4e2ddb5bb506e8b20` | `refs/tags/archive/2026-10-04/codex-cheap-quality-abstention-20260929` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/cheap-quality-abstention-20260929`. |
| 2 | `codex/archive-20261004/codex-cheap-quality-complementarity-20260929` | `63ff0f6d5cef6c4779ac173a95f9e8c703b2b949` | `refs/tags/archive/2026-10-04/codex-cheap-quality-complementarity-20260929` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/cheap-quality-complementarity-20260929`. |
| 3 | `codex/archive-20261004/codex-cheap-quality-retrieval-20260929` | `bdc71a6d58c8c0f6353b1d3ea5291aedba55617e` | `refs/tags/archive/2026-10-04/codex-cheap-quality-retrieval-20260929` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/cheap-quality-retrieval-20260929`. |
| 4 | `codex/archive-20261004/codex-cheap-quality-shortcuts-20260929` | `97081cb9cb9af1f7d1be5ced74c788b8b5f89326` | `refs/tags/archive/2026-10-04/codex-cheap-quality-shortcuts-20260929` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/cheap-quality-shortcuts-20260929`. |
| 5 | `codex/archive-20261004/codex-cheap-quality-stability-20260929` | `996cf0ebc946ceccc9a36e0e38a5303ad8326c40` | `refs/tags/archive/2026-10-04/codex-cheap-quality-stability-20260929` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/cheap-quality-stability-20260929`. |
| 6 | `codex/archive-20261004/codex-cheap-quality-token-fit-20260929` | `1861a4859fbd3f4ced8215ece711c3d65186ea10` | `refs/tags/archive/2026-10-04/codex-cheap-quality-token-fit-20260929` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/cheap-quality-token-fit-20260929`. |
| 7 | `codex/archive-20261004/codex-clare-phase4-scope-fail-closed-20260930` | `c006715d82fbebfccf4e19a97ff4b149befb0b48` | `refs/tags/archive/2026-10-04/codex-clare-phase4-scope-fail-closed-20260930` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/clare-phase4-scope-fail-closed-20260930`. |
| 8 | `codex/archive-20261004/codex-confirm002-paired-correctness-audit-20260930` | `224233406c42d90ab30b6599303323a10c54fb0f` | `refs/tags/archive/2026-10-04/codex-confirm002-paired-correctness-audit-20260930` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/confirm002-paired-correctness-audit-20260930`. |
| 9 | `codex/archive-20261004/codex-data-admission-001-20260930` | `00b1ee366207a039f344724bf0acb4b2e6e35281` | `refs/tags/archive/2026-10-04/codex-data-admission-001-20260930` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/data-admission-001-20260930`. |
| 10 | `codex/archive-20261004/codex-mimo-artifact-stream-budget-20260930` | `9d882d38d307c5a6a2cecb8075e9ae314ae879b3` | `refs/tags/archive/2026-10-04/codex-mimo-artifact-stream-budget-20260930` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/mimo-artifact-stream-budget-20260930`. |
| 11 | `codex/archive-20261004/codex-mimo-capture-failure-accounting-20261001` | `07b65f3e3651788a80bcb042d0d369954fc7cd0c` | `refs/tags/archive/2026-10-04/codex-mimo-capture-failure-accounting-20261001` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/mimo-capture-failure-accounting-20261001`. |
| 12 | `codex/archive-20261004/codex-mimo-capture-fixed-input-audit-20260930` | `bbe2343a5817e9512a408270f9215df0116625d0` | `refs/tags/archive/2026-10-04/codex-mimo-capture-fixed-input-audit-20260930` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/mimo-capture-fixed-input-audit-20260930`. |
| 13 | `codex/archive-20261004/codex-mimo-exact-prefix-capture-20261001` | `b379f727b06e075b7eaeef3e029583f146e7fe81` | `refs/tags/archive/2026-10-04/codex-mimo-exact-prefix-capture-20261001` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/mimo-exact-prefix-capture-20261001`. |
| 14 | `codex/archive-20261004/codex-mimo-job-containment-20261001` | `f4622874de970441e9d255297a72b0b671c8bdb2` | `refs/tags/archive/2026-10-04/codex-mimo-job-containment-20261001` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/mimo-job-containment-20261001`. |
| 15 | `codex/archive-20261004/codex-mimo-job-handle-fix-20261001` | `863476aaf1e394a0796f4c59dcf96f0cc3793d4a` | `refs/tags/archive/2026-10-04/codex-mimo-job-handle-fix-20261001` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/mimo-job-handle-fix-20261001`. |
| 16 | `codex/archive-20261004/codex-mimo-supervisor-death-cpu-20261001` | `c42d494e751543b5fb896bbccecd16436003354d` | `refs/tags/archive/2026-10-04/codex-mimo-supervisor-death-cpu-20261001` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/mimo-supervisor-death-cpu-20261001`. |
| 17 | `codex/archive-20261004/codex-mimo-suspended-job-launch-20261001` | `44cbce50dfe85fb79a35eefee4918b2a0bffef9a` | `refs/tags/archive/2026-10-04/codex-mimo-suspended-job-launch-20261001` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/mimo-suspended-job-launch-20261001`. |
| 18 | `codex/archive-20261004/codex-triskill-geometric-gapfix-20261001` | `332b7f9beffe0c25361258c937fa481df0c68f58` | `refs/tags/archive/2026-10-04/codex-triskill-geometric-gapfix-20261001` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/triskill-geometric-gapfix-20261001`. |
| 19 | `codex/archive-20261004/codex-triskill-geometric-pilot-20261001` | `09a0514e1cdf8e7b1dc7960fc4a8aa2aea94cbbe` | `refs/tags/archive/2026-10-04/codex-triskill-geometric-pilot-20261001` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/triskill-geometric-pilot-20261001`. |
| 20 | `codex/archive-20261004/codex-triskill-stage-a` | `ef4b4d208dfa1a603a5f5bde88a98a06c5ca81bf` | `refs/tags/archive/2026-10-04/codex-triskill-stage-a` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `codex/triskill-stage-a`. |
| 21 | `codex/archive-20261004/fma-finalize` | `5ea07037b15471d6930347f65ead399e7b40b045` | `refs/tags/archive/2026-10-04/fma-finalize` | Exact commit preserved by existing tag. |
| 22 | `codex/archive-20261004/historical-main-snapshots-22` | `7d9b4fd450bc56405f6cec41459a7e83df7958ae` | `refs/tags/archive/2026-10-04/historical-main-snapshots` | Exact commit preserved by existing tag. |
| 23 | `codex/archive-20261004/historical-main-snapshots-retained-refs-heads-fma-finalize` | `5ea07037b15471d6930347f65ead399e7b40b045` | `refs/tags/archive/2026-10-04/fma-finalize` | Exact commit preserved by existing tag. |
| 24 | `codex/archive-20261004/historical-main-snapshots-retained-refs-heads-research-factorized-memory-attention-v1` | `1cce3aaa7d5975c97a21d0534e9652bf993a61f2` | `refs/tags/archive/2026-10-04/research-factorized-memory-attention-v1` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `research/factorized-memory-attention-v1`. |
| 25 | `codex/archive-20261004/research-factorized-memory-attention-v1` | `1cce3aaa7d5975c97a21d0534e9652bf993a61f2` | `refs/tags/archive/2026-10-04/research-factorized-memory-attention-v1` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `research/factorized-memory-attention-v1`. |
| 26 | `codex/archive-20261004/research-fwanda-delta-retention-v1-retained-refs-heads-research-fwanda-delta-retention-v1` | `058e0a8a61c603cb175710e63003ce4a31812e34` | `refs/tags/archive/2026-10-04/research-fwanda-delta-retention-v1` | Source is one commit behind tagged FWANDA tip `80fe326f6620d3d084165929637230c6b3620239`; verified source ancestor. |
| 27 | `codex/archive-20261004/research-fwanda-delta-retention-v1` | `80fe326f6620d3d084165929637230c6b3620239` | `refs/tags/archive/2026-10-04/research-fwanda-delta-retention-v1` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `research/fwanda-delta-retention-v1`. |
| 28 | `codex/archive-20261004/research-jumplora-rineya-skill-isolation-v1-retained-refs-heads-research-jumplora-rineya-skill-isolation-v1` | `2f7c0e138e27f1fc4bc0fc13dfc814fb86144921` | `refs/tags/archive/2026-10-04/research-jumplora-rineya-skill-isolation-v1` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `research/jumplora-rineya-skill-isolation-v1`. |
| 29 | `codex/archive-20261004/research-jumplora-rineya-skill-isolation-v1` | `2f7c0e138e27f1fc4bc0fc13dfc814fb86144921` | `refs/tags/archive/2026-10-04/research-jumplora-rineya-skill-isolation-v1` | Exact commit preserved by existing tag. Named equivalent outside the 38-ref group: `research/jumplora-rineya-skill-isolation-v1`. |
| 30 | `codex/archive-20261004/retained-refs-heads-main-7d9b4fd4` | `7d9b4fd450bc56405f6cec41459a7e83df7958ae` | `refs/tags/archive/2026-10-04/historical-main-snapshots` | Exact commit preserved by existing tag. |

## 2. Worktree parent-reference aliases — five rows

| Manifest row | Delete only this exact branch | Expected head — full SHA | Independent preservation ref | Reason / named equivalent outside the 38-ref group |
| --- | --- | --- | --- | --- |
| 32 | `codex/worktree-save-20261002/chatbubble-binding-integration-pilot-20260920-ref-refs-heads-codex-dataset-retention-audit-v1` | `5b4c22dae78f587c41575269f419087b0c7997f6` | `refs/heads/codex/worktree-save-20261002/chatbubble-binding-integration-pilot-20260920` | Direct parent of retained full snapshot `8e9e4ba9420ab3dc650d247bba36d47329b12ccd`. |
| 33 | `codex/worktree-save-20261002/chatbubble-binding-mixture-pilot-20260920-ref-refs-heads-codex-dataset-retention-audit-v1` | `5b4c22dae78f587c41575269f419087b0c7997f6` | `refs/heads/codex/worktree-save-20261002/chatbubble-binding-integration-pilot-20260920` | Direct parent of retained full snapshot `8e9e4ba9420ab3dc650d247bba36d47329b12ccd`. |
| 34 | `codex/worktree-save-20261002/chatbubble-compose-lane-20260918-ref-refs-heads-codex-dataset-retention-audit-v1` | `8afe9f6bc902dcb270e8121a36a2bd24e982bf87` | `refs/heads/codex/worktree-save-20261002/chatbubble-compose-lane-20260918` | Direct parent of retained full snapshot `dc511f919ce0b57dc8b20b254ce11ae0150e5316`. |
| 35 | `codex/worktree-save-20261002/chatbubble-retention-doselora-resume-20260920-ref-refs-heads-codex-dataset-retention-audit-v1` | `5b4c22dae78f587c41575269f419087b0c7997f6` | `refs/heads/codex/worktree-save-20261002/chatbubble-binding-integration-pilot-20260920` | Direct parent of retained full snapshot `8e9e4ba9420ab3dc650d247bba36d47329b12ccd`. |
| 36 | `codex/worktree-save-20261002/chatbubble-retention-pilot-20260917-ref-refs-heads-codex-dataset-retention-audit-v1` | `5b4c22dae78f587c41575269f419087b0c7997f6` | `refs/heads/codex/worktree-save-20261002/chatbubble-binding-integration-pilot-20260920` | Direct parent of retained full snapshot `8e9e4ba9420ab3dc650d247bba36d47329b12ccd`. |

## 3. Redundant original/duplicate references — three rows

| Manifest row | Delete only this exact branch | Expected head — full SHA | Independent preservation ref | Reason / named equivalent outside the 38-ref group |
| --- | --- | --- | --- | --- |
| 31 | `codex/rvs001-inference-pilot-20260929` | `ac9b79ac07ad650d8815c76eb4194a180f23e8d4` | `refs/tags/archive/2026-10-09/worktrees/rvs001-frozen-pilot` | Exact duplicate of `research/rineya-verified-resampling-v1`; independent frozen-pilot tag preserves the tip. Named equivalent outside the 38-ref group: `research/rineya-verified-resampling-v1`. |
| 37 | `fma-finalize` | `5ea07037b15471d6930347f65ead399e7b40b045` | `refs/tags/archive/2026-10-04/fma-finalize` | Exact tag preserves this tip; original audit also proves central ancestry. |
| 38 | `multi-reasoning-slft` | `954a0c2637e691b493c5ac3d488653a8c1adf675` | `refs/tags/archive/Titan-Sparse/single-layer-fine-tuning` | Exact duplicate of `single-layer-fine-tuning`; independent lightweight tag preserves the tip. Named equivalent outside the 38-ref group: `single-layer-fine-tuning`. |

## 4. Keep these references

### Durable named branches

- `main`
- `post-training-Mimo`
- `research/post-training-mimo-centralized-v1`
- `research/rineya-centralized-reasoning-system-v1`

Keep CRS as a durable named complementary track after its reviewed integration. EFE’s original PR hold is resolved: PR #28 was closed after its source was merged through PR #29. EFE and the other newly integrated names are eligible only as listed in this guide’s verified 29-source table.

### Two untagged historical-main snapshots — keep both

The existing `archive/2026-10-04/historical-main-snapshots` tag peels to `7d9b4fd450bc56405f6cec41459a7e83df7958ae`. It does **not** cover the two tips below. Both had unique merge history beyond that ancestor in the original audit. Each requires an independently verified exact-tip preservation ref before any future retirement review.

| Keep exact branch | Audited head — full SHA |
| --- | --- |
| `codex/archive-20261004/historical-main-snapshots-24` | `6b539d9abf3a71e0c60c7b8342a3c79871205f0a` |
| `codex/archive-20261004/historical-main-snapshots-27` | `4f273e4efc443d422ed5c151bb00938d7fa65ccb` |

### All 14 full worktree snapshots — keep

These complete snapshots are excluded from the baseline. Some contain additional ignored/untracked research evidence or commits beyond the corresponding named research branch. A research-source merge does not by itself prove coverage of its full worktree snapshot. In particular, keep the CED loader snapshot at `840714fa269db8d5d3382b00b0aaf302b3bf1bf4` and the detached CRS native snapshot at `f9e5e11997bc7e23bdb0378574fe447dc3c3ee8b`; the older CED qualification tag and named CRS histories were not sufficient coverage proofs for those tips.

| Keep exact branch | Audited head — full SHA | Role |
| --- | --- | --- |
| `codex/worktree-save-20261002/chat-bubble-rineya-retrofit` | `6647eabf3bac9c3f08fce9dfa4998cc07bac3d16` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-binding-integration-pilot-20260920` | `8e9e4ba9420ab3dc650d247bba36d47329b12ccd` | Required keeper for section 2 parent aliases. |
| `codex/worktree-save-20261002/chatbubble-binding-mixture-pilot-20260920` | `f0a32c916a935852017d62846b46b080deed5c59` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-ced-native-loader-qualification` | `840714fa269db8d5d3382b00b0aaf302b3bf1bf4` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-compose-lane-20260918` | `dc511f919ce0b57dc8b20b254ce11ae0150e5316` | Required keeper for section 2 parent aliases. |
| `codex/worktree-save-20261002/chatbubble-cross-scale` | `174ee4fecd7a69b3746d8c2e6107220685047dc9` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-dmrf` | `1e67cef6f3b9a900ed6acfbc0df185d9599c1a4c` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-dose-param` | `40bfac02c8bdf547bcfd7fdec3684c4918613dd7` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-mrs-phase2-dataset-selection-v1` | `9cbc48031dd52789f3e136b464a5cd9adb60e2b8` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-retention-doselora-resume-20260920` | `621d09ccccd82f7f16d2c13eb4c5b70643e5c961` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-retention-pilot-20260917` | `fae9f3fed9a1387400e35b721c708b87cda27262` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/chatbubble-slft` | `7f34ce84ae1ccf1b6ab7b9b5512ffe99468791ca` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/detached-crs-native-desktop-v2` | `f9e5e11997bc7e23bdb0378574fe447dc3c3ee8b` | Full worktree evidence snapshot; retirement not cleared. |
| `codex/worktree-save-20261002/rineyabeta-supa-fwa-v1` | `96c3931eb51c1b272951d4ddbc60b8c863c17257` | Full worktree evidence snapshot; retirement not cleared. |

### Preserve family successors and other named holds

The conditional candidates below rely on these five retained successor names. Their SHA values are original proven checkpoints, not claims about their latest live heads. Keep the successor ref or first establish a separately reviewed durable replacement that preserves the same history.

| Retained successor | Audited checkpoint — full SHA |
| --- | --- |
| `codex/ce8-reviewed-evidence-20261009` | `16f57f9dc387933f3b40f4c3ea596c5e530ec9c3` |
| `codex/clare-phase4-scope-fail-closed-20260930` | `c006715d82fbebfccf4e19a97ff4b149befb0b48` |
| `codex/mimo-supervisor-death-review-fix-20261002` | `8424bfc46398113757157f763fa6e8a9821d11fc` |
| `research/rineya-reasoning-corpus-v1` | `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` |
| `research/triskill-geo001-parentq` | `ae026104a0ba42ec44f9c2aa4db8920d161212f1` |

Also retain the following named references under this baseline:

| Reference | Audited head — full SHA | Why it remains |
| --- | --- | --- |
| `research/rineya-taps-adaptive-recurrence-v1` | `7ece5cead285ba65c7d73ab57c0327fae2dc40bb` | Ancestry in adaptation-loop proven, but successor introduces a large native campaign. Keep baseline named until native workflow/dependency handoff and campaign review are complete. |
| `codex/rineya-taps-adaptation-loop` | `b36c2dabf0edea4b0edee21b77df6a593738652c` | Diverged history with unique work. Preserve head; inspect complete tree and overlapping files before integration. |
| `research/tgsa-programme-ledger` | `9427c6d57e0c142ca5844b738f4cf8dd17f7d48f` | GitHub reports no common ancestor with both main and central. Link/retain the TGSA programme ledger separately; do not allow unrelated-history merge for branch-count reduction. |
| `single-layer-fine-tuning` | `954a0c2637e691b493c5ac3d488653a8c1adf675` | Named SLFT branch retained while only its exact duplicate alias is removed. |

Keep unresolved DMRF, CED, TGSA, retention-dose, Titan-tangent and other divergent family branches. The registry’s separate-review dispositions still apply. No name outside this guide’s exact 67-row set and final staging condition is approved by this handoff.

## 5. Fifteen proven ancestors — conditional, not ready for deletion

These 15 names are **outside the 38-ref baseline**. Five were ancestors of central; ten were ancestors of retained family successors. They need a separately recorded consumer handoff and exact-head recovery reference. Ancestry proves that historical commits remain reachable; it does not prove that all old files remain in the successor’s current tree, that the successor is scientifically better, or that a native workflow no longer uses the old branch name.

Every row needs four completed gates: retain/recheck the proven keeper; verify or establish an exact-tip archive tag and recovery instruction; complete the stated consumer handoff; and recheck its live head, open PR usage and branch restrictions. An existing tag reduces preservation work but does not clear the handoff. These gates are **not recorded as completed here**.

| Conditional branch — keep for now | Expected source head — full SHA | Retained keeper and proven checkpoint | Concrete handoff prerequisite |
| --- | --- | --- | --- |
| `codex/comp-rs-methods-warmstart-20260930` | `ecb6f00f58dfdf5c00df9fb901bd1a23b0bfe287` | `refs/heads/codex/ce8-reviewed-evidence-20261009` at `16f57f9dc387933f3b40f4c3ea596c5e530ec9c3` | Record the COMP-RS/CE8 native-host and method/amendment handoff to codex/ce8-reviewed-evidence-20261009; update branch-based warmstart, S1/native-Q1 and qualification references while preserving their original SHA-bound evidence. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `codex/comp-rs-native-q1` | `8f6afcf5ad8a06f6267b3c86c2039d0d54acf679` | `refs/heads/codex/ce8-reviewed-evidence-20261009` at `16f57f9dc387933f3b40f4c3ea596c5e530ec9c3` | Record the COMP-RS/CE8 native-host and method/amendment handoff to codex/ce8-reviewed-evidence-20261009; update branch-based warmstart, S1/native-Q1 and qualification references while preserving their original SHA-bound evidence. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `codex/comp-rs-s1-dev-20260930` | `eee2e0bd31854ea207b231d65eedd4c1c94e1011` | `refs/heads/codex/ce8-reviewed-evidence-20261009` at `16f57f9dc387933f3b40f4c3ea596c5e530ec9c3` | Record the COMP-RS/CE8 native-host and method/amendment handoff to codex/ce8-reviewed-evidence-20261009; update branch-based warmstart, S1/native-Q1 and qualification references while preserving their original SHA-bound evidence. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `codex/dataset-retention-audit-v1` | `07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd` | `refs/heads/research/post-training-mimo-centralized-v1` at `bf2567da46a310aa39f590cc6c73c99961d2ae90` | Update CONTINUATION.md instructions that name this branch to the retained central ref or pinned archive tag. Hand off retention/CED checkout and launcher consumers; keep the full worktree snapshots and their additional evidence. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `codex/mimo-job-handle-fix-20261001` | `863476aaf1e394a0796f4c59dcf96f0cc3793d4a` | `refs/heads/codex/mimo-supervisor-death-review-fix-20261002` at `8424bfc46398113757157f763fa6e8a9821d11fc` | Hand off Windows job-handle/supervisor launchers, checkout instructions and any running native supervisor to the retained review-fix branch. Keep original containment/failure receipts and the existing exact tag. Existing exact-tag evidence: `refs/tags/archive/2026-10-04/codex-mimo-job-handle-fix-20261001`. |
| `codex/mrs-phase2-dataset-selection-v1` | `a4a250338e85442b3762e9c5f0a620ee803c9969` | `refs/heads/research/rineya-reasoning-corpus-v1` at `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` | Hand off dataset-selection, format-probe and corpus-index consumers to research/rineya-reasoning-corpus-v1; preserve exact dataset/source revisions and split-custody receipts. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `codex/triskill-geometric-gapfix-20261001` | `332b7f9beffe0c25361258c937fa481df0c68f58` | `refs/heads/research/triskill-geo001-parentq` at `ae026104a0ba42ec44f9c2aa4db8920d161212f1` | Hand off geometric pilot/gapfix parent-selection and qualification consumers to research/triskill-geo001-parentq; preserve the original controls, parent identities and native-host evidence. Existing exact-tag evidence: `refs/tags/archive/2026-10-04/codex-triskill-geometric-gapfix-20261001`. |
| `codex/triskill-geometric-pilot-20261001` | `09a0514e1cdf8e7b1dc7960fc4a8aa2aea94cbbe` | `refs/heads/research/triskill-geo001-parentq` at `ae026104a0ba42ec44f9c2aa4db8920d161212f1` | Hand off geometric pilot/gapfix parent-selection and qualification consumers to research/triskill-geo001-parentq; preserve the original controls, parent identities and native-host evidence. Existing exact-tag evidence: `refs/tags/archive/2026-10-04/codex-triskill-geometric-pilot-20261001`. |
| `research/comp-rs-001-amendment-1` | `48efd7d147ef472fc9efffa6b9f8c873fa0ed51d` | `refs/heads/codex/ce8-reviewed-evidence-20261009` at `16f57f9dc387933f3b40f4c3ea596c5e530ec9c3` | Record the COMP-RS/CE8 native-host and method/amendment handoff to codex/ce8-reviewed-evidence-20261009; update branch-based warmstart, S1/native-Q1 and qualification references while preserving their original SHA-bound evidence. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `research/jumplora-rineya-skill-isolation-v1` | `2f7c0e138e27f1fc4bc0fc13dfc814fb86144921` | `refs/heads/research/post-training-mimo-centralized-v1` at `bf2567da46a310aa39f590cc6c73c99961d2ae90` | Move any JumpLoRA adapter-registry, skill-isolation launcher and qualification-manifest branch lookups to central or the exact archive tag; retain historical adapter/source bindings. Existing exact-tag evidence: `refs/tags/archive/2026-10-04/research-jumplora-rineya-skill-isolation-v1`. |
| `research/maxrl-comp002-pretraining-qualification-v1` | `21532b765269b19e6aabf8d596e7409be51b37ea` | `refs/heads/research/post-training-mimo-centralized-v1` at `bf2567da46a310aa39f590cc6c73c99961d2ae90` | Hand off COMP-002 qualification and CE8/COMP-RS baseline consumers to central or the exact archive tag; preserve original baseline identity, controls and qualification receipts. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `research/mrs-format-probe-v1` | `6f66caba3d38f0b779311f6bfd3ebe1704a8f071` | `refs/heads/research/rineya-reasoning-corpus-v1` at `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` | Hand off dataset-selection, format-probe and corpus-index consumers to research/rineya-reasoning-corpus-v1; preserve exact dataset/source revisions and split-custody receipts. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `research/titan-clare-sparse-composable-skills-v1-claude` | `d5fd2c2a4b196a87fc43e076c984e19b13fdcb6f` | `refs/heads/codex/clare-phase4-scope-fail-closed-20260930` at `c006715d82fbebfccf4e19a97ff4b149befb0b48` | Hand off CLARE phase/qualification entry points to the retained fail-closed successor or exact tag. The successor is scope-stripped: this retirement does not approve a wholesale merge into central. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `research/titan-opallo-exact-endpoint-v3` | `016da519cc685c0aae31782b9460a24b6328605a` | `refs/heads/research/post-training-mimo-centralized-v1` at `bf2567da46a310aa39f590cc6c73c99961d2ae90` | Hand off Opallo collector, supervisor and exact-endpoint qualification consumers to central or the exact archive tag; preserve endpoint/source identity and native-host receipts. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |
| `research/rineya-small-vocab-depth-distill-v1` | `52430ac6c5b74f1bad5f06e9227af9959595ff0f` | `refs/heads/research/post-training-mimo-centralized-v1` at `bf2567da46a310aa39f590cc6c73c99961d2ae90` | Verify an exact-tip archive tag, then update small-vocabulary/depth-distillation launchers and model-qualification references to central or that tag; preserve the historical checkpoint bindings. Verify or establish an exact-tip recovery tag; none is pinned for this row in this baseline. |

For central-contained rows, the `bf2567da46a310aa39f590cc6c73c99961d2ae90` checkpoint is the original ancestry proof. Central can advance through reviewed merges; the executor must establish that the current retained central history still contains the candidate. None of the 15 is marked ready by this document.

## 6. Recovery and completion record

Keep the JSON manifest and tags after deleting branch refs. `python3 docs/repository-consolidation/2026-10-22/prune_branches.py --print-recovery` prints restoration commands derived from the pinned 38 entries without network access or mutation. Those commands use the explicit ChatBubble remote and a missing-ref-only lease. Use that guarded output or the expanded manual manifest’s missing-ref-only commands. The immutable original `cleanup_manifest.json` retains its older illustrative `origin` strings as historical audit data.

For manual work, record each removed full branch name, expected head, preserved ref, timestamp and observed outcome in a separate receipt. A missing branch may already have been removed; record that state rather than assuming this guide performed the action. The 29 newly absorbed source names and the final staging condition are recorded at the beginning of this guide.

### Source records

- [Immutable baseline allowlist and recovery mappings](cleanup_manifest.json)
- [Original 149-branch registry and ancestry/successor evidence](branch_registry.json)
- [Recorded latest manual preflight](manual_pruning_preflight.json)
- [Machine-readable manual candidates, conditional gates and keep sets](manual_pruning_manifest.json)

