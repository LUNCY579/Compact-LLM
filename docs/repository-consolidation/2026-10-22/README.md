# ChatBubble branch consolidation — executed plan and manual pruning handoff

**Updated 2026-10-22T18:21:09.363Z (UTC).** Source integration is complete for this reviewed batch. **28 source branches are merged into MiMo central and one publication branch is merged into durable CRS.** The owner will delete obsolete branch names manually. No branch deletion was performed by the assistant.

## Current result

| Track / action | Result | Evidence |
| --- | --- | --- |
| MiMo cohort 1 | 7 source branches; 393 added source files | [Merged PR #29](https://github.com/LUNCY579/ChatBubble/pull/29) |
| Durable CRS publication | 1 source branch; 551 additions and 10 modifications; no removed files | [Merged PR #30](https://github.com/LUNCY579/ChatBubble/pull/30) |
| MiMo cohort 2 | 14 source branches; 842 added files | [Merged PR #31](https://github.com/LUNCY579/ChatBubble/pull/31) |
| MiMo cohort 3 | 7 source branches; 253 additions and one EFE README update | [Merged PR #32](https://github.com/LUNCY579/ChatBubble/pull/32) |
| Old EFE PR | Closed as superseded after exact source preservation in #29 | [PR #28](https://github.com/LUNCY579/ChatBubble/pull/28) |
| Abstention | Held for damaged inherited data | [Draft PR #33](https://github.com/LUNCY579/ChatBubble/pull/33) |
| Retrieval | Held for unavailable mandatory tokenizer fixture | [Draft PR #34](https://github.com/LUNCY579/ChatBubble/pull/34) |

The latest verified MiMo source-integration commit is [`66a1c223630fee8aa1a5fdab22ac1db1125a050b`](https://github.com/LUNCY579/ChatBubble/commit/66a1c223630fee8aa1a5fdab22ac1db1125a050b). This concluding documentation update advances that branch without changing project code. Its own final head and merge SHA are recorded by the GitHub PR carrying these documents.

The durable CRS tip is [`1f5e72df4d609b3af70f60b3677d09c32a36455a`](https://github.com/LUNCY579/ChatBubble/commit/1f5e72df4d609b3af70f60b3677d09c32a36455a). The separate CRS track remains required by its preservation charter. The `main` and `post-training-Mimo` historical baselines were rechecked unchanged.

## What to prune manually

Use **[MANUAL_PRUNING.md](MANUAL_PRUNING.md)** for the exact full names, full head SHAs, preservation destinations, recovery mappings and deletion order. It contains:

- **38 preserved alias/duplicate names:** 30 archive aliases, five worktree parent aliases and three redundant original/duplicate names. Thirty-three have existing tag preservation; five are parents of two retained worktree snapshots.
- **29 actually absorbed source names:** 28 now reachable from MiMo central and one from durable CRS.
- **One temporary staging name:** `research/rineya-consolidation-20261022`, to retire last after this concluding documentation PR merges and its exact head is verified in central.

The fresh inventory observed **150 branches**: the original 149 plus the one reused staging branch. All 67 pinned candidates matched their reviewed heads, none was protected or used by an open PR, and all original preservation references matched. If those 67 names and final staging are removed at the verified heads, the count becomes **82**. This is conditional; the assistant has not reported or performed those deletions.

The 38-ref [cleanup_manifest.json](cleanup_manifest.json) remains byte-for-byte unchanged at SHA-256 `624f2b5ea197d1d98035e856b10aff205876737a19621ab2ddf8b28ee25a53e4`. Its guarded executor remains limited to those 38 entries and was not used for deletion. The larger owner-managed set is recorded separately in [manual_pruning_manifest.json](manual_pruning_manifest.json).

## Why these merges preserve the work

Every MiMo source tip is an actual second parent of a dedicated merge commit. Each cohort was promoted using a normal GitHub merge PR with an expected head SHA. There were no force pushes. Complete recursive Git trees were compared rather than relying on capped PR file listings.

PR #29 preserved 9,565 pre-existing file entries unchanged and imported its 393 files at original blob IDs and modes. PR #31 preserved all 9,967 then-existing files and imported 842 more exactly. PR #32 preserved 10,808 then-existing files unchanged, imported 253 files, and applied the single reviewed EFE README update. The EFE update is a three-line link which explicitly preserves the finding that trained endpoints were missing and no fresh native experiment ran.

Two nontrivial union decisions were checked independently. The cheap-quality sources share two identical campaign files but have different run directories, so their per-file union retains every sibling directory. Quality-SFT carries the unchanged 24-file SDA ancestor; central now has a newer 51-file SDA tree. The correct merge keeps all 51 newer files and adds only the 55 quality-SFT files. No incoming SDA change was discarded.

Recursive self-distillation retains both its own package and the complete 57-file quality-queue addition, together with its original CPU workflow. SpikingBrain retains its complete 369-file subtree, including historical artifacts. These preservation checks do not claim that all historical campaigns were rerun.

## Validation and practical limits

| Cohort | Local validation | Limit |
| --- | --- | --- |
| PR #29 | 654 passing test cases; 40 skip records | 39 individual skips plus one RGI module skip; optional Torch/native coverage was not exercised in that cohort |
| PR #31 | 1,859 passing test cases; 479 additional passing subtests; five optional skips | Four triadic checks needed Transformers 4.57.1; one check needed an optional sibling worktree |
| PR #32 | 244 passing test cases; 31 additional passing subtests; no final failures/errors/skips | Data-admission is separately static-only: two syntax-checked Python files, no standalone runtime suite |
| PR #30 | Read-only publication review; 17 immutable pins and 12 final014 freeze rows verified | No fixture replay, native action or model test |

These are cohort execution counts, not a deduplicated count of unique tests across cohorts. Subtests are reported separately. Initial fixture-layout and missing-dependency attempts are retained in the validation records; complete unchanged suites subsequently passed where reported. Exact original source/dependency bytes were rechecked after tests. Source, test, custody and qualification guards were not edited to obtain a pass.

GitHub's RVS jobs for #29 and CDE/RSD jobs for #31 failed before a runner was allocated: `runner_id=0`, empty runner names and no steps. Hosted test commands did not run. The exact startup cause was unavailable through the connector, so hosted validation is recorded as unavailable, not passing. No repository settings, branch protections, workflow requirements or rulesets were weakened to merge. PR #32 had no matching hosted check runs.

Only bounded local CPU mechanics and synthetic fixtures were exercised. No pretrained checkpoint downloads, native training campaigns, GPU runs, provider calls or protected benchmark execution were performed. Passing these fixtures does not establish model quality, native-host qualification or scientific efficacy.

## Keep and resolve the two blocked branches

**Abstention remains in [draft #33](https://github.com/LUNCY579/ChatBubble/pull/33).** Two exact incoming blobs replace valid inherited JSON/JSONL with large NUL-filled byte sequences that fail UTF-8 and ordinary UTF-16/32 parsing. A third inherited replacement could not be retrieved completely. The 16 passing isolated package tests do not validate those unrelated data changes. Recover intended bytes, decide whether restoration is appropriate, and review all three replacements before integration.

**Retrieval remains in [draft #34](https://github.com/LUNCY579/ChatBubble/pull/34).** Ten existing tests passed; `test_native_tokenizer_exact_consumed_budgets` errored because the original tokenizer asset was unavailable. Its mandatory SHA-256 and candidate committed blob are pinned in the draft and [review_blockers.md](review_blockers.md). Obtain the original bytes from an authenticated checkout or existing trusted asset, verify both hashes, and rerun the unchanged complete suite. An approximate tokenizer or skipped assertion does not satisfy that gate.

The separate archive aliases for these two branches are still preserved manual-cleanup candidates. Keep the active source names and open draft PRs until resolved.

## Preserve the CRS boundary

PR #30 absorbed the safe, already-published CRS snapshot into the same durable named branch. Its two source commits are a linear descendant of the prior CRS tip; all 561 changes stay inside the CRS package, with no removed files or workflow changes. Original unpublished raw local history was not introduced as merge parents.

The governing result remains **source6 rejected before fixture; scope014 exhausted at 2/2; full acceptance false; runtime NOT_READY**. Source5/source6 fixture invocations remain zero. The 45 discovered methods are plans, not passing tests. The binding addendum withdrawing overall fixture eligibility remains present. Historical native consumption remains 47.421 seconds, the grant registry remains empty, canonical weights are read-only and CHECK/FINAL remain closed. This consolidation does not authorize another source7, retry, renamed milestone or budget reset under exhausted scope014. The owner-exception branch and detached native worktree remain separate.

## Recommended order for remaining work

1. **Finish this manual cleanup.** Use the exact 67-row list, retain archive tags and snapshot keepers, then remove staging after the concluding documentation PR merges. Recheck a row immediately before deleting it; any moved head or new consumer needs review.
2. **Resolve retrieval and abstention through their draft PRs.** Retrieval needs original bytes and its mandatory test; abstention needs a reviewed data-restoration decision. Their blockers are concrete, so unrelated branches did not wait for them.
3. **Review the 15 conditional ancestor names.** Their successors already preserve the source history, but branch-based launchers, CI, worktree instructions and qualification/checkpoint references still need an explicit handoff. Their exact heads, keeper checkpoints, tag requirements and consumer tasks are in the manual guide. They are not included in the ready set.
4. **Integrate Windows supervisor work selectively.** The remaining MiMo process/containment branches need native-host-specific review and tests. Preserve qualification receipts and operating-system behavior; a Linux CPU fixture pass cannot substitute for those gates.
5. **Keep architecture and evidence tracks until their distinct work is reviewed.** CED/TGSA qualification, divergent DMRF/dose/tangent work and unrelated-history TGSA ledger need selective comparison or separate tracks. Four scope-stripped branches remove broad repository content and are unsuitable for a wholesale central merge. Negative results and old timestamps alone do not make research obsolete.
6. **Preserve unique snapshots.** Historical-main snapshots 24 and 27 have tips not covered by the existing historical-main tag. All 14 full worktree snapshots remain; some capture previously ignored/untracked evidence. The five removable parent aliases are not those full snapshots.

## Branch policy after consolidation

Keep the two durable central branches and the named historical baselines. Use a short-lived branch for each bounded change and a PR into its appropriate central track. Preserve research milestones by exact commits and recovery tags where needed. Retire a source name after its source head, intended file union and branch-name consumers have been accounted for. The registry records the original source SHA separately from later central execution state so historical evidence remains traceable.

## Evidence index

| File | Purpose |
| --- | --- |
| [MANUAL_PRUNING.md](MANUAL_PRUNING.md) | Human-readable full deletion candidates, exact heads, keep list, conditional list and recovery order |
| [manual_pruning_manifest.json](manual_pruning_manifest.json) | The 67 pinned candidates, staging condition, recovery mappings and 15 deferred ancestors |
| [manual_pruning_preflight.json](manual_pruning_preflight.json) | Fresh heads, preservation refs, open-PR exclusions, branch count and baseline checks |
| [branch_registry.json](branch_registry.json) | Every original branch, its immutable audited SHA and its updated disposition |
| [execution_receipt.json](execution_receipt.json) | Actual PR outcomes, merge SHAs, zero assistant deletions and hosted-CI limitations |
| [merge_plan.json](merge_plan.json) | First cohort's full file map and links to later executed cohorts |
| [wave2_manifest.json](wave2_manifest.json) | Fourteen-source full scope, merge parents, complete tree checks and validation |
| [wave3_manifest.json](wave3_manifest.json) | Seven-source union, two held sources, overlap decisions, exact file maps and tests |
| [crs_manifest.json](crs_manifest.json) | CRS publication provenance, 561-file scope and preserved governing state |
| [validation.json](validation.json) | Cohort test counts, skips, static-only review, environments and practical limits |
| [independent_reviews.json](independent_reviews.json) | Independent source-map, tree-object and test-count reviews |
| [review_blockers.md](review_blockers.md) | Concrete blocked-PR diagnosis and remediation steps |
