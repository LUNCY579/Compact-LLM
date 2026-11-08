# Compact-LLM branch-tip consolidation audit — 2026-11-08

**Scope:** GitHub branch-tip ancestry comparisons against `research/post-training-mimo-centralized-v1`, performed on 2026-11-08. This is a **structural triage**, not a line-by-line code review or an assertion that every divergent change is useful. Historical source publications and previous consolidation records were read before making a merge decision.

## Inventory at audit time

| Classification against the integrated research branch | Live branch names | Meaning |
| --- | ---: | --- |
| Exact current integration ref | **1** | The target branch itself |
| Fully behind / contained in integration history | **13** | No unique commits to merge into this target |
| Diverged | **49** | Both sides have nonshared commits; content may be duplicate, superseded, or genuinely new |
| No common ancestor | **1** | Standalone TGSA programme ledger; a normal merge comparison is inapplicable |
| **Total** | **64** | Includes the already-merged README PR #47's remaining head branch; a temporary documentation PR branch made *after* this inventory is not counted |

At the first inspection, there were 63 branches; creating the historical-main README PR added one branch. Merging [PR #47](https://github.com/LUNCY579/Compact-LLM/pull/47) put a **documentation-only** commit onto historical `main`, so it changed from a strict ancestor to **diverged** (1 ahead, 961 behind at the post-merge check). Neither main's historic research payload nor the integrated code was replaced.

## Existing completed merges

[PRs #29–#32 and #35–#46](https://github.com/LUNCY579/Compact-LLM/pulls?q=is%3Apr+is%3Amerged) have already consolidated many independent research and preservation branches. Read the [2026-10-22 consolidation record](../2026-10-22/README.md) for source maps, actual test limits, preserved merge ancestry, and owner-only branch-pruning guidance. Do **not** re-import absorbed branches or treat merged source history as new evidence.

Only two substantive code PRs were still open when audited:
- [#33: abstention](https://github.com/LUNCY579/Compact-LLM/pull/33) — **blocked**, including two confirmed NUL-damaged inherited JSON/JSONL blobs and one additional unresolved replacement. Its 16 isolated tests do not certify those three inherited-file changes.
- [#34: retrieval](https://github.com/LUNCY579/Compact-LLM/pull/34) — **blocked**, needing the exact `tokenizers==0.22.2` fixture and the pinned tokenizer SHA-256 to pass its mandatory raw-token budget test. Ten tests passed and one errored; an approximate tokenizer or skipped assertion is invalid.

GitHub's `mergeable` flag is only a merge-conflict signal. It does **not** override the scientific/data-integrity blockers above.

## Structural classification of every branch in the snapshot

### Fully contained (13)

- `codex/ce8-production-dev-20261002`
- `codex/ce8-reviewed-evidence-20261009`
- `codex/exp-peer-stabilizer-matrix`
- `codex/multi-reasoning-frozen-scout-v1-completion`
- `codex/rineya-taps-adaptation-loop`
- `codex/titan-quality-offline-publication-20261009`
- `post-training-Mimo`
- `research/ced-dual28-local-p1-preparation-v1`
- `research/ced-looped-flow-paired-host-v1`
- `research/factorized-memory-attention-v1`
- `research/retention-dose-parameterization-v1`
- `research/rineya-reasoning-posttraining-v1`
- `research/rineya-taps-adaptive-recurrence-v1`

### Same branch (1)

- `research/post-training-mimo-centralized-v1`

### Unrelated history (1)

- `research/tgsa-programme-ledger` — **no common Git ancestor**, so preserve its independent source lineage rather than fabricating a merge.

### Diverged (49; **not** automatically safe to merge)

- `codex/archive-20261004/historical-main-snapshots-24`
- `codex/archive-20261004/historical-main-snapshots-27`
- `codex/cheap-quality-abstention-20260929`
- `codex/cheap-quality-retrieval-20260929`
- `codex/clare-phase4-scope-fail-closed-20260930`
- `codex/crs-owner-exception-001`
- `codex/mimo-artifact-stream-budget-20260930`
- `codex/mimo-capture-failure-accounting-20261001`
- `codex/mimo-capture-fixed-input-audit-20260930`
- `codex/mimo-exact-prefix-capture-20261001`
- `codex/mimo-job-containment-20261001`
- `codex/mimo-supervisor-death-cpu-20261001`
- `codex/mimo-supervisor-death-review-fix-20261002`
- `codex/mimo-suspended-job-launch-20261001`
- `codex/rineya-dataset-evidence-20261009`
- `codex/rineya-worktree-cleanup-20261009`
- `codex/supa-adaptive-compute-v2`
- `codex/triskill-stage-a`
- `codex/worktree-save-20261002/chat-bubble-rineya-retrofit`
- `codex/worktree-save-20261002/chatbubble-binding-integration-pilot-20260920`
- `codex/worktree-save-20261002/chatbubble-binding-mixture-pilot-20260920`
- `codex/worktree-save-20261002/chatbubble-ced-native-loader-qualification`
- `codex/worktree-save-20261002/chatbubble-compose-lane-20260918`
- `codex/worktree-save-20261002/chatbubble-cross-scale`
- `codex/worktree-save-20261002/chatbubble-dmrf`
- `codex/worktree-save-20261002/chatbubble-dose-param`
- `codex/worktree-save-20261002/chatbubble-mrs-phase2-dataset-selection-v1`
- `codex/worktree-save-20261002/chatbubble-retention-doselora-resume-20260920`
- `codex/worktree-save-20261002/chatbubble-retention-pilot-20260917`
- `codex/worktree-save-20261002/chatbubble-slft`
- `codex/worktree-save-20261002/detached-crs-native-desktop-v2`
- `codex/worktree-save-20261002/rineyabeta-supa-fwa-v1`
- `docs/readme-research-findings-20261108-main`
- `main`
- `research/delta-multi-reasoning-factorial-v1`
- `research/dmrf-pairwise-disjointness-v1`
- `research/dmrf-source-observability-v1`
- `research/fwanda-delta-retention-v1`
- `research/rineya-centralized-reasoning-system-v1`
- `research/rineya-ppt-centralized-reasoning-v1`
- `research/rineya-reasoning-corpus-v1`
- `research/rineya-synth-centralized-reasoning-v1`
- `research/supa-a1.1-fast-weight-attention-v1`
- `research/supa-abstention-gate-off-v1`
- `research/tgsa-rineya-pretrained-retrofit`
- `research/tgsa-v5-1-unified-programme`
- `research/titan-tangent-task-arithmetic-v1`
- `research/triskill-geo001-parentq`
- `single-layer-fine-tuning`

The GitHub compare view reports a capped file list for some large divergences, so commit/file counts and path excerpts alone cannot prove safe union. A branch with a positive ahead-count may still have its functional changes preserved via an earlier different merge, cherry-pick, or successor. Conversely, a small one-commit branch may modify authoritative data and be unsafe.

## Next selective integration work

1. **MiMo capture/supervisor cluster:** `codex/mimo-*` branches diverge from an older MiMo base. Inspect changed capture code, operating-system-specific process containment, supervisor death/cleanup behavior, saved receipts, and which later successor wins. Run relevant fixtures on the correct native host before importing; do not equate Linux mechanics with Windows qualification.
2. **DMRF/dose/tangent and TGSA architecture tracks:** compare actual code and scientifically valid source records against the later central implementations; keep mutually incompatible experiment definitions and recorded negative results separately. The unrelated-history TGSA ledger needs an explicit standalone preservation decision.
3. **CRS and source-authority branches:** keep `research/rineya-centralized-reasoning-system-v1` durable and distinct from MiMo; the previous audit says CRS scope014 was exhausted and runtime NOT_READY. New merge work is not permission to resume that campaign.
4. **Historical snapshots:** two `codex/archive-20261004/historical-main-snapshots-*` and fourteen full `codex/worktree-save-20261002/*` branches contain provenance/untracked evidence; do not wholesale merge them or delete their last refs.
5. **Blocked quality branches:** obtain exact source bytes/fixtures, resolve the three inherited abstention replacements and retrieval tokenizer checks, rerun relevant tests, then re-review #33/#34.
6. **Branch retirement:** follow [manual pruning safeguards](../2026-10-22/MANUAL_PRUNING.md) with exact current SHAs, existing tag/snapshot preservation, and owner policy. This audit **deleted no branches** and did **not** change test gates, historical data, or launch permissions.

**Disposition:** no remaining divergent code branch was demonstrated safe for an unconditional merge by branch-tip comparison alone. The README improvements can be merged independently without masking those issues. This note does not claim new model training, native GPU execution, or revalidation of inherited experimental scores.
