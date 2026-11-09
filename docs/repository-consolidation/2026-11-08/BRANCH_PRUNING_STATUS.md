# Compact-LLM branch consolidation and pruning — verified status (2026-11-08)

> This is a **branch-ref and source-preservation receipt**, not a claim that all experiments are compatible, tests passed, or data/model results are release-qualified. The snapshot can become stale if new changes are pushed.

**Current verified status: 49 starting branches → 4 remaining, 45 source refs successfully pruned.** The remaining four are `main` plus three independent research-track refs. No new branches were created. Last verified default-branch head before this report commit: `cb9622bdc59bb782c74d7ca9e1bda127377b3440`. Earlier historical snapshots and the initial 21-deletion table below are retained for audit; they are NOT a current inventory.

## Final branch state and pruning verification

| Remaining ref | Verified tip | Disposition |
| --- | --- | --- |
| `main` | `cb9622bdc59bb782c74d7ca9e1bda127377b3440` (before documentation update) | Active default branch |
| `research/rineya-centralized-reasoning-system-v1` | `1f5e72df4d609b3af70f60b3677d09c32a36455a` | Preserve enduring CRS research; scope014 exhausted, native gates closed |
| `research/rineya-ppt-centralized-reasoning-v1` | `c27c9ed0a6d0343a34173adc5be04fd6d95c2d8f` | Preserve complementary PPT CPU sampler; native scoring unqualified |
| `research/rineya-synth-centralized-reasoning-v1` | `d1ec44e723784e9a393cd9a460bb5576acb56d71` | Preserve separate Synth-CRS research, datasets and independent validation scope |

**Most recent completed prune:** [GitHub Actions run 42560515833](https://github.com/LUNCY579/Compact-LLM/actions/runs/42560515833), which successfully deleted four previously merged historical/CRS snapshot branches after exact SHA, open-PR, protection and ancestry checks. All four deleted refs returned 404 on independent post-run verification; the [temporary pruning workflow was removed](https://github.com/LUNCY579/Compact-LLM/commit/cb9622bdc59bb782c74d7ca9e1bda127377b3440) after the run.

### Later verified deletion batches (24 refs after the initial 21)

| Batch | Branches deleted | Successful Actions run |
| --- | ---: | --- |
| MRS Phase II, RTG-1, and DMRF source | 4 | [42540124918](https://github.com/LUNCY579/Compact-LLM/actions/runs/42540124918) |
| FWA, TGSA and adaptive-compute source | 6 | [42540715285](https://github.com/LUNCY579/Compact-LLM/actions/runs/42540715285) |
| Binding/retention worktrees and foundry publication | 6 | [42541684484](https://github.com/LUNCY579/Compact-LLM/actions/runs/42541684484) |
| Historical CED/FWA/cross-scale worktree snapshots | 4 | [42542350330](https://github.com/LUNCY579/Compact-LLM/actions/runs/42542350330) |
| Two historical-main and two CRS snapshots | 4 | [42560515833](https://github.com/LUNCY579/Compact-LLM/actions/runs/42560515833) |

**Final four deleted refs**, with retained source commits:

- `codex/archive-20261004/historical-main-snapshots-24` @ `6b539d9abf3a71e0c60c7b8342a3c79871205f0a`
- `codex/archive-20261004/historical-main-snapshots-27` @ `4f273e4efc443d422ed5c151bb00938d7fa65ccb`
- `codex/crs-owner-exception-001` @ `e47796f1e9b198f9c3213f471f0151ce517a183b`
- `codex/worktree-save-20261002/detached-crs-native-desktop-v2` @ `f9e5e11997bc7e23bdb0378574fe447dc3c3ee8b`

These source commits remain reachable as extra merge parents of [`185d1b86`](https://github.com/LUNCY579/Compact-LLM/commit/185d1b8639c319e34a54daf510e24f794e5f29a4). The current experiment trees were not overwritten. See [CRS/PPT/Synth disposition](CRS_AND_HISTORICAL_SNAPSHOT_DISPOSITION.md) for the separate authorization boundaries and recovery method.

---
### Earlier 21-ref audit snapshot (historical, not current)



## Exact verified deletion batches

| Cohort | Deleted | GitHub Actions verification |
| --- | ---: | --- |
| Previously integrated research | 7 | [success](https://github.com/LUNCY579/Compact-LLM/actions/runs/42536591304) |
| MiMo prior sources | 7 | [success](https://github.com/LUNCY579/Compact-LLM/actions/runs/42537453712) |
| DMRF research | 3 | [success](https://github.com/LUNCY579/Compact-LLM/actions/runs/42537794901) |
| CLARe Phase 4 (scope tests passed) | 1 | [success](https://github.com/LUNCY579/Compact-LLM/actions/runs/42538177871) |
| Composable skills and retention snapshot | 2 | [success](https://github.com/LUNCY579/Compact-LLM/actions/runs/42538533695) |
| Standalone unrelated-history TGSA ledger | 1 | [success](https://github.com/LUNCY579/Compact-LLM/actions/runs/42538828437) |

Each deletion run required the exact approved source SHA, confirmed the named branch was unprotected, checked it was unused by open pull requests, and required `compare/main...<sha>` to report `ahead_by=0`. It re-read the ref before each exact-path deletion and checked that GitHub returned 404 afterward. Runs completed successfully; all deleted refs were independently rechecked as absent. Deleting names leaves source commits reachable through merge-parent ancestry on `main`.


### Previously integrated research

- `codex/exp-peer-stabilizer-matrix` at `3b2aa5f239188da92f8cfcb402e17e001ad5a86e`
- `research/ced-dual28-local-p1-preparation-v1` at `6a56cb64ec03fb729b006192fd58a7d55975e707`
- `codex/triskill-stage-a` at `ef4b4d208dfa1a603a5f5bde88a98a06c5ca81bf`
- `research/triskill-geo001-parentq` at `ae026104a0ba42ec44f9c2aa4db8920d161212f1`
- `research/titan-tangent-task-arithmetic-v1` at `bd512fadeedf50de15bd65367d49a465387767d0`
- `codex/mimo-supervisor-death-review-fix-20261002` at `8424bfc46398113757157f763fa6e8a9821d11fc`
- `single-layer-fine-tuning` at `954a0c2637e691b493c5ac3d488653a8c1adf675`

### MiMo prior sources

- `codex/mimo-artifact-stream-budget-20260930` at `9d882d38d307c5a6a2cecb8075e9ae314ae879b3`
- `codex/mimo-capture-fixed-input-audit-20260930` at `bbe2343a5817e9512a408270f9215df0116625d0`
- `codex/mimo-capture-failure-accounting-20261001` at `07b65f3e3651788a80bcb042d0d369954fc7cd0c`
- `codex/mimo-exact-prefix-capture-20261001` at `b379f727b06e075b7eaeef3e029583f146e7fe81`
- `codex/mimo-job-containment-20261001` at `f4622874de970441e9d255297a72b0b671c8bdb2`
- `codex/mimo-suspended-job-launch-20261001` at `44cbce50dfe85fb79a35eefee4918b2a0bffef9a`
- `codex/mimo-supervisor-death-cpu-20261001` at `c42d494e751543b5fb896bbccecd16436003354d`

### DMRF research

- `research/delta-multi-reasoning-factorial-v1` at `e0d1617e51b0825832a09aeeb13bb832e543c54b`
- `research/dmrf-pairwise-disjointness-v1` at `11f42f6bc11b8f9e235cb5951ca96a263259434a`
- `research/dmrf-source-observability-v1` at `3c76199fb28fb55f0022aa25d93cf876a566f5db`

### CLARe Phase 4 (scope tests passed)

- `codex/clare-phase4-scope-fail-closed-20260930` at `c006715d82fbebfccf4e19a97ff4b149befb0b48`

### Composable skills and retention snapshot

- `codex/worktree-save-20261002/chatbubble-slft` at `7f34ce84ae1ccf1b6ab7b9b5512ffe99468791ca`
- `codex/worktree-save-20261002/chatbubble-dose-param` at `40bfac02c8bdf547bcfd7fdec3684c4918613dd7`

### Standalone unrelated-history TGSA ledger

- `research/tgsa-programme-ledger` at `9427c6d57e0c142ca5844b738f4cf8dd17f7d48f`

## Research and code preserved this round

| Main commit | Scope |
| --- | --- |
| [`a32e9f1d`](https://github.com/LUNCY579/Compact-LLM/commit/a32e9f1daf827f4866b66a3239fc6804f8e6a308) | MiMo bounded artifact stream reads and exact mechanical prompt identities, plus regression tests |
| [`fb75e3bd`](https://github.com/LUNCY579/Compact-LLM/commit/fb75e3bd3ab5a036928bc877b26192d64c5d0869) | Five prior MiMo source lineages and 59 nonoverlapping historical CPU artifacts |
| [`53de4b53`](https://github.com/LUNCY579/Compact-LLM/commit/53de4b5309efef8535d03fb3b38e4001d54b96f4) | DMRF magnitude/dose, pairwise disjointness and source observability, 19 new paths and 11 archived conflicting blobs |
| [`19bd2ae5`](https://github.com/LUNCY579/Compact-LLM/commit/19bd2ae5f304fc5c9700ffe5a0d803fac6413575) | CLARe Phase 4 operator-tier fail-closed guard and tests |
| [`2b4a5ae1`](https://github.com/LUNCY579/Compact-LLM/commit/2b4a5ae174d83d568fa9f2ec6546a576952e7c94) | Composable-skills negative pilot including historical adapter data, nine dose/binding evidence files |
| [`33400779`](https://github.com/LUNCY579/Compact-LLM/commit/3340077993b201af87bb21a7300a0f933c5cd30a) | Unrelated-history TGSA ledger, exact tree retained and divergent root rules archived |

Earlier direct integrations remain intact: Titan Stage A, Titan geometric and quality-repair archives, DMRF tangent research, and the MiMo Windows process containment evidence.

## Test findings that remain open

[GitHub Actions MiMo CPU run 42537218653](https://github.com/LUNCY579/Compact-LLM/actions/runs/42537218653) executed 70 tests: **67 passed, 1 Windows-specific skipped, 1 failed, 1 errored**. The frozen historical producer hash for `qualify_capture.py` conflicts with a newer one-line call-accounting repair; the other error came from the sparse checkout omitting `experiments/opallo_native_collector_v1/supervisor.py`. **These are not silently waived or repaired**, and no claim of native qualification follows. [CLARe scope-test run 42538177871](https://github.com/LUNCY579/Compact-LLM/actions/runs/42538177871) passed its focused tests before branch retirement. No GPU training or model inference was launched in this consolidation.

The one-time pruning workflow was removed in commit [`507722f2`](https://github.com/LUNCY579/Compact-LLM/commit/507722f25fcd97d8a13fb71f17873c7f9c0a4097). It is not a standing deletion automation.

## Earlier remaining 27 branches (historical pinned snapshot — superseded)

These heads were checked against `main` after the merges and remained **diverged**; none qualified for ancestry-only deletion. Preserve them until each has a path/commit-level content reconciliation and its own original source SHA is verified. This includes separate CRS/not-ready authorization boundaries, old worktree and historical snapshots, runtime variants, and dataset custody evidence. For a branch with published source history, use narrow additive imports or clearly scoped alternative archives rather than silently replacing current experiment gates.

| Branch | Head SHA observed | Review cohort |
| --- | --- | --- |
| `codex/archive-20261004/historical-main-snapshots-24` | `6b539d9abf3a71e0c60c7b8342a3c79871205f0a` | historical snapshot |
| `codex/archive-20261004/historical-main-snapshots-27` | `4f273e4efc443d422ed5c151bb00938d7fa65ccb` | historical snapshot |
| `codex/crs-owner-exception-001` | `e47796f1e9b198f9c3213f471f0151ce517a183b` | CRS separate line |
| `codex/rineya-dataset-evidence-20261009` | `01b6ae89277319a45f2f1d209a8bdc0917f5bea4` | dataset evidence |
| `codex/rineya-worktree-cleanup-20261009` | `7f1a5e6b63db7abd3a7a998cbae5ca12330fb021` | research variant |
| `codex/supa-adaptive-compute-v2` | `3a16214b028b8f6fd34828f82f1e233d9da132f4` | research variant |
| `codex/worktree-save-20261002/chat-bubble-rineya-retrofit` | `6647eabf3bac9c3f08fce9dfa4998cc07bac3d16` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-binding-integration-pilot-20260920` | `8e9e4ba9420ab3dc650d247bba36d47329b12ccd` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-binding-mixture-pilot-20260920` | `f0a32c916a935852017d62846b46b080deed5c59` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-ced-native-loader-qualification` | `840714fa269db8d5d3382b00b0aaf302b3bf1bf4` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-compose-lane-20260918` | `dc511f919ce0b57dc8b20b254ce11ae0150e5316` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-cross-scale` | `174ee4fecd7a69b3746d8c2e6107220685047dc9` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-dmrf` | `1e67cef6f3b9a900ed6acfbc0df185d9599c1a4c` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-mrs-phase2-dataset-selection-v1` | `9cbc48031dd52789f3e136b464a5cd9adb60e2b8` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-retention-doselora-resume-20260920` | `621d09ccccd82f7f16d2c13eb4c5b70643e5c961` | worktree snapshot |
| `codex/worktree-save-20261002/chatbubble-retention-pilot-20260917` | `fae9f3fed9a1387400e35b721c708b87cda27262` | worktree snapshot |
| `codex/worktree-save-20261002/detached-crs-native-desktop-v2` | `f9e5e11997bc7e23bdb0378574fe447dc3c3ee8b` | worktree snapshot |
| `codex/worktree-save-20261002/rineyabeta-supa-fwa-v1` | `96c3931eb51c1b272951d4ddbc60b8c863c17257` | worktree snapshot |
| `research/fwanda-delta-retention-v1` | `80fe326f6620d3d084165929637230c6b3620239` | research variant |
| `research/rineya-centralized-reasoning-system-v1` | `1f5e72df4d609b3af70f60b3677d09c32a36455a` | CRS separate line |
| `research/rineya-ppt-centralized-reasoning-v1` | `c27c9ed0a6d0343a34173adc5be04fd6d95c2d8f` | CRS separate line |
| `research/rineya-reasoning-corpus-v1` | `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` | research variant |
| `research/rineya-synth-centralized-reasoning-v1` | `d1ec44e723784e9a393cd9a460bb5576acb56d71` | CRS separate line |
| `research/supa-a1.1-fast-weight-attention-v1` | `d0b417632930170a08d471be2629ed44d76f225c` | research variant |
| `research/supa-abstention-gate-off-v1` | `fefabf086888295690b9ba8eb9d4eb7d0441810e` | research variant |
| `research/tgsa-rineya-pretrained-retrofit` | `927405ab1b35a42cf7a5acb16fd42d2123ded567` | TGSA variant |
| `research/tgsa-v5-1-unified-programme` | `b60d4703fc546581998bd06e64d67c3a0e3067c5` | TGSA variant |

## Rules for further consolidation

Never delete `main`, a moved ref, a protected ref, an open-PR source or a branch whose unique work is not durably preserved. Preserve conflicting source files under their exact blob identities and record which version remains authoritative. Code-level merge conflicts are **not** scientific validation. Do not convert historical CPU fixtures, exposed development scores, blocked training gates or uncertain datasets into native qualification or publication claims. Do not launch new model training or spend sealed evaluation data merely to facilitate branch cleanup.
