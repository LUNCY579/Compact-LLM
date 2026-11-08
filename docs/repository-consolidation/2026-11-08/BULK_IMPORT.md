# Bulk selective research imports — 2026-11-08

**Mode:** exact Git subtree union, with the existing `research/post-training-mimo-centralized-v1` as the only parent of the import commit. **Not an octopus merge of every branch.** This import is deliberately limited to source trees absent from the integration target; it does not replace central files, does not delete source heads, and does not mark conflicted branches as fully merged.

## Source mapping

| Added path | Source branch | Source tip at review | Exact Git subtree |
| --- | --- | --- | --- |
| `Rineya_Fast_Weight_Attention_Lab_v1` | `research/supa-a1.1-fast-weight-attention-v1` | `d0b417632930170a08d471be2629ed44d76f225c` | `1b374a9ed226c5aa771f137a6bec53b07aceb0bf` |
| `Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1` | `research/tgsa-v5-1-unified-programme` | `b60d4703fc546581998bd06e64d67c3a0e3067c5` | `ec488f6fd03dfbae41b208431e80e1c0f1d9193d` |
| `Rineya_TGSA_Retrofit_v1` | `research/tgsa-rineya-pretrained-retrofit` | `927405ab1b35a42cf7a5acb16fd42d2123ded567` | `d7d0173d02c6f18bd1bc55afafbf9f58a1c14c86` |
| `rineya_halo_long_context_sft_v1` | `research/titan-tangent-task-arithmetic-v1` | `bd512fadeedf50de15bd65367d49a465387767d0` | `90e75e339e18381dcb56bf186d8add710f366091` |
| `rineya_positional_sequential_attention_lab` | `research/titan-tangent-task-arithmetic-v1` | `bd512fadeedf50de15bd65367d49a465387767d0` | `02dec4141d6376bbc636527b1be0e55939bb851e` |
| `rineya_single_layer_ft_v1` | `codex/worktree-save-20261002/chatbubble-slft` | `7f34ce84ae1ccf1b6ab7b9b5512ffe99468791ca` | `20e7ac308b0576acb63cb2a65c3d8d73b0437c18` |
| `TGSA_PROGRAMME` | `research/tgsa-programme-ledger` | `9427c6d57e0c142ca5844b738f4cf8dd17f7d48f` | `03cef3a44520bdf8fc7926eac09b3f8344f13b0a` |
| `LCR_Input_Preflight` | `codex/worktree-save-20261002/chatbubble-ced-native-loader-qualification` | `840714fa269db8d5d3382b00b0aaf302b3bf1bf4` | `23d724d9bd98786d565a8aeae1ac4745941e16d5` |
| `experiments/supa_adaptive_compute_v2` | `codex/supa-adaptive-compute-v2` | `3a16214b028b8f6fd34828f82f1e233d9da132f4` | `3e356df1442e66d6adbbdcb5afa2132b91766668` |
| `experiments/mrs_format_probe_v1` | `research/rineya-reasoning-corpus-v1` | `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` | `b724e0a378db138797f6e17f68214ee85dfbff33` |
| `experiments/multi_reasoning_scout_v2` | `research/rineya-reasoning-corpus-v1` | `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` | `85009a600ab06629647bb7e80ffd59ddbd2f0f5b` |
| `experiments/rineya_reasoning_corpus_v1` | `research/rineya-reasoning-corpus-v1` | `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` | `f14d38a260ef6e33db3d0f80aee3eebabb3a874f` |
| `research/mrs_format_probe_v1` | `research/rineya-reasoning-corpus-v1` | `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` | `a337af5b90550d2555a6bcf4c767dc1cf78701ec` |
| `research/multi_reasoning_scout_v2` | `research/rineya-reasoning-corpus-v1` | `1f91ddcf124beb9b56f43767141d01cf59ccf9e7` | `2c71bf7dd798d815c6be47b34b017a5aed5fdd81` |

**Preflight checks:** the original central root contained 72 entries; the new tree contained 80, with exactly eight additive root directories. The only changed original root tree hashes were `experiments/` and `research/`, which gained **four** and **two** new child directories respectively. All existing original child entries and the existing root README blob remained unchanged. Each added directory points at the exact source Git tree object (no file-content rewrite or partial encoding). The GitHub tree API performed the subtree construction.

This means **14 entirely absent directories** were imported, not that 14 branches were fully integrated. Several source branches contain other, shared-path updates that have *not* been brought into central. Existing conflicting or newer files win by remaining unchanged. Branches are retained for later explicit reconciliation.

## What these packages are

- **Fast-weight attention:** FWA lab, alternative memory/update experiments, evidence and figures; no present claim of universal quality improvement.
- **TGSA v5.1 and pretrained retrofit:** trajectory/attention methods and their separate configuration and empirical records. Independent ledger source has unrelated ancestry; only its `TGSA_PROGRAMME/` directory is imported, not its Git history.
- **Halo-SFT and positional attention:** historical adaptation and positional/significance experiments, with a retained single-layer fine-tuning package. Other divergent versions still exist on preserved branches.
- **LCR preflight:** source-input contract and review aids, not a new authorized dataset.
- **Adaptive compute:** isolated `supa_adaptive_compute_v2` experiment package; data and CPU artifacts are inherited historical evidence, not new runs.
- **Multi-reasoning successor:** MRS format-probe and MRS-2 experiment/research packages, plus source code for the reasoning corpus. Their historical results require the exact protocol and frozen bank, not a generic capability claim.

## Important exclusions and reasons

- **Do not auto-merge [PR #33](https://github.com/LUNCY579/Compact-LLM/pull/33):** incoming inherited data corruption requires a reviewed recovery decision; package tests do not certify those data replacements.
- **Do not auto-merge [PR #34](https://github.com/LUNCY579/Compact-LLM/pull/34):** the exact tokenizer fixture and mandatory native-tokenizer test remain unresolved.
- **Do not automatically merge divergent shared-path branches:** MiMo supervisors, DMRF/geometry, retention-dose, older CED, scientific instruments, and TGSA alternatives often modify the same experiments, gates, model registries, or workflows. Git conflict-free status does not prove scientific compatibility.
- **Do not wholesale import large archives or raw history:** `Rineya_Intelligence_Quality_Foundry_v1` has a Git-tree listing truncated at 24,497 blob entries (over 130 MB of indexed blob content); full worktree snapshot directories can contain generated evidence, weights or unreconciled data. These remain at their existing refs.
- **Do not collapse the durable CRS track:** `research/rineya-centralized-reasoning-system-v1` has a separate scope/authorization boundary and NOT_READY runtime status; it remains preserved independently.
- **Do not delete branches:** the previous exact-SHA owner pruning policy still applies. The historical `main` branch remains separate.

## Validation scope

Tree shape and SHA preservation were checked through Git object IDs. No native model inference, training, evaluation, GPU checks or Python regression suite was executed for this import. Data provenance and licensing are not re-certified. A structural union does not establish that new research packages can all execute together as a single application.

## Remaining reconciliation

A true "all branches" consolidation requires a path-level three-way merge of each still-diverged source, recovery of broken inputs, test and runtime verification appropriate to each subsystem, and a decision for mutually incompatible scientific snapshots. **No claim is made that this batch completes those steps.**
