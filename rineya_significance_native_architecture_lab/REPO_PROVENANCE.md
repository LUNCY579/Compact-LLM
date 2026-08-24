# Repository and model provenance

Recorded before new ChatBubble experiments. Times and mutable branch names are
secondary to the immutable commit and blob identifiers below.

## Working repository: ChatBubble

| Field | Recorded value |
|---|---|
| Remote | `https://github.com/LUNCY579/ChatBubble.git` |
| Branch inspected | `main` |
| Base commit | `03777703dd8587f1248d9a72115eafb9356d50b8` |
| Remote relationship at audit | local `main` matched `origin/main` |
| Worktree at audit | clean |
| Role | writable local experiment repository |

Relevant historical commits:

| Commit | Relevance |
|---|---|
| `9eb838a` | introduced the R2 CrossDepth lab and local R2 scaffold |
| `3775b78` | recorded R2 hybrid stitch manifests |
| `fbec58c` | recorded the R2 CPU showcase |
| `64661f3` | recorded R2 CPU chat/RAG development evaluations |
| `0377770` | completed the R2 dual CPU scoreboard |

Architectural references in ChatBubble:

- `lineage.md`
- `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze/CORE.json`
- `Rineya_Published_Cores/Rineya-Halo-1.2.0-devfreeze/config.json`
- the corresponding `CORE.json` and `config.json` files for Halo Uncensored,
  Lyra, Nimbus, and Pulse
- `Rineya_R2_CrossDepth_Soup_Mini_v1/README.md`
- `Rineya_R2_CrossDepth_Soup_Mini_v1/configs/hybrids_r2.json`
- `Rineya_R2_CrossDepth_Soup_Mini_v1/configs/layer_map.json`
- `Rineya_R2_CrossDepth_Soup_Mini_v1/scaffolds/rineya-core-en32k-28L/config.json`
- `Rineya_R2_CrossDepth_Soup_Mini_v1/copy_manifest.json`
- `Rineya_R2_CrossDepth_Soup_Mini_v1/outputs/preflight.json`
- `Rineya_R2_CrossDepth_Soup_Mini_v1/outputs/scoreboard.md`
- `Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py`
- `Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py`

## Read-only mechanism source: RineyaBeta

| Field | Recorded value |
|---|---|
| Remote | `https://github.com/LUNCY579/RineyaBeta.git` |
| Access rule | strictly read-only; no push, branch, tag, issue, PR, or API write |
| Symbolic remote `HEAD` | `refs/heads/r2-healing-harness` |
| Symbolic-HEAD commit | `1664f54de0bbdfde45e311b75bbf3038dbfffe42` |
| Other ref at same commit | `research/csgqr-quantized-remnant` |
| `main` commit inspected | `9e00be74098878290dc2b0d54cbffee4652c6088` |

### The symbolic-HEAD trap

The remote's symbolic `HEAD` is **not** `main`. The two refs must not be treated
as interchangeable:

- Use `main@9e00be74098878290dc2b0d54cbffee4652c6088` as the code authority for
  `scripts/kv/significance.py`.
- That file's `main` blob is
  `30ee29724dce52de5a288f6421123fb718240541`; it includes the `_gather_counts`
  count-alignment fix and `matrix_bytes` accounting.
- The symbolic-HEAD version has blob
  `2dd6a0f207d4f0f3a413bb45684aa88fdbecffbb` and lacks those corrections. It
  is not the implementation source for this lab.
- Use `r2-healing-harness@1664f54de0bbdfde45e311b75bbf3038dbfffe42`
  for the audited paper at `paper/significance_matrix/main.tex`, blob
  `b83c7f37d871f1ddb59a00ccf521c89d84b1c3e3`.

This intentional mixed-source selection must be preserved in every run
manifest. A manifest that records only `RineyaBeta HEAD` is invalid provenance.

Mechanism and evidence references:

- `main@9e00be74:scripts/kv/significance.py`, blob
  `30ee29724dce52de5a288f6421123fb718240541`
- `main@9e00be74:scripts/kv/chunked_prefill.py`, blob
  `499b3ccc8d4e6954d4dac89aacea590ef25fc46f`
- `main@9e00be74:experiments/kv_significance/src/passkey.py`
- `main@9e00be74:experiments/attention_shape/src/compat.py`
- `r2-healing-harness@1664f54:paper/significance_matrix/main.tex`
- source result artifacts present on both inspected refs:
  `experiments/kv_significance/results/pk_128k_pe0.0.json`,
  `experiments/kv_significance/results/pk_128k_pe0.25.json`,
  `experiments/kv_significance/results/pk_128k_pe0.5.json`,
  `experiments/kv_significance/results/pk_128k_pe0.75.json`,
  `experiments/kv_significance/results/pk_128k_pe1.0.json`,
  `experiments/kv_significance/results/pk_64k_ladder.json`, and
  `experiments/kv_significance/results/pk_128k_grid12.json`

Important mechanism-history commits inspected:

| Commit | Source-repository role |
|---|---|
| `9fabd8a` | initial Significance Matrix implementation |
| `46cb130` | promotion of the mechanism |
| `42899d0` | pooling/sink correction |
| `066334e` | fail-loud behavior |
| `c190293` | chunked prefill |
| `8e3525c` | query-conditioned probe |
| `e18815b` | normalization exponent |
| `e0c2dc2` | paired exponent sweep |
| `b999cb0` | count gathering/alignment fix |
| `e03f268` | corrected S+C memory accounting |
| `0bad535` | matched 32K control |
| `84a8e02` | 12.5%-keep saturation study |
| `e8ad999` | audited paper |
| `1664f54` | inspected symbolic-HEAD lineage |

All facts in this section are provenance records. They do not assert that this
new package has reproduced any source-repository result.

## Active published Qwen3-0.6B family

The active empirical target is Halo; the remaining listed cores received the
same one-case runtime compatibility smoke. All are published 28-layer
Qwen3-0.6B-family cores with 1024 hidden size, 16 query heads, 8 KV heads, head
dimension 128, and vocabulary 151,936.

| Core | Weight SHA-256 | Current evidence |
|---|---|---|
| Halo 1.2 | `38fb96fe286776919032c4418eef0a2534e2c2c2865c1465cc9dd4cfe7e36544` | smoke + bounded mini evaluation |
| Halo Uncensored 1.5 | `c5d4a17f69ca932a5d4ad29e4f9ba1a29d7a137c08951fd89f61b4cb95444503` | compatibility smoke only |
| Lyra 1.4 | `87c57b8eef24238cbba8bc98a38298c9c200a0fa05102b368d1c6cdaf363c5aa` | compatibility smoke only |
| Nimbus 1.6 | `7b66b8d0ef06cf7f71bc9001b0c9a330d5f8a0eec1d11eb173ece70ca66ef49c` | compatibility smoke only |
| Pulse 1.6 | `2357574db5557c288f57a6b97b2761830ea265baf3c6a1c0270357a7d0eaa322` | compatibility smoke only |

Halo's verified tokenizer/config hashes are
`be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506`
and `2284cc1687a4f4f6e3a64a13dfcf7562e592b951afa2f91244cc7b212c9cfb85`.
Every run manifest records its own exact files and content-tree hashes.

## Historical out-of-scope R2 reference

The source repository identifies R2 as `artifacts/rineya-core-en32k-28L`, a
Qwen3-derived causal decoder with 474,021,888 parameters. ChatBubble contains a
local copied scaffold at:

`Rineya_R2_CrossDepth_Soup_Mini_v1/scaffolds/rineya-core-en32k-28L`

The copy manifest records:

| Field | Value |
|---|---|
| Model SHA-256 | `a6bc04ca357d446d092789c998ddae658a6e405946f6449ae68be28dbe10c562` |
| Tokenizer JSON SHA-256 | `249fa5ad6183e98ab44786bbbf95b072250773c69d8573c0cb8f2d44212d6958` |
| Config JSON SHA-256 | `ccf0ad01ac3f80d887aa827e43664438fb38a857c68403379f28eae17d081ca6` |
| Weight bytes | `948078864` |
| Vocabulary | `32768` |
| Layers / hidden / MLP | `28 / 1024 / 3072` |
| Query heads / KV heads / head dimension | `16 / 8 / 128` |
| Native maximum positions | `40960` |
| Native RoPE | theta `1000000`, no scaling |

The source history also contains YaRN x4 runtime experiments. This does not
change the ChatBubble checkpoint's native configuration and cannot be silently
enabled in a baseline.

R2 is retained here only because it is part of the mechanism's discovery and
repository audit trail. It and Aegis are excluded from the current experiments,
results indexes, and conclusions. No R2 path is an active confirmation
candidate for this scope.

## Runtime observed during audit

| Component | Version |
|---|---|
| Python | `3.11.15` |
| PyTorch | `2.12.1+cu126` |
| Transformers | `5.14.1` |
| Accelerate | `1.14.0` |
| safetensors | `0.8.0` |

Transformers is pinned by this lab because Qwen3 attention and cache internals
are an integration boundary. A dependency upgrade requires baseline-identity
and cache-invariant reruns.

## Evidence classes

| Label | Meaning |
|---|---|
| `INHERITED_SOURCE_EVIDENCE` | found in immutable RineyaBeta source/history; not reproduced here |
| `EXISTING_CHATBUBBLE_EVIDENCE` | predates this lab in ChatBubble |
| `NEW_CHATBUBBLE_EVIDENCE` | produced by this package with a complete manifest |
| `NOT_RUN` | no result exists |
| `INVALID_OR_CONFOUNDED` | execution occurred but cannot support the intended claim |
