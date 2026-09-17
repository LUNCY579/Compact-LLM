# Current-state audit — MRS-1 completion continuation

Status: draft, prior to `FREEZE.json`. This report separates fresh repository inspection, inherited source-reported evidence and current resource observations. It does not claim native scout execution or model efficacy. Exact initial local/remote ref inventories and dirty status are retained in [SOURCE_PINS.json](SOURCE_PINS.json). The completed [SOURCE_REAUDIT.json](SOURCE_REAUDIT.json), inspected at 2026-09-17 03:54 UTC, records current remote/local refs, preserved historical heads, freshly rehashed selected assets and current resources; it preserves the initial audit's SHA-256 rather than rewriting it. Candidate assets and lineages are in [MODEL_MANIFEST.json](MODEL_MANIFEST.json) and [MODEL_LINEAGE.json](MODEL_LINEAGE.json). Execution still requires its own final-source receipt and fresh preflight; this audit is not execution authority.

## Isolation and current source

The original ChatBubble checkout at `C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification` was inspected at `codex/dataset-retention-audit-v1`, SHA `fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`, with existing modified/deleted/untracked research and audit files. Those changes belong to the existing work and were not reset, cleaned or imported as qualified executable source.

An earlier MRS draft task was interrupted. Its branch `research/multi-reasoning-frozen-scout-v1` remains at the same parent SHA. The completion continues in `C:/Users/clclc/BusinessProject/ChatBubble-multi-reasoning-frozen-scout-v1` on the new isolated branch `codex/multi-reasoning-frozen-scout-v1-completion`, initially also at `fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`. Its additive experiment/research drafts remain uncommitted at this inspection; no freeze or completed result is inferred from the files being present. Historical branch heads and evidence remain preserved.

The selected host is `LUNCY579/ChatBubble`: it contains the dense native path and actual RRP candidate lineage needed for this inference-only study. RineyaArchitecture supplies useful integrity/design evidence but its original loader accepts only exact Supa and its inherited qualifier includes backward probes. Neither those probes nor historical training are part of this scout.

## Repository evidence snapshot

| Repository | Current inspected identity | Evidence and limitation |
|---|---|---|
| `LUNCY579/ChatBubble` | Local/new-parent `fbec90a0d0653aeb745d5223137ae3cfdd9c48dd`; current remote source branch recorded in SOURCE_REAUDIT | Original checkout is dirty. New isolated MRS branch preserves its parent and historical heads. |
| `LUNCY579/SupaExperiment` | Local/remote main `c4a07e37d2ffdbebc7d1b4849f984d5b747012b9`; nonrecurrent `b27fd2b9d9ce582439564c7b43b22d94f2100845`; current matched-MLP shipping branch `3e8a3ec03881ad1ea948db6c1a62c12f73da798f` | Exact refs and local untracked paths are recorded in SOURCE_REAUDIT. Custom matched-MLP policy is not interchangeable with dense zero-bias serving. |
| `LUNCY579/RineyaBeta` | Actual local checkout `C:/Users/clclc/RineyaExperiment`, main `3532c7bc8dbdd48ef8801ba6080cb80b0e90379a`, matching live remote main; official query refinement remote `5500e875c5fa80c8e00b831702a020e57cc1145d` | The actual checkout has pre-existing untracked paths, preserved in SOURCE_REAUDIT. The similarly named `RineyaBeta-supa-fwa-v1` remains a ChatBubble worktree; the later discovery corrects the initial remote-only audit scope. |
| `LUNCY579/RineyaArchitecture` | Clean local/remote main `2d50f33cd896ade7a14dfd70741ed0e9432c82ae` | Read-only current remote inspection found newer branches beyond local tracking refs; details below. No model weight files were found in this checkout. |

No ancestor/root AGENTS file applies to the Architecture main checkout. Applicable instructions in the inspected research branches require isolated additions, preserved historical evidence, exact source/model/runtime qualification and no sealed reads. No source edits were made in Architecture, SupaExperiment or RineyaBeta by this audit.

## Selected candidate identities and limitations

| Candidate | Checkpoint SHA-256 | Current decision |
|---|---|---|
| Frozen Supa 1.7 source, `supa-1.7-frozen` | `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` | Selected secondary source reference; BLOCKED ON RESOURCE LIMIT, fresh native/scout qualification still required. |
| D1 seed-17 retained export, `d1-seed17-bf16-export` | `2d817d0c06ac11d655ff921780b4b132a6d7c966b31dd9a59e20e26eac24f4bd` | Selected primary old comparator; BLOCKED ON RESOURCE LIMIT, fresh native/scout qualification still required. |

The selected files are BF16 storage loaded into FP32 eager execution; this does not restore pre-export precision. The proposed shared policy is native chat, thinking disabled, deterministic greedy, no sampling, no custom abstention/logit bias, no TF32 and batch one. Exact config/tokenizer/assets/runtime versions and source/lineage pins are mandatory in MODEL_MANIFEST; historical results are not transferable qualification receipts for the new runner. No checkpoint is modified.

The existing manifest classifies Opallo as BLOCKED ON NATIVE QUALIFICATION and not selected, RRP asymmetric A1 as BLOCKED ON MISSING CHECKPOINT in the recorded searched paths, additional D1 seeds as NOT RELEVANT to the bounded primary comparison, and matched-MLP as INCOMPARABLE MODEL FAMILY / POLICY for this dense zero-bias instrument. These are bounded audit findings, not claims that no other export can exist elsewhere. SOURCE_REAUDIT additionally records the available `RRP015-control-preserved-hf` export as NOT RELEVANT to this bounded comparison: a third trained checkpoint does not isolate a new inference mechanism and its different historical acquisition bank does not rank it directly against D1.

The completed candidate audit found a locally available approximately 474M-parameter R2 with BF16 storage/FP32 execution, 32,768-token vocabulary and a newline-stop contract. It is INCOMPARABLE MODEL FAMILY / POLICY for this fixed native-chat study and is not selected. Its presence does not justify substituting its historical first-content endpoint for the registered full structured-output endpoint.

Current Supa matched-MLP shipping evidence is at `LUNCY579/SupaExperiment@3e8a3ec03881ad1ea948db6c1a62c12f73da798f`, `experiments/matched_mlp_shipping_v1/README.md`, `SOURCE_MANIFEST.json` and `evidence/2026-09-14-m0/REPORT.md`. The source is custom Supa A1.1, FP16/eager with Transformers 5.14.1; its config SHA-256 is `89c4ae7398449895d044750c858614584b07db8d67130074920ef524c6eb5705`, independently rehashed from the local source config. Its weight hash is `a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026`; the separately stored 156,225-parameter shared FP32 adapter tensor-file hash is `f2ccf8faa099ba9e0dc5b07474c6a3a4ec42f5bee9141422dd795d9ae6612d79`. Its native qualification remains BLOCKED_PREMODEL, after 142 scoped CPU tests only. Canonical bias 5.15625 and deployed bias 3.75 are distinct policies: the recorded adapter transfer result at canonical bias is not a measurement at 3.75, and adapter-aware behavior at 3.75 remains unmeasured. Preserve those distinctions rather than replacing them with this scout's zero-bias policy.

SOURCE_REAUDIT confirms historical heads preserved and zero model loads/protected payload reads. Its fresh selected-asset inventory digests are `5dfa980efc5fc1464614633ae3c5495c36945879c4ee7a363169137f7f0954d6` for Supa and `ffa99edee427a792185f803cf3b2d5beef66431920d6350bdb51cc9cb2be44e4` for D1. These bind inventories and are distinct from the checkpoint-byte SHA-256 values in the table.

## RineyaArchitecture: current remote evidence

Fresh `git ls-remote origin` identified these current relevant heads:

- `research/looped-flow-paired-host-quality-v1`: `70b13aaa1e0d1bb36c558828ccd93850428bfeb7` (absent from stale local tracking refs; inspected through read-only GitHub API).
- `research/model-generation-process-fgp-m1-frozen-qualification-v1`: `22351b4aa0631c4097503a086cb8b45c67c6e969`.
- `research/eam-immediate-diagnostics-v1`: `df20b53358ebc3a9d417334abdbdb297d5850ce9`.
- `research/eam-immediate-insight-v1`: `f5f44ab54ec7df929db175196a4d4e921559c60a`.
- `research/supa-cpu-baseline-readout-v1`: `fa788fe616b70b118b8c40b8092b9b1c99f0aa02`.

Main `docs/NATIVE_QUALIFICATION.md` records 74 tests including five tiny random Qwen3 native tests. This is mechanical evidence, not released-weight qualification. Fresh Actions inspection of run `34487725498` showed mechanics success while job `102906541432`, “Official Supa CPU baseline and readout (no optimizer),” was skipped. The green workflow does not establish official Supa execution. Current EAM head runs `34868010078` and `34867131964` failed; no new native pass is inferred.

At `22351b4a`, `model_development/fgp_m1_frozen_qualification_v1/evidence/IMPLEMENTATION_REPORT.md` records 195 scoped mechanical tests, native checkpoint execution NOT_RUN and no optimization executor. At `70b13aaa`, `experiments/paired_lf_v1/evidence/RESULT.json` and README record native UNEXECUTED and restricted A–G residual answering with new reader training; the CED_LAST4 host is source-initialized, not a recovered trained candidate. EAM diagnostic/insight `evidence/RESULT.md` files record zero pretrained calls. These are not runnable trained candidates for the present study.

The Architecture Supa profile has 596,049,920 parameters, official config SHA-256 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`, tokenizer `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`, and tokenizer config `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`. Its exact-only loader rejects custom bias/policy exports and requires FP32/eager/Transformers 4.57.1 with 6,144 MiB CUDA free before load. Its old qualifier is not adopted unchanged as an inference-only qualifier.

## Bank access and dataset arrival

No sealed, ANY_READ or confirmation payload was opened by the Architecture audit or this documentation continuation. Filename-only Architecture inspection found no local quarantine/confirmation/consumption-ledger files. Absence from that checkout does not clear the cross-repository quarantine inventory; use ACCESS_MANIFEST and the final scout-bank audit.

Architecture EAMI's source-reported audit demonstrates perfect question-suffix and case-ID answerability shortcuts in its old 128-row bank, so it is not adopted. Its replacement is an explicitly open 384-row/24-world diagnostic bank, not independent confirmation. Historical EAMI guidance reports ChatBubble `splits_onpanel` quarantined for overlap; the fresh ChatBubble quarantine metadata takes precedence. Do not restore historical proposed selection paths merely because they appear in an old audit.

The user confirmed that `Rineya_LCR_Full_v1_6000` is the intended dataset. The completed [P0 audit](evidence/dataset-acceptance-001/DATASET_ACCEPTANCE.md) binds manifest `65995575f13187ef78464e4120d1768696dd9114c7dfd8a379ed3ca6e28b266d` and records **STOP_NOT_ACCEPTED**. Local checks independently rederive all 5,400 permitted train/dev answers, reproduce exact file hashes, and qualify the native tokenizer with no context-window truncation. They also find 41 short-band violations, 392 shared name-blind sentence forms and a cheap answerability rule at 87.5% balanced accuracy. The producer's reported conditional balance failure remains unresolved, and protected contamination/holdout checks remain unverified because their payloads were not opened. These distinct findings are preserved in the receipt; no training proceeds on this candidate.

This documentation continuation did not read dataset payloads. Follow PHASE_II_CONTINUATION: binding P0 checks must pass and P1 must be frozen before any training, with current authority and resource checks also required. Preserve the exact open scout and future paired protocol rather than redesigning it after dataset/model outcomes.

## Resource boundary and current claims

SOURCE_REAUDIT records 2,313,621,504 bytes (about 2.15 GiB) available host RAM, below the 6 GiB preload floor; available GPU memory was 7,191,134,208 bytes (about 6.70 GiB) and disk 14,012,956,672 bytes (about 13.05 GiB). These are transient sampled observations, not reservations or memory-fit guarantees. The host guard alone blocks loading now. Refresh the source-bound preflight immediately before any later execution; do not lower the guard, silently change dtype, duplicate checkpoints in memory, evict unrelated processes, or switch devices to make the experiment start.

At this draft boundary, native scout A/B outcomes are unmeasured. No efficacy, complementarity, selector gain or cost-per-repair result follows from software fixtures. Software verification, native execution and open-dev/mechanism results need their own actual receipts. CONFIRMATORY CLAIM = FALSE and RELEASE CLAIM = FALSE.
