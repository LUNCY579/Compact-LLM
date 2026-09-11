# CED / RAB-001 review and implementation delivery

**Version:** HOTFIX-001, schema-2 evidence. **Review date:** September 10, 2026, America/Chicago.  
**Input:** uploaded `experiments.zip`, SHA-256 `452ca557f370ef0017c3abc6bf7edc608a4fbb75c13668009212b62dfd634193`.  
**Scope:** full experiment source, supplied datasets, protocol, lineage, saved receipts and documentation; read-only ChatBubble architecture README reference. No remote commits, branch changes, tags or Actions dispatches were performed.

## Bottom line

The upload is a coherent experimental snapshot, not obviously a half-written or corrupt folder. Its six historical receipt JSON hashes and included small artifacts verify. Four referenced checkpoint binaries are not included. The code matches its recorded implementation identity when its Windows path-key convention is reproduced. The historical gate-ladder markdown was already rewritten in the ZIP; this snapshot alone does not reveal exactly which instruction was interrupted next.

The important defect is not "CED is bad." It is that strict CED numerical failures were replaced with a looser post-result acceptance rule, and the accompanying causal/benignness claims outran the evidence. This delivery repairs acceptance, provenance and numerical diagnostics, then adds tested bounded recovery-engineering infrastructure. It does **not** claim those official CED failures are now resolved.

## What was independently checked here

| Check | Result and boundary |
|---|---|
| Original local mechanical suite | 103 passed, 2 native modules skipped; CPU only |
| Patched local mechanical suite | **162 passed, 2 native modules skipped**; zero test failures |
| Added coverage | Strict-vs-scaled counterexamples, true FP64, masks/prefix/chunks/continuation, hidden-head loss/gradient equivalence, checkpointed CED gradients, parent DAG corruption, data provenance/windowing, native-receipt discrimination, ladder resume/failure handling, cached teacher integrity and matched 25-step B28/D40/C28/C40 tiny-model smokes |
| Native Transformers execution | **Not run**: Transformers/tokenizers unavailable; attempted package installation failed because network name resolution was unavailable |
| Official Supa weights loaded here | **No**: not in the upload |
| Actual RTX 4070/CUDA testing | **No**: CPU-only execution environment |
| Original source/data/evidence preservation | Source lock and original data/evidence files preserved byte-for-byte |
| Historical official receipt JSONs | Six content hashes match; parent hash references present in the supplied set |
| Historical checkpoint verification | Four required model binaries missing; cannot fully validate/replay those artifacts |
| Four-agent external review workflow | Not resumed or reproduced here; its reported usage-limit failure is not a clean review |

The tests using tiny source/cache/receipt fixtures are temporary synthetic mechanics. They do not establish official-model G0/G1/G2 success, learned reasoning quality, or fit on the user's GPU. Loader unit tests do not count as native random-Qwen parity tests.

Final tested implementation digest: `eab80e9f99494268a113fba381f4178818aab759e3acbc6193ea936d33560fe6`. The exact manifest and runtime are in `verification/`; the digest includes Python source/tests, source lock, protocol, dependency requirements and declared amendment. Documentation outside the amendment and generated verification artifacts are not executable-source attestations.

## Evidence reconstruction

The uploaded historical implementation hashes under two path conventions are:

- POSIX keys: `7e2b3333fe3f04e76f0b49ed905c0107dfee08237c731ca4fe0dc86165438634`.
- Windows keys: `dc212c9fc032258a3d5e844d99b7f27a41b850327e0c06b432b3b7fa852cbd0c` — exactly the value in the historical receipts.

The difference is due to `str(relative_path)` using OS-dependent separators, not an established code/receipt mismatch. The hotfix uses POSIX keys everywhere while preserving raw file-content hashes and explicit runtime identity. It does not rewrite original receipts with the new digest.

The historical CED receipt fields record:

| Arm and comparison | Strict violating elements | Historical argmax |
|---|---:|---|
| C28 tail-only | 19 / 151,936 | matched |
| C28 chunked suffix | 4,114 / 2,734,848 | matched |
| C40 tail-only | 0 / 151,936 | matched |
| C40 chunked suffix | 1,251 / 2,734,848 | matched |

These are **recorded observations, not newly reproduced model runs**. Three of four comparisons fail the strict elementwise rule. The ordinary D40 comparison is documented in the historical markdown, but the uploaded material does not let this review independently execute it or prove the proposed shared-bank mechanism.

PyTorch documents that floating-point operation order and batched/sliced execution can yield different results [1]. That supports considering a rounding hypothesis; it does not prove this CED difference is harmless or excludes an implementation defect. Exact argmax on one prompt is not a guarantee across margins, longer prefixes, sampled decoding or new data. Likewise, this table is not evidence of inferior language-model quality: there is no matched held-out quality study here.

## Findings and implemented changes

### F1 — high: post-result relaxation of the CED gate

The old rule required exact argmax plus `max(abs(delta)) <= atol + rtol*max(abs(expected))`. A large unrelated logit can make near-zero coordinates pass that global rule even though their own per-coordinate tolerance fails. The strict violations were recorded but did not block the stage.

**Change:** require finite elementwise `abs(delta) <= 1e-4 + 1e-4*abs(expected)` and exact argmax. Preserve global bound/TV/KL/margins only as diagnostics. Use eight open pilot rows, token/chunk/halves schedules, tail-only, prefix and same-shape suffix-isolation checks. True FP64 and autoregressive-prefix diagnostics cannot issue qualification. No tolerance was enlarged. This is a declared forward-looking amendment, not retroactive success.

### F2 — high: parent digests were recorded, not recursively enforced

The original validator checked the current receipt's context/artifacts but did not walk the parent DAG. It also accepted arbitrary all-true check names rather than a mandatory per-stage schema. A claim that every actual ancestor had been reverified was too strong.

**Change:** schema-2 parents include actual relative receipt paths and hashes. Every ancestor must have the required stage, source/code/runtime/data identity, named checks and immutable artifact set. Wrong, absent, modified or stale ancestors block execution. Additional CED smoke parents enforce actual matched ordinary controls. Old schema-1 records are readable only as history.

### F3 — medium: native RoPE rounding mismatch and an incomplete FP64 lane

The original RoPE used the algebraically equivalent negative-power form instead of the pinned Qwen reciprocal-power construction. Its frequency rounding can differ, particularly at long absolute positions. RMSNorm, RoPE and attention accumulation also forced FP32, making a naïve `.double()` comparison incomplete.

**Change:** use reciprocal-power FP32 frequencies matching the intended native construction; preserve float64 intermediates throughout the diagnostic path. Tiny numerical tests pass. Two additional optional native tests cover 512-token inputs and long absolute positions. Native/official qualification remains unexecuted here, so this is not a claim that the RoPE change fixes the observed CED cache discrepancy.

### F4 — medium: digest portability and weak native-test counting

The original implementation hash used platform-specific path separators and omitted requirement-file changes. Mechanical native counting matched any class containing `test_native`, including loader-only fixtures.

**Change:** normalized paths, pinned requirements/amendment included in the implementation manifest; numerical/runtime/device identity expanded. Native qualification summaries require the actual padded and unpadded random-Qwen parity test names, not loader tests alone. Native skips remain explicit.

### F5 — medium: excess model copies and full vocabulary activation storage

Original identity expansion unnecessarily deep-copied the source before cloning retained/inserted blocks. C40 conversion loaded an unused B28 teacher before restoring its D40 donor. Conversion/reload could keep donor, converted and restored models alive together. Loss chunking accepted already-materialized full vocabulary logits.

**Change:** meta construction plus explicit independent clones for identity expansion, no redundant C40 B28 load, and earlier model release during CED save/reload. Added optional normalized-hidden output, checkpointed backbone blocks, and checkpointed vocabulary chunks. Full-vocabulary CE/forward-KL is retained mathematically; teacher/student widths may differ, and the teacher remains detached. Tests compare objective and gradients against the original full-logit formulation. These are memory-conscious changes, not measured official-model speed/VRAM gains.

### F6 — medium: bounded recovery preparation was missing

**Change:** offline per-row FP32 teacher-state cache with exact token/row/group/source/runtime hashes; frozen official CPU vocabulary head; 25-step cached-target smokes for all four arms; C28 requires matched B28 and C40 matched D40. Exact controls, optimizer/seed/data/cache fingerprints, finite/all-intended gradients, actual parameter updates, save/reload and projection recomputation costs are checked. The teacher backbone is not held in the student loop. These smokes are not G3 recovery.

### F7 — medium: interruptions and data handling needed explicit semantics

**Change:** a local process-isolated ladder records STARTED/PASS/FAIL/BLOCKED, preserves failed attempts, resumes only validated current PASS receipts and continues independent siblings. Zero exit without evidence is a failure. An exclusive session lock avoids accidental concurrent writes.

Data preparation now records immutable source/license declarations, assigns groups before windows, saves exact official token IDs, excludes cross-split exact duplicate windows, reports removed targets, and preserves optional source metadata. It rejects relabelled evaluation inputs and tokenizer drift. Human-review declarations, near-duplicate/benchmark screening and source license review still require real curation. No dataset is automatically downloaded.

### F8 — medium: source/readout qualification contained placeholder checks

**Change:** compute actual registered-context and tied/unmodified-native-readout checks rather than returning hardcoded true; use the official generation configuration's EOS set; check finite native/reference/generation logits and prevent generation from crossing the registered context budget. Preserve the native nonpersistent RoPE-buffer loader fix already present in the upload.

## What was deliberately not changed

The four shapes and budget comparison, source hashes, tokenizer, head_dim=128, dense causal shared bank, no decoder-local attention, donor choice, GQA grouping, residual basis and FFN growth policy remain intact. Old evidence/data are not rewritten. G3–G5 remain unimplemented and blocked; no larger training run, sealed DEV/TEST evaluation, quantization conversion, adapter insertion or model promotion is implied.

## Required next execution

1. Apply to a new local branch/worktree using the full folder or exact-base hotfix, not both.
2. Install the pinned native environment and run the full suite without native skips.
3. Use saved checkpoints for paired FP32/FP64 diagnosis if available, without upgrading old receipts.
4. Run a fresh official local qualification ladder. Preserve strict CED failures if they recur.
5. Only passing arms can enter the optional matched 25-step cached-teacher engineering lane.
6. Curate/freeze the broader recovery corpus, budgets, controls, margins and resource envelope before implementing G3. A filled readiness template alone stays blocked.

Detailed commands, retry behavior and CPU/CUDA cautions are in RUNBOOK.md. Public data acquisition and source choices are in RECOVERY_DATASETS.md. No external data is needed just to rerun the existing arm-qualification exercise.

## Remaining limitations and honest completion boundary

This delivery supplies a tested local implementation and guarded continuation path, not pretrained execution. The numerical discrepancy's cause is still undetermined; a shared-bank amplification mechanism is only a hypothesis. The 512-token registered context has not become a 32K campaign. The cached-target lane uses per-row shards and repeated integrity checks suited to bounded engineering, not a benchmarked million-token storage/throughput system. A scientific G3 runner still needs consolidation/resume, matched budgets, held-out evaluation and a reviewed optimizer/resource plan. The 8-GB GPU fit and actual recovery quality must be measured on your machine/source weights.

The external four-agent review did not complete in the supplied narrative. This review performed direct code inspection and the recorded local tests; it does not impersonate those four independent reviewers.

## Sources

[1] PyTorch 2.10 numerical accuracy: https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html

[2] PyTorch checkpointing documentation: https://docs.pytorch.org/docs/2.10/checkpoint.html

[3] ChatBubble intended architecture README, read-only reference: https://github.com/LUNCY579/ChatBubble/blob/research/benchmark-001-architecture-ced40-v1/experiments/architecture_benchmark_001/README.md

[4] Original uploaded code, `protocol.json`, `source_lock.json`, `evidence/OFFICIAL_GATE_LADDER.md`, six supplied gate receipts and local test outputs. The uploaded snapshot, not an older remote foundation README, is the primary code-review basis.
