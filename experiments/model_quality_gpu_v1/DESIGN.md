# Implementation contract and limits

## Isolation

All additions are under `experiments/model_quality_gpu_v1/`. There are no imports from historical command-line modules, no modifications to RAB's hashed root, its evidence, the repair registration, Opallo, CED or the sibling behavior screen, and no workflow/Actions changes. The reviewed RAB CLI mutates module globals and `sys.argv` during import; importing it as a library would create hidden execution coupling. MQ-001 therefore uses the public pinned native Qwen implementation directly and qualifies that lane independently.

## Source and runtime

`source_lock.json` pins exact Supa weight/config/generation/tokenizer hashes. Source audit rejects missing/symlink assets and extra weight formats, checks full source hashes before and after loading, verifies native tensor materialization/finiteness, the 596,049,920 unique parameters and tied vocabulary weights. No pickle checkpoint, remote custom code, download, architecture fallback or quantization fallback is allowed. Native `from_pretrained` owns tensor/buffer initialization; MQ-001 does not manually build a Qwen model on meta.

Use a dedicated Torch 2.10.0 / Transformers 4.57.1 / safetensors 0.7.0 environment. The local CPU tests do not certify this GPU stack. Qualification binds actual Python/platform/library versions, Torch build, CUDA build/device geometry, precision, visibility and CUBLAS settings. Native arithmetic is FP32 eager, deterministic algorithms, no TF32. This differs from some historical RAB environments; historical model scores and runtime receipts are not inherited.

## Objective masks and data

V1 deliberately supports only one user/assistant exchange, optionally preceded by a system message. Native template rendering must equal native generation-prefix + exact answer + EOS + newline. Tokenized generation-prefix must be an exact prefix of tokenized training text. The native full-template token API must reproduce the encoded IDs. Any merge across the boundary, unexpected suffix, embedded protocol delimiter or unsupported conversation shape causes refusal, not a heuristic mask.

The response mask includes the answer and EOS, not the assistant header, empty thinking stub or trailing newline. Labels are indexed by input position and shifted ONCE inside CE. The whole objective includes every next-token target. Masking prompt targets does not stop gradients from passing through prompt representations needed to predict responses.

Selection uses sorted IDs and a dedicated selection seed; ordering has a separate seed. The same full-row prefix is frozen for all cells before outcomes are observed. All candidate rows are tokenized and validated; oversized rows are not silently discarded. Selection stops before a row would exceed the processed-token cap and requires at least 90% coverage. Recorded actual tokens, not the nominal 25k ceiling, determine WSD progress. Each full row appears once. V1 does not support repeated epochs.

Update groups are immutable full-row groups with up to 2,048 processed tokens; no packing or cross-document attention occurs. Each microbatch is one row. CE sums are divided by the TOTAL supervised targets in the group, not by each row's length and then averaged. Whole/response denominators differ by definition, but examples, processed tokens and update boundaries do not. The last partial group is updated and evaluated, never lost.

Exact ID/family/normalized prompt/digit-blind cross-split checks are necessary but incomplete. Independent source-rights, labels, response-contract and generator/scenario review remains mandatory. Checksums do not authenticate a dishonest review. The active dataset is not built by this branch.

## Training resources

Activation checkpointing uses non-reentrant checkpointing with explicit arguments. Output-head CE is chunked in 64 selected token positions, with recomputation during backward. All vocabulary logits for the whole training sequence are never retained by this loss. This preserves the chosen token-sum objective, not historical mean-of-row behavior.

CPU-resident FP32 Adam moments use bounded per-parameter transfers. No custom disk optimizer is imported, and no disk-spill equivalence is asserted. This is deliberately slower than a highly optimized GPU optimizer; it is tested against PyTorch's non-foreach AdamW. GPU parameters/gradients remain FP32. For 314.6M trainables, CPU moments alone are approximately 2.34 GiB; GPU weight/gradient/activation requirements are separate. A 6-GiB free-host / 6-GiB free-GPU pre-load guard and 1.5-GiB / 0.5-GiB subsequent guard are prerequisites, not a measured fit guarantee. OOM refuses/fails without changing precision or moving another user's process.

Each cell runs alone in a foreground subprocess with a 3,600-second timeout. The lease is cooperative; old jobs that do not use it must be coordinated by the operator. No automatic stale-lock removal. After timeout/kill, verify the owned PID is gone before manually removing a stale lock. The supervisor kills only its own child tree. It cannot guarantee completion within the watchdog budget.

## Native qualification

A new approved data-review object, full clean dedicated-branch checkout, exact sources and matching runtime are required. Raw data are rehashed and the entire plan recompiled with the real tokenizer. The bounded native GPU probe checks hidden/head equivalence to native logits, cached-vs-full logits, actual response-mask CE agreement, finite trainable gradients with activation checkpointing, absence of frozen gradients, source-state identity and exact tail inventory. It performs no optimizer update and says nothing about task quality. Native numerical tolerance for cached/CE comparisons is atol=rtol=2e-4; hidden/head equality is exact.

Qualification is bound to plan/review/source/code/checkpoint/runtime identity. The output receipt is an integrity record, not a digital signature or defense against an actor fabricating results. A subsequent code or checkout change invalidates it. Native tests in this delivery remain unexecuted if their runtime is unavailable; no dummy native module fills that gap.

## Checkpoints and resume

Each trainable tensor has one safetensors file holding parameter, first moment and second moment, with explicit shape, name and file hash. The create-only COMPLETE manifest is written last and holds source/plan/arm/code/runtime binding, RNG and processed/supervised token/update cursors. At most one parameter's data are copied for serialization at a time. Frozen weights are reconstructed from the exact source and independently checked unchanged.

A checkpoint can only be saved when gradients are cleared at a completed optimizer boundary. There is no partial-gradient resume. Failed updates invalidate in-memory state; resume from the last COMPLETE checkpoint in a NEW attempt directory. Progress must equal the sum over the plan's completed groups. The validation pass checks the entire checkpoint before mutation; callers must keep checkpoint directories immutable during restoration. There is no malicious-concurrent-writer protection or claimed filesystem transaction across all tensors.

Checkpoint history is NOT automatically deleted. A full tail state (trainables + two moments) is roughly 3.52 GiB, excluding JSON/filesystem overhead. Disk is checked conservatively before training. Default checkpoint-every-four-updates may need tens of GiB per cell. The seven source assets are not embedded. These are training-state shards, not a ready HF release.

## Evaluation and analysis

Development generation uses the same native model/device, greedy decoding, full EOS list and at most 128 new tokens. Gold strings must fit that budget; long prompts are rejected. Raw predictions, generated IDs, termination and cap hits are retained. No CPU/GPU or BF16 comparison is silently interpreted as learning.

Only explicitly declared exact/numeric contracts produce accuracy; numeric parsing accepts a complete finite bare decimal, not a number found somewhere in prose. Open text has response CE and raw predictions but no invented semantic score. Literal UNKNOWN is exact; semantic abstention requires separate annotation/evaluation. The analyzer verifies artifact hashes and matched bindings, reports paired changes, and produces no pooled p-value or winner. Development CE must not be used alone to promote quality. Fresh confirmation, semantic evaluation and final serving/export qualification are outside the executable v1 boundary.
