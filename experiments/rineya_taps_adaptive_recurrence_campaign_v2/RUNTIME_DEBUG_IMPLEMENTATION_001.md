# Bounded runtime failure localization implementation

`runtime_debug.py` is a separate prospective GPU FP32 diagnostic worker. It loads no model on import and does not invoke or monkeypatch the frozen bulk runner, GPU runner, historical CPU entrypoint or dataset readers. The coordinator owns `runtime_debug_entry.py`, supervision, admission and the sole possible launch. This author has run only CPU fake fixtures; native qualification and error reproduction are **NOT_RUN**.

The implemented lane is `GPU_FP32_RUNTIME_DEBUG`; required admission/review statuses are `READY_FOR_RUNTIME_DEBUG_FP32_DISPATCH` and `PASS_RUNTIME_DEBUG_FP32_DISPATCH`. Successful completion is `RUNTIME_DEBUG_COMPLETE_NOT_EFFICACY`. This status means the entire bounded diagnostic population completed with its guards, not that the historical fault was explained or long-run reliability established.

## Exact population and counters

The three fixed bank IDs are dialogue-memory 21:v0 (interrupted case), 20:v1 (completed predecessor), 13:v1 (metadata-selected maximum), under nonce 9463250d48f1. Every case runs A0, A1, fixed alpha 1.1473425965417514 and original A3 once at seed 17, max 8 generated tokens. Fresh representative SMOKE is exactly the inherited mathematics and unanswerable IDs named in `NONNATIVEdebugproposal_001.md`; each supplies six legacy/new no-extra/unit/original-A3 parity evaluations and four policy evaluations, max 24 tokens. No A5, frozen random pool, label-informed fitting, full-bank restart or validation primary endpoint is accessed.

This is 12 diagnostic and 20 SMOKE physical evaluations, 32 total. The 576 generation-call cap plus ten maximum 357 attempts (nine complete plus one forced Python exception) and two cache rejects gives 588 model-API attempts. The worker enforces the cap before an additional model invocation, records attempts separately from completed model forwards, and refuses a successful terminal unless all 32 outcomes, ten maxima and two cache checks exist. Partial forward computation remains unknown; every failed operation and worker elapsed time is charged.

## Source, model and generation semantics

The worker checks the exact actual human allowance, immutable GPU 4c1a05c6/bulk 1ed52cc source, the reviewed dispatch closure, exact campaign bank/token hashes and canonical historical source/asset/runtime bindings. It inspects only the safetensors embedding header before loading, requiring 151,936 embedding rows; tokenizer length 151,669 is not the GPU indexing bound. It retains frozen/eval CUDA 0 FP32/eager, seed 17, deterministic algorithms, TF32 off, the custom UNKNOWN 51378/bias 5.156249985098839 head, one-shot L0–L2, and zero/one extra whole L12–L13 application. All floating parameters and buffers must be FP32 on CUDA 0, and the loaded/qualified/final healthy state must match the exact historical hash. On-disk assets and admitted source hashes are checked after qualification and at healthy completion. No training, downloads, cache, quantization, backend fallback or original guard changes exist.

The explicit generation loop preserves the frozen loop's chat rendering, full-prefix argmax/item on CUDA, CPU decode/output flags, canonical EOS, per-case cap and safety stops. It makes no additional model forward for first-logit diagnostics. SMOKE uses actual captured first-prefill hidden metrics, whole-trajectory RMS extrema and unmodified `armr.logit_metrics` against each policy's actual same-case A0 first logits. The unmodified `armr.reporting.summarize` applies its original norm, cosine, KL, score and validity gates. Legacy/reference parity rows use explicitly identified self readout references and independent exact value/token parity; they are not represented as A0 comparisons. Debug-case strict scoring happens only after generation and is archival, not a primary efficacy measurement.

## Fault localization and restoration

`operations.jsonl` writes and flushes START/COMPLETED/FAILED events for encoding, device transfer, CPU-held input verification, masks, model call, logit clone/finite check, CUDA sync, argmax/item, CPU decode/flags, resource guard, token tensor, concatenation and readout metrics. Context includes exact CPU-tracked prefix IDs, generated IDs at decode, shape/dtype/device/stride/storage offset, case/policy/seed and call index. IDs are checked on CPU before each CUDA prefix and after argmax, using embedding rows. The original missing 227-token continuation is not reconstructed; the first new prefix is the exact bound 226-token prompt, and any continuation is marked fresh greedy recomputation.

Metadata hooks return None and are removed under normal and exceptional exits. They validate observed automatic positions and retain layer entry/exit tensor metadata. Existing wrapper hooks/forward attributes are restored and checked against an actual pre-wrapper snapshot. Scalar events from generation and new-wrapper maxima are archived. Maximum qualification requires no-extra/unit/original-A3 exact value parity, forced-exception restoration, both cache-rejection variants, source/state preservation and no gradients before the twelve diagnostic evaluations.

Exceptions preserve a full Python traceback and first failing stage using CPU-held data. `DebugRecorder.synchronize` checks both a recorded CUDA fault and an active original-layer CUDA exception before inherited timer `finally` blocks could issue another sync. It records a skipped-after-fault event and calls no CUDA API in that path. No explicit final CUDA snapshot, state recovery or continuation occurs after a fault; resident state and final CUDA measurements are null. Healthy final measurements remain separately reported.

## Admission, deadline and human allowance

The child must receive `CUDA_LAUNCH_BLOCKING=1` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before the worker/Torch import. It rejects prior Torch import and missing environment pins. `RUNTIME_DEBUG_PARENT_STARTED_MONOTONIC` carries the coordinator's child-creation clock, so import, containment wait, loading and qualification consume the same 300-second parent budget. The worker stops body work at 275 seconds to retain 25 seconds for evidence. It also retains the inherited per-case wall limit at each resource boundary, including after a completed model call before an EOS can finish the case. Parent containment enforces the hard deadline independently. Fresh completed blocked SMOKE/maximum 357 timings, excluding the first cold call, must forecast the entire remaining 96-call diagnostic cap at the observed maximum×1.5 plus elapsed and 25 seconds; otherwise the worker refuses the registered cases rather than shrinking them.

Resource floors remain parent host 7 GiB/initial GPU free 6 GiB, worker host 6 GiB/RSS 6 GiB, peak GPU allocated 5 GiB, running host/GPU reserves 1 GiB. The coordinator additionally enforces campaign/disk reserves, exact GPU UUID/driver and one contained worker. Those are prospective gates, not measurements reproduced by this implementation author.

Both parent and worker require the exact actual campaign path `HUMAN_RUNTIME_DEBUG_ALLOWANCE_001.json` and its admission-bound hash. The worker checks exact payload fields: status `EXPLICIT_ADDITIONAL_ATTEMPT_AUTHORIZED`, lane, attempts 1, max native worker seconds 300, parent run `native-bulk-001`, exact reviewed worker hash, training False and full matrix retry False. A clearly labeled proposed-template wrapper cannot substitute for this path. The required actual allowance remains absent before direct human approval; the code author creates no authorization. Parent anti-replay checks the campaign ledger before child creation; the worker does not reject the parent record of its current authorized attempt.

## CPU verification and reproducible invocation

The author fixture suite initially passed 15 tests; the inherited SMOKE schema review correction adds a sixteenth test that invokes the unmodified real summarizer and verifies both a passing identity and a rejected cosine/KL divergence. A seventeenth fake-clock test verifies the unchanged per-case guard. Five affected tests pass after that correction. Tests cover frozen-loop token/EOS/cap/prefix parity, failures at model/argmax/item/guard/tensor/cat/sync stages, full traceback/CPU-prefix preservation, header-bound ID/position rejection, explicit CUDA device/cache/API limits, normal/forced hook/forward restoration, complete forecast refusal, proposed-template rejection, and suppressed postfault timer synchronization. No native model or GPU inventory is used.

CPU apparatus command (existing interpreter remains read-only):

```powershell
& 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe' -B -m pytest experiments/rineya_taps_adaptive_recurrence_campaign_v2/tests/test_runtime_debug.py -q -p no:cacheprovider
```

After source review, a new frozen admission/review packet and direct human allowance, only the coordinator may execute the supervised entry:

```powershell
& 'C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe' -B -m experiments.rineya_taps_adaptive_recurrence_campaign_v2.runtime_debug_supervisor --source 'C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-A1.1-1.1.0-devfreeze' --out 'experiments/rineya_taps_adaptive_recurrence_campaign_v2/runs/native-runtime-debug-001' --admission 'experiments/rineya_taps_adaptive_recurrence_campaign_v2/runtime_debug_admission_v1.json'
```

This is an executable entry specification, not permission to run it. The initial infrastructure retry is exhausted. A clean isolated probe may be nonreproduction caused by shorter history or blocking/timing/allocator perturbation; it does not clear the previous illegal-access cause or authorize an 18,000-second matrix retry. Any later retry needs its own safety disposition, complete forecast, source/qualification review, exposure treatment and human attempt allowance.
