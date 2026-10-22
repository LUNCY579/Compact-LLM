# RVS-001 — runnable verified-resampling experiment

**IMPLEMENTED / CPU-TESTED / TITAN NATIVE EFFICACY UNMEASURED**

This release continues the planning snapshot `88efa852689c162c131d9c698923d4327cdec853` on `research/rineya-verified-resampling-v1`. It implements an actual experiment, not merely a protocol: original centralized task generation, immutable public-only input export, exact-Titan native inference, independent sampling, label-blind selection, isolated oracle evaluation, and cost/failure accounting. No optimizer or training path is included.

The current implementation contract is [AMENDMENT_001_IMPLEMENTATION.md](AMENDMENT_001_IMPLEMENTATION.md). The original `plan.json`, `PROTOCOL.md`, roadmap, sources, and first delivery receipt remain historical planning records; their old implementation-status statements do not describe this release. No centralized E0/E1 file or source weight is modified.

## Immediately runnable CPU path

From the repository root, Python 3.11 or later:

```bash
python -m unittest discover -s experiments/rineya_verified_resampling_v1/tests -v
python -m experiments.rineya_verified_resampling_v1 demo --out experiments/rineya_verified_resampling_v1/runs/demo-001 --parents 128
python -m experiments.rineya_verified_resampling_v1 doctor
```

The demo runs **2,176 scripted completions** and writes JSON/CSV/Markdown reports. It exercises mechanics only. It neither downloads nor loads a language model. The `flat`, `minority`, `vote_gain`, and `all_correct` scenarios exercise no-headroom, selector failure, useful selection, and ceiling cases. Add `--scenario flat`, for example. Use a new output directory for every demo/report.

Tests use the standard library; installed Torch enables additional real CPU tensor/sampler tests. The full repository enables the original centralized generator integration tests. Skips are printed explicitly. See [receipts/CPU_VALIDATION.json](receipts/CPU_VALIDATION.json) for measured local scope, not a native success claim.

## Native Titan experiment: Windows PowerShell

Use your already pinned Titan interpreter, not an unrelated Python environment. Required versions are Torch `2.10.0+cu126`, Transformers `4.57.1`, safetensors `0.7.0`, and tokenizers `0.22.2`; psutil is also required. `doctor` reports installed metadata without loading the model. This release does not install packages, download weights, invoke APIs, or change your CUDA environment.

Fetch and switch to the dedicated branch without discarding local work. Native launch requires a clean committed checkout; output below is in this experiment's ignored `runs/` directory. Do not delete or force-reset unrelated files to satisfy that check.

```powershell
git fetch origin
git switch research/rineya-verified-resampling-v1
git pull --ff-only

$M = "experiments.rineya_verified_resampling_v1"
$R = "experiments/rineya_verified_resampling_v1/runs/titan-rvs-001"
$MODEL = "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Experimental-1.0.0-devfreeze"

python -m $M doctor
python -m $M prepare-centralized --out "$R/data" --per-family 32 --role DEV
python -m $M freeze --data "$R/data" --out "$R/study"
```

Review the data manifest and frozen `study.json`. Preparation calls the existing generator's role allocator and independent verifier, preserves original prompts, and renders **no CHECK/FINAL**. It does not require Torch or a tokenizer. The default seed selects a deterministic open-DEV subset, not a claim of new independent confirmation or exclusion from pretraining.

Then run inference-only qualification:

```powershell
python -m $M qualify --study "$R/study" --model-dir $MODEL --out "$R/qualification" --allow-model-load
```

Stop on a nonzero exit. Do not proceed unless `$R/qualification/qualification.json` says `PASS_NATIVE_INFERENCE_ONLY`. This is a new inference qualification, not the inherited E0 training command. It checks source hashes, strict BF16 loading/ties, frozen prompt tokenization, seeded replay, cached/full-prefix greedy and likelihood parity, and a maximum-sequence allocation probe. It performs **zero optimizer updates**. A qualified mechanical trace is not a quality result.

Start with a bounded smoke collection of the first 32 scheduled attempts:

```powershell
python -m $M collect --study "$R/study" --model-dir $MODEL --qualification "$R/qualification" --out "$R/run" --allow-model-load --allow-inference --max-attempts 32
```

`PAUSED` after those attempts is expected. Review the journal and supervisor receipts; do not tune decoding from these outputs. Continue the **same frozen experiment** without repeating completed attempts:

```powershell
python -m $M collect --study "$R/study" --model-dir $MODEL --qualification "$R/qualification" --out "$R/run" --allow-model-load --allow-inference --resume
python -m $M evaluate --study "$R/study" --run "$R/run" --oracle "$R/data/oracle.private.jsonl" --out "$R/report"
```

Collection automatically divides work into at most 128-attempt segments, with a 30-minute watchdog per segment and six-hour total collection wall budget. Every segment reloads the model and counts that overhead. This bounds sessions at the cost of extra loading time. The budget may stop before completion; it is not a prediction of runtime. A source/runtime/config/host change invalidates resumption/qualification. Do not change a budget after viewing outcomes and call the result the same frozen experiment.

If a hard interruption leaves a pending attempt, ordinary resume blocks. Inspect the evidence first. `--resume --abandon-interrupted` records the uncertain attempt as failed and advances without drawing a replacement. Torn/hash-invalid logs or missing supervisor cost receipts fail closed and require investigation; they are not auto-repaired. A report with failed or missing attempts remains `INCOMPLETE_BANK` with bounds, not a silently smaller sample.

## What the experiment measures

The default study uses 128 DEV parents (32 each arithmetic, text, sets, registers), one greedy attempt plus 16 independent temperature-0.7 samples per parent, and budgets 1/2/4/8/16. Temperature 0.7 is a fixed exploratory preset, not a tuned optimum. Maximum prompt/completion/total sizes are 896/128/1024 tokens. Longer context is not enabled in this release.

`REPORT.md`, `report.json`, `curves.csv`, and `cases.private.jsonl` distinguish:

- Greedy and first stochastic accuracy; prefix pool coverage and the complete-bank combinatorial pass@k estimate.
- Actual strict-JSON plurality and seeded random selection, matched selection loss, and conditional recovery.
- Paired repairs/damages, parent-cluster intervals, domain breakdowns, invalid outputs, and UNKNOWN abstention counters.
- Tokens/forward positions, generation and selection time, all segment wall costs, failures, and unobserved-work caveats.

Coverage increasing while vote accuracy stays flat identifies a selection problem. Flat coverage limits the tested proposal policy and budget, not all possible Rineya reasoning. Both increasing supports further study of the selected system, **not** a checkpoint improvement, broad preservation claim, or release approval. Oracle access is evaluation-only; it never enters the sampler or selectors. The current four-domain panel does not establish long-context retention or general-chat preservation.

## Resource and integrity boundaries

Native support is the registered **Windows CUDA Titan host only**. The CPU demo is portable. Preserve 7.4 GiB free host memory before loading, 6 GiB free host memory while running, 4 GiB GPU admission, 0.5 GiB ongoing free GPU, 4 GiB free disk, and the inherited external 4 GiB process-tree RSS watchdog. The 6 GiB quantity is NOT an RSS limit. The common centralized GPU lock prevents concurrent experiment use. No unrelated process is killed.

Source/template/model hashes and active runtime identities are recorded. Git's CRLF conversion is accepted only when normalized source bytes match the pinned Git blob; active-byte fingerprints still bind qualification/resume. Public exports deliberately omit reference targets and hidden domain labels. Hashes detect corruption/drift, not malicious rewriting by someone with access to all artifacts; the Python processes are not an adversarial security sandbox. Keep private oracle and report files local.

Not implemented: learned verifier, adaptive allocation, model portfolios, long-context or language-retention evaluation, RHD trajectory execution, training export, distillation, or a production service. They remain separate follow-ups. No Titan model execution occurred in the implementation environment; local tests used synthetic records and a tiny CPU Torch fixture.
