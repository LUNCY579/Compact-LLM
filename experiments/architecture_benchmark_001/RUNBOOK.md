# RAB-001 HOTFIX-001 — local rerun and recovery-engineering runbook

No GitHub Actions are required. All commands run synchronously in your own checkout. Use a new authorized branch/worktree and keep old results unchanged. This runbook supersedes commands embedded in preserved historical evidence.

## 1. Apply the delivery safely

The **full** ZIP contains the complete experiment folder. The **hotfix-only** ZIP contains only changed/new files, a before/after hash manifest, unified diff, and `apply_hotfix.py`. Both yield the same experiment contents; choose one, not both. No source/model checkpoints are bundled.

For a checkout whose experiment matches the uploaded ZIP, extract the hotfix ZIP outside the repository and run:

```powershell
python C:\Downloads\ced-hotfix\apply_hotfix.py --root C:\work\ChatBubble --check
python C:\Downloads\ced-hotfix\apply_hotfix.py --root C:\work\ChatBubble --apply
```

`--check` is read-only. `--apply` verifies all payload and input hashes before copying, backs up touched existing files outside the experiment folder, and does not invoke git or touch unrelated experiments. A locally changed touched file causes refusal. Resolve that difference in a new worktree; never use a forced overwrite. Files already at the hotfix hash are idempotently accepted. The manifest describes byte-exact changes, so a line-ending-converted checkout may require a deliberate merge rather than this exact-base overlay.

For the full ZIP, extract to a new worktree/copy and use its `experiments/architecture_benchmark_001` folder. Do not put backup Python files inside that folder: all Python source files there are implementation-hashed. Source weights and historical run outputs must remain outside the experiment package.

## 2. Install and qualify the runtime

From the repository root, use a clean Python environment. The delivered local mechanical tests used Python 3.13.5 and CPU PyTorch 2.10.0. Native dependencies are pinned in `requirements-native.txt`, including Transformers 4.51.3. The original official receipt used Python 3.11.15 on Windows; this is a new qualification, not a replay claim.

CPU example:

```powershell
python -m venv .venv-ced-hotfix
.\.venv-ced-hotfix\Scripts\Activate.ps1
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/architecture_benchmark_001/requirements-native.txt
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m pytest -q experiments/architecture_benchmark_001/tests --junitxml=ced-hotfix-tests.xml
python -m experiments.architecture_benchmark_001 status
```

For CUDA, install the appropriate official PyTorch 2.10.0 CUDA wheel instead of the CPU wheel, then use `--device cuda` consistently. There is no automatic fallback. A CPU G0 cannot authorize CUDA stages. Changing device, numerical backend flags, thread configuration, software versions or implementation invalidates old receipts. Do not enable TF32 or BF16 mid-run. The code retains an eager FP32 reference rather than adding an unqualified attention backend.

**Native tests must not skip.** This delivery could run neither native test module because Transformers/tokenizers were absent and package installation failed due to unavailable network resolution. Its successful tests do not qualify your native environment. See `verification/` for actual counts and logs. The two added native RoPE cases need your pinned environment as well.

## 3. Confirm the required local inputs

Set the real **ungated official source directory**, not a metadata-only GitHub path. It must contain every source-lock asset and the exact 1,192,135,096-byte `model.safetensors` file.

```powershell
$S = 'D:\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze'
$D = 'experiments/architecture_benchmark_001/data'
$R = 'D:\ced-runs\hotfix001-cpu'
python -m experiments.architecture_benchmark_001 audit --source "$S"
python -m experiments.architecture_benchmark_001 audit-evidence
```

The included calibration and smoke datasets are sufficient only for the existing bounded engineering exercise. They have pending human-review provenance, not demonstrated recovery-training quality. No external dataset download is required simply to rerun G0/G1 or the existing 25-step smoke.

The ZIP omitted the four old converted/trained model files. Existing local `G1_WIDTH`, `G2`, C28 and C40 checkpoint directories can still support read-only diagnostics, but copied receipt JSON alone is not enough to restore a model.

## 4. Cheapest useful diagnostic before rebuilding

For an old locally saved checkpoint, do not edit its config, receipt, lineage or weights. The diagnostic command accepts a checkpoint without treating its old receipt as current authorization:

```powershell
python -m experiments.architecture_benchmark_001 diagnose-numerics --source "$S" --checkpoint 'D:\old-runs\ced-c28' --arm C28 --device cpu --precision float32 --rows 8 --continuation-steps 8 --out 'D:\ced-diagnostics\c28-hotfix-fp32'
python -m experiments.architecture_benchmark_001 diagnose-numerics --source "$S" --checkpoint 'D:\old-runs\ced-c28' --arm C28 --device cpu --precision float64 --rows 8 --continuation-steps 8 --out 'D:\ced-diagnostics\c28-hotfix-fp64'
```

Repeat paired runs for C40 and ordinary D40. B28 can be diagnosed with `--arm B28` and no `--checkpoint`, which loads the exact official source. All outputs must be new directories.

FP64 preserves double precision through RMSNorm, RoPE and attention accumulation; it is not simply `.double()` followed by hidden FP32 downcasts. It consumes more memory and is an observation-only CPU-oriented diagnostic. FP64 success does not license FP32. Compare identical inputs/chunk schedules and ordinary controls; disappearance of a difference in FP64 supports a rounding hypothesis but does not prove there is no bug.

`NUMERICS.json` records elementwise and position violations, finite checks, argmax differences, maximum/mean error, top-1 margins, margin certificates, TV/KL, prefix isolation, and continuation comparisons. Continuations feed the same canonical next token to both paths. The first disagreement is the first free-running divergence; later comparisons are explicitly teacher-forced. This is not a reasoning benchmark.

These commands run **the new implementation** against old saved tensors. They do not recreate the exact old implementation and cannot upgrade old gate labels.

## 5. Fresh official qualification ladder

A fresh ladder is required to authorize this implementation because forward numerics, acceptance logic, receipt schema and implementation hashing changed. Do not reuse old PASS JSON files under a new digest.

```powershell
python -m experiments.architecture_benchmark_001.ladder --source "$S" --calibration "$D/calibration.jsonl" --out "$R" --device cpu
```

Default: five **qualification** stages—G0, G1_IDENTITY, G1_WIDTH, G1_CED_C28, G1_CED_C40—and **no optimizer**. This builds all four arms if prerequisites pass. B28 is the immutable source, not a copied converted checkpoint. Each stage runs in a separate foreground process so model memory is released between stages.

To reproduce the six-stage form including the legacy ordinary D40 engineering smoke, choose a different new root and explicitly add:

```powershell
python -m experiments.architecture_benchmark_001.ladder --source "$S" --calibration "$D/calibration.jsonl" --train "$D/train-smoke.jsonl" --include-smoke --out 'D:\ced-runs\hotfix001-six-stage' --device cpu
```

The extra stage is **25-step legacy G2**, not recovery. It is not mandatory merely to inspect CED conversion.

### Interrupted execution and resume

Repeat the exact command with `--resume`. Successful stages are reused only after their full current receipt DAG and artifacts validate. Failed/incomplete stages receive a new `attempt-0002` directory; earlier attempts remain intact. Changing inputs/runtime/code requires a new root, not a resume.

`RUN.json` freezes the session. `JOURNAL.jsonl` preserves starts, outcomes and interrupts. `LADDER-0001.json`, `LADDER-0002.json`, etc. summarize each invocation and contain the exact selected receipt paths. Stage logs live beside attempt directories. A zero exit code without a valid receipt is FAIL. C28 failure does not suppress an independent C40 attempt when its own parents pass. Any missing or failed expected stage makes the summary FAIL.

A hard-killed process may leave `ACTIVE.lock`. Verify no ladder/child process is still running before removing that one stale lock. Do not delete failed attempt directories to make history look clean. A failed review agent or usage-limit interruption never becomes an empty clean review.

### Strict failure is an informative stop

The new CED gate can legitimately fail again. It requires every scored element to meet `1e-4 + 1e-4*abs(expected)` and exact argmax, across the expanded grid. The global scaled bound is not authorization. `failure.json`, the failed G1 receipt, checkpoint and `numerics.json` remain available. Do not widen tolerances, rewrite PASS or launch CED optimization from a failed G1.

Next investigate paired controls, true FP64 and the first layer/path of divergence. If a different runtime equivalence criterion is scientifically justified, it requires a separately reviewed prospective protocol amendment and fresh qualification—not a silent patch selected to pass the observed values.

## 6. Lower-memory cached-teacher engineering, after fresh qualification

Use actual PASS paths from the latest ladder summary. The examples below assume first attempts succeeded:

```powershell
$G0 = "$R/G0/attempt-0001/G0.json"
$W = "$R/G1_WIDTH/attempt-0001/G1_WIDTH.json"
$C28 = "$R/G1_CED_C28/attempt-0001/G1_CED_C28.json"
$C40 = "$R/G1_CED_C40/attempt-0001/G1_CED_C40.json"
$E = 'D:\ced-engineering\hotfix001'
python -m experiments.architecture_benchmark_001 cache-teacher --source "$S" --calibration "$D/calibration.jsonl" --train "$D/train-smoke.jsonl" --g0 "$G0" --device cpu --out "$E/teacher"
python -m experiments.architecture_benchmark_001 smoke-cached --source "$S" --calibration "$D/calibration.jsonl" --train "$D/train-smoke.jsonl" --arm B28 --parent "$G0" --teacher-cache "$E/teacher/TEACHER_CACHE.json" --device cpu --out "$E/b28"
python -m experiments.architecture_benchmark_001 smoke-cached --source "$S" --calibration "$D/calibration.jsonl" --train "$D/train-smoke.jsonl" --arm D40 --parent "$W" --teacher-cache "$E/teacher/TEACHER_CACHE.json" --device cpu --out "$E/d40"
python -m experiments.architecture_benchmark_001 smoke-cached --source "$S" --calibration "$D/calibration.jsonl" --train "$D/train-smoke.jsonl" --arm C28 --parent "$C28" --teacher-cache "$E/teacher/TEACHER_CACHE.json" --control "$E/b28/G2_CACHED_B28.json" --device cpu --out "$E/c28"
python -m experiments.architecture_benchmark_001 smoke-cached --source "$S" --calibration "$D/calibration.jsonl" --train "$D/train-smoke.jsonl" --arm C40 --parent "$C40" --teacher-cache "$E/teacher/TEACHER_CACHE.json" --control "$E/d40/G2_CACHED_D40.json" --device cpu --out "$E/c40"
```

The last two commands are blocked if their strict CED qualification or matched control failed. Ordinary controls and CED candidates receive the same cached teacher targets, training rows/order, optimizer recipe, seed and 25-step budget. No winner selection or DEV access is involved.

`cache-teacher` saves the final normalized teacher hidden states and exact input IDs, not full-vocabulary logits. The original frozen tied head reconstructs full teacher distributions in chunks on CPU during training. The objective remains shifted CE plus `KL(teacher || student)` with T-squared and mask/packed-boundary handling. Student/teacher widths can differ. There is no top-K truncation, tail renormalization, synthetic response-only surrogate, or feature-matching objective. Projection batch shape/device can change rounding; mathematical objective equivalence is not a bitwise-logit guarantee.

Backbone and vocabulary-head checkpointing reduce retained activations. The teacher backbone is absent from the student loop; the CPU teacher head and its projection work are not free. Metrics count projections including recomputation, shared teacher preparation, supervised targets, optimizer state changes and actual CUDA peak allocation where applicable.

For 1M tokens, FP32 1,024-wide teacher states are about **4.096 GB** before token IDs and file overhead, versus **607.744 GB** for full 151,936-way FP32 logits. The frozen CPU head is approximately **593.5 MiB**. This is a storage calculation, not a measured performance result. The current shard-per-row layout and repeated artifact verification are intended for bounded engineering; a multi-million-token campaign needs separately reviewed consolidated shards/indexing, resumable optimization and throughput qualification.

Full FP32 parameters plus gradients alone are about 4.77 GB at 596M parameters. Adam-style two FP32 moment buffers would add about another 4.77 GB before activations and workspaces. This package does not quietly substitute Adam, QLoRA, BF16 or offloaded optimizers. The RTX 4070 8-GB fit must be measured; a fit claim is not warranted by the memory-saving design alone.

## 6b. Capability-retention qualification (HOTFIX-005)

`G1_QUALITY` scores an arm's output against the frozen teacher; every check it applies is
likelihood- or teacher-agreement-shaped, and those are measured to decouple from capability.
`G1_RETENTION_<arm>` and `G2_RETENTION_<arm>` score **answers** on the 150-row held-out probe
`data/monitor_probe.jsonl`, with the typed grammar in `retention.py`.

```powershell
$Q = 'outputs/quality-<digest8>'
$RT = 'outputs/retention-<digest8>'
# Step-zero retention. B28 IS the frozen source, so it takes no --checkpoint; every other arm
# must supply one, and it must hash to the model.safetensors its parent lineage declares.
python -m experiments.architecture_benchmark_001 qualify-retention --source "$S" --calibration "$D/calibration.jsonl" --parent "$Q/B28/G1_QUALITY_B28.json" --arm B28 --stage G1_RETENTION --device cpu --out "$RT/B28"
# Training-side. Needs the trained checkpoint, its training data, and the step-zero receipt
# whose reference it inherits rather than recomputes.
python -m experiments.architecture_benchmark_001 qualify-retention --source "$S" --calibration "$D/calibration.jsonl" --parent "$E/b28/G2_CACHED_B28.json" --arm B28 --stage G2_RETENTION --g1-retention "$RT/B28/G1_RETENTION_B28.json" --checkpoint "$E/b28" --train "$D/train-smoke.jsonl" --device cpu --out "$RT/B28-trained"
```

An arm whose `G1_QUALITY` receipt FAILs cannot hold a retention receipt at all — `validate_receipt`
refuses the parent. That is intended: a degenerate model has nothing to retain. It also means the
registered gate cannot be demonstrated on the arms it exists to reject, so
`experiments/rab001_exploration/retention_probe.py` runs the same package scorer outside the
receipt system for that purpose. It emits no receipt and opens no stage.

**Read `arm_answers_changed` before `status`.** A PASS on a checkpoint whose answers are
byte-identical to the source says the stage executed, not that anything was retained: the
registered 25-step engineering lane changes 0 of 150 answers.

Two limits to state when reporting: this is a finite-panel measurement, not a population claim,
and the separately designed 300-family confirmation certificate is **not** implemented or
licensed. Its 9/10 retention floor is also inert by construction — see
`protocol.json:G1_RETENTION.confirmation_min_retention_is_inert`.

## 7. Preparing recovery data is not opening G3

See RECOVERY_DATASETS.md for source choices and raw JSONL schema. The preparation command uses only local files and the official tokenizer:

```powershell
python -m experiments.architecture_benchmark_001 prepare-data --source "$S" --input 'D:\data\reviewed-recovery-pool.jsonl' --out 'D:\data\ced-recovery-v1' --max-tokens 512 --seed 17
python -m experiments.architecture_benchmark_001 recovery-readiness --plan experiments/architecture_benchmark_001/recovery_plan.example.json
```

The readiness result intentionally remains BLOCKED. A later G3 implementation must freeze reviewed recovery/regression/DEV/TEST manifests, source teacher, equal token/search budgets, training seeds, primary endpoint, margins, resource envelope, recovered-ordinary-donor plan and matched continuation accounting. Keep TEST sealed. A proposed 1–2M-token pilot is a budget proposal for that future registration, not a supported `--steps` override here.

## 8. What to retain from each run

Preserve the exact checkout/commit and implementation manifest, installed runtime versions, protocol/amendment, all data/source hashes, `RUN.json`, all journals/logs, successful **and failed** receipts, checkpoint files and lineages. Keep teacher cache shards and their original external G0 parent while they are in use. A copied `teacher_cache_manifest.json` inside a smoke directory is an evidence snapshot, not a relocated standalone cache; its relative paths still describe the original cache layout.

No repository changes were pushed by this delivery. Review and commit in your own new branch only after local verification; do not overwrite older experiments or published cores.
