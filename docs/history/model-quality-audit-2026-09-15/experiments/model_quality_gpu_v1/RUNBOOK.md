# Operator runbook

## 1. Checkout and prerequisites

Use a full checkout of `research/model-quality-targeted-gpu-v1`. Keep all earlier worktrees and branches unchanged. Native qualification/training require a clean checkout and a separately installed environment with `requirements.txt` pins. Install an official Torch 2.10.0 CUDA build appropriate to the host into THAT environment; do not alter the existing RAB or SupaExperiment environment. No package installation or download is performed by MQ-001.

```bash
python -m experiments.model_quality_gpu_v1 environment
python -m pytest experiments/model_quality_gpu_v1/tests -q
```

The two random native-Qwen cases explicitly skip if Transformers is absent. A test run containing those skips is not native qualification. The assistant implementation environment is CPU-only and lacks Transformers and official source assets.

For native work, stage regular, non-symlink copies of the seven locked Supa source assets. A model with the same name or dimensions is insufficient. Use the EXACT source weight SHA in `source_lock.json`. Native training does not accept Opallo, quantized weights, or a different Qwen checkpoint.

Use one GPU job at a time. The new cooperative lock does not control historical scripts or other agents. Check the host's real running jobs and available memory before starting. Do not kill someone else's job, lower the guard, or set up unattended retry loops. Pre-load guards require 6 GiB available host RAM and 6 GiB free GPU memory after imports; these are conservative prerequisites, not a tested maximum-allocation certificate.

## 2. Prepare explicit bounded data files

Prepare UTF-8 JSONL for TRAIN, DEV (at most 64 rows), and an **open historical exclusion bank**. Each row needs an ID, `template_family`, and one optional system + user + assistant exchange. Preserve source identifiers and reviewed family/scenario identity; do not invent disjoint IDs to disguise related scenarios. Extra metadata is retained in raw row hashes, but is not used for model-output-driven selection.

Example row shape (illustration, not an approved dataset):

```json
{"id":"scenario-01-a","template_family":"scenario-01","messages":[{"role":"user","content":"Answer with only the integer: how many objects are in a set of seven?"},{"role":"assistant","content":"7"}],"evaluation":{"scorer":"number","answers":["7"],"answerable":true}}
```

For missing-evidence literal tasks, use `evaluation={"scorer":"exact","answers":["UNKNOWN"],"answerable":false}` and actually request that contract in the prompt. Do not label generic policy refusal or any wrong answer as epistemic abstention. For prose, omit `evaluation`; the runner records response CE and raw outputs without pretending exact-match is semantic quality. New summary/structured-semantic scorers require independent qualification.

The compiler refuses malformed conversations, injected protocol delimiters, long rows, insufficient budget coverage, duplicate IDs, cross-split family/normalized/digit-blind prompt overlap. It does not silently truncate or discard rows. First create an explicitly documented short-row subset if the source corpus contains longer/multi-turn material. The first v1 native screen is at most 25k processed tokens, not the entire future quality pool.

Paths below are illustrative absolute paths. On Windows use quoted drive paths and the Python executable in the dedicated environment. Arguments are passed as subprocess lists, not shell interpolation.

```bash
python -m experiments.model_quality_gpu_v1 prepare \
  --source /absolute/Supa-Core-1.7 \
  --train /absolute/reviewed-data/train.jsonl \
  --dev /absolute/reviewed-data/dev.jsonl \
  --excluded /absolute/reviewed-data/open-history-exclusions.jsonl \
  --selection-seed 17 --order-seed 17 \
  --out /absolute/mq-runs/prepared-001
```

This emits `plan.json` and `DATA_REVIEW_PENDING.json`. It does not load model weights or start training. The plan holds full native token IDs, target boundaries, input hashes, selected IDs, ordering and fixed update groups.

## 3. Independently review the frozen plan

Review source rights, answer validity, EOS/response contracts, scenario/generator shortcuts and the role of the development set. Keep confirmation labels/outcomes out of the screen. Create a **new** review JSON from the pending template, binding the exact plan hash, and record `APPROVED`, reviewer, rationale and the four required checks only after that work is actually complete. The code does not fabricate this approval. Exact overlap checks are not a full leakage certificate.

Raw input files remain needed for qualification and training, which rehash/recompile the plan. Moving them requires explicit re-preparation/review because their paths are in the plan. Every cell of one matched screen uses that same plan, not a separately sampled dataset.

## 4. Native GPU qualification — zero optimizer steps

```bash
python -m experiments.model_quality_gpu_v1 qualify \
  --source /absolute/Supa-Core-1.7 \
  --plan /absolute/mq-runs/prepared-001/plan.json \
  --review /absolute/mq-runs/prepared-001/DATA_REVIEW_APPROVED.json \
  --out /absolute/mq-runs/qualification-001
```

The supervisor writes `launch.json`, `worker.log`, and `supervisor.json`; the worker records binding/resources and either `qualification.json` or `failure.json`. On success, the qualification is tied to this source, plan, review, code, checkout and GPU runtime. It opens only explicit execution of this bounded screen, not quality or release claims. Errors and timeouts are preserved, never overwritten.

Output directories should be outside the tracked checkout, or inside this package's ignored `outputs/` directory. Otherwise a new output file can correctly make the checkout dirty. The qualification and training must use the SAME commit; do not commit a documentation or evidence change between them and reuse the old qualification.

## 5. Execute one explicitly selected cell

```bash
python -m experiments.model_quality_gpu_v1 train \
  --source /absolute/Supa-Core-1.7 \
  --plan /absolute/mq-runs/prepared-001/plan.json \
  --review /absolute/mq-runs/prepared-001/DATA_REVIEW_APPROVED.json \
  --qualification /absolute/mq-runs/qualification-001/qualification.json \
  --arm response_1e-5 --confirm-training \
  --out /absolute/mq-runs/response-1e5-attempt-001
```

Only after inspecting each job should the next cell run. Replace `--arm` with the six names in README and always use a new output path. There is no automatic six-job launch or cloud dispatch. Native code uses source-independent source baselines in the SAME serving/runtime path, not stored historical panel numbers.

Checkpoint history is create-only and may need tens of GiB per cell. The preflight estimates required disk conservatively. Files contain training state shards, not a merged HF release. No cleanup deletes earlier evidence automatically.

## 6. Resume an interrupted attempt

Verify the old process is gone. If it was killed, its cooperative lock may remain; inspect the recorded PID and only then remove that stale lock manually. Do not remove a live job's lock. Choose its last directory containing a COMPLETE `manifest.json` and all hash-matched safetensors files.

Run the SAME cell/plan/qualification command into a NEW attempt directory, adding:

```bash
--resume /absolute/mq-runs/old-attempt/checkpoint-00004
```

The source is independently loaded again. Saved trainables, Adam moments, RNG and next-update cursor are restored only after validation. Final evaluation follows the actual last optimizer step. A failed or truncated attempt cannot be inserted into the completed-cell table.

## 7. Inspect outcomes without claiming an automatic winner

```bash
python -m experiments.model_quality_gpu_v1.analyze \
  /absolute/mq-runs/whole-1e5-attempt-001 \
  /absolute/mq-runs/response-1e5-attempt-001 \
  --out /absolute/mq-runs/development-comparison-001.json
```

Use one completed attempt per arm with an explicit retry-selection record, not whichever attempt scores best. The analyzer verifies hashes and shared bindings and reports development CE, paired correct gains/losses, false abstention and output cap hits. It refuses mismatched sources/plans/runtimes. It does not pool across seeds as independent questions, compute a release p-value, or promote an arm.

Proceed to replication and fresh confirmation only under a recorded decision policy. All extended methods and budgets in PLAN remain queued. Existing repair and behavior-screen experiments retain their own commands, prerequisites and evidence boundaries.
