# TGSA v5.1 — Stage 0 apparatus qualification: execution plan

**Status:** PREPARED, NOT RUN. Blocked on one human action (§2).
**Branch:** `research/tgsa-v5-1-unified-programme` @ `b60d470` — on origin, CI green (run `33889037837`), 458 tests passing
**Worktree:** `C:/Users/clclc/BusinessProject/Chat-Bubble-v51`
**Plan:** `Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/configs/phase1_v5_plan.json`
**Contract:** `.../configs/v5_1_programme_contract.json` (`programme_sha256 d52d89a3020972dd…`)
**Written:** 2026-09-03. All timings measured on this machine, this date.
**Revised:** 2026-09-04 — the gate-hardening work was merged fast-forward into the
registered branch (`fd6e20f` -> `b60d470`), so the probe and qualification below now run
against the strengthened apparatus. The frozen plan bytes are unchanged
(`0538faf4d2ad142dbce9c408910f9dcbcf1fd064`) and `programme_sha256` is unchanged at
`d52d89a3…`; the added gate is registered in `PHASE1_V5_PREREGISTRATION.md` §7 and
lives in code, not in the plan, which is why the registration bytes could stay frozen.

---

## 1. What Stage 0 is, and what it can never establish

Stage 0 qualifies the *apparatus*, not the mechanism. Per §4 of
`V5_1_UNIFIED_PROGRAMME.md`, a `GO` means only that the scratch apparatus is
competent, non-saturated, and stable enough for a **separately preregistered**
mechanism experiment. No MTP objective is present (`mtp.enabled: false`).

The plan's own `claim_ceiling` is binding:

> May establish only that a null-anchored synthetic regime is competent,
> non-saturated, reproducible, and suitable for a separately preregistered
> MTP-free mechanism comparison. It is not evidence for or against TGSA, does
> not validate Rineya, and supports no quality, latency, or wall-clock claim.

`mechanism_test_authorized: false`, `mechanism_semantics: "none"`,
`confirmatory: false`. Nothing here may be read as a TGSA result in either
direction.

---

## 2. The single blocking gate — a human action, not an engineering one

`phase1_v5_plan.json` carries `development_only: false`. It is the *registered*
plan, so `scripts/phase1_v5_1_qualify.py` refuses to run it unless all three
hold:

1. `--plan` resolves to exactly `repo_root/<SOURCE_V5_PLAN_PATH>`;
2. the plan's git blob sha1 equals the locked `SOURCE_V5_PLAN_BLOB`
   (`0538faf4d2ad142dbce9c408910f9dcbcf1fd064` — verified present and matching);
3. `--attest-no-prior-outcome-inspection` is passed.

Absent (3) it raises `V51ContractError`. **That flag records a human operator's
attestation that no nondevelopment outcome produced under the locked v5 plan was
inspected before the prospective v5.1 lock. I have not passed it and must not.**

The contract must be frozen *before* any nondevelopment run. Running first and
locking afterwards destroys the prospective registration irrecoverably. Every
run described in §6 below was `development_only=true` and produced no outcome
under the registered plan.

**To start Stage 0, the operator adds `--attest-no-prior-outcome-inspection` to
the commands in §3 and §4 — and only if that attestation is truthful.**

---

## 3. Prerequisite probe

Fixed, one-hop, `cascade` only. From `plan.probe`:

| field | value |
|---|---|
| arm | `cascade` (only) |
| family | `pointer_chase` (primary; one hop) |
| seeds | 17, 29, 43 |
| exposure `P1` | batch 64, 50,000 unique train examples, 1,280,000 presentations |
| **steps/cell** | **1,280,000 / 64 = 20,000** |
| dev / test | 3,000 / 6,000 |
| dataset seed | 2026090202 |

Capacity ladder, climbed in order, **stopping at the first passing rung**:

| rung | embed_dim | heads | layers | trajectory_rank |
|---|---:|---:|---:|---:|
| C1 | 96 | 6 | 2 | 16 |
| C2 | 96 | 6 | 4 | 16 |
| C3 | 192 | 6 | 4 | 16 |

### Pass rule (`plan.probe.thresholds`, all must hold)

- `minimum_margin_over_family_null: 0.15` — the **weakest** seed must clear the
  family null by ≥ 0.150
- `maximum_mean_accuracy: 0.95` — mean accuracy ≤ 0.95 (anti-saturation)
- `maximum_seed_coefficient_of_variation: 0.35` — seed CV ≤ 0.35
- `maximum_generalization_gap: 0.10`
- `minimum_seed_runs: 3`

The plan validator independently refuses any weakening of these: the null margin
may not go below 0.150, the saturation ceiling may not exceed 0.95, the CV bound
may not exceed 0.35, the gap bound may not exceed 0.10.

### Commands

Run C1 first. Only if it fails, run C2; only if that fails, C3.

```bash
cd C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1
```

```bash
PYTHONPATH=C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe scripts/phase1_v5_1_qualify.py --contract configs/v5_1_programme_contract.json --plan configs/phase1_v5_plan.json --stage probe --rung C1 --output-dir C:/Users/clclc/tgsa-v5-stage0/probe_C1 --repo-root C:/Users/clclc/BusinessProject/Chat-Bubble-v51 --attest-no-prior-outcome-inspection
```

Then verify, and only read the verified report:

```bash
PYTHONPATH=C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe scripts/verify_phase1_v5.py --plan configs/phase1_v5_plan.json --report C:/Users/clclc/tgsa-v5-stage0/probe_C1/probe_report.json --output C:/Users/clclc/tgsa-v5-stage0/probe_C1/verification.json
```

Substitute `C2` / `C3` and a matching fresh `--output-dir` for later rungs.
`evidence_policy.overwrite` is `forbidden` — always use a fresh root.

### If C3 fails

**The task family is rejected and the question returns to task design.** Do not
invent a fourth rung, do not widen the ladder, do not relax a threshold. A
`NO_GO` here is a legitimate scientific outcome and exits `0`.

---

## 4. Full qualification

Runs at the **frozen** rung from §3. From `plan.registered_arms` and
`plan.exposure_grid`:

- arms: `cascade`, `history_control`, `learned_delta`
- seeds: 17, 29, 43
- exposures: 4 cells, all required (`require_complete_exposure_grid: true`)

| exposure | unique train | presentations | steps/cell |
|---|---:|---:|---:|
| E1 | 50,000 | 1,280,000 | 20,000 |
| E2 | 200,000 | 1,280,000 | 20,000 |
| E3 | 50,000 | 2,560,000 | 40,000 |
| E4 | 200,000 | 2,560,000 | 40,000 |

**3 arms × 3 seeds × 4 exposures = 36 cells.**
Steps per (arm, seed) = 20,000 + 20,000 + 40,000 + 40,000 = 120,000.
Total = 9 × 120,000 = **1,080,000 optimizer steps.** ✔ matches the briefing.

One invocation covers one `(rung, exposure)` and iterates all 3 arms × 3 seeds
internally — **9 cells per invocation, so 4 invocations**, one per exposure.

### 4.0 First freeze the band — the qualification will NOT run without it

`--frozen-band` is **mandatory** for the registered plan.
`phase1_v5_runner.py::_validate_frozen_band` returns early only when
`development_only` is true; for the registered plan a missing `--frozen-band`
raises

```text
V5ContractError: non-development qualification requires --frozen-band
produced from the passing prerequisite probe
```

before any output is created. Produce the band from the probe rungs you actually
ran — **one `--report` per rung from `C1` up to and including the passing one, in
ladder order, and none after it**. If `C1` passes, that is a single report:

```bash
cd C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 && PYTHONPATH=$PWD C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe scripts/select_phase1_v5_1.py --contract configs/v5_1_programme_contract.json --repo-root C:/Users/clclc/BusinessProject/Chat-Bubble-v51 capacity --plan configs/phase1_v5_plan.json --report C:/Users/clclc/tgsa-v5-stage0/probe_C1/probe_report.json --output C:/Users/clclc/tgsa-v5-stage0/frozen_band.json
```

Note the argument order: `--contract` and `--repo-root` are **global** and come
*before* the `capacity` subcommand.

> **The ladder is not skippable, and a failing rung's report is still required.**
> Do **not** simply point `--report` at whichever rung passed.
> `freeze_band_after_probe` walks the registered ladder `C1 -> C2 -> C3` from the
> *first* rung and raises
>
> ```text
> V5ContractError: probe report for rung C1 is missing; rung selection must
> follow the registered order
> ```
>
> the moment a rung has no report. A failing rung's report is the evidence that
> the climb was registered, so it must be supplied. A `NO_GO` rung yields
> `decision: ADVANCE_CAPACITY_RUNG`, which is deliberately *not* one of the
> loop's break conditions (`REJECT_SATURATED_BAND`, `REJECT_TASK_FAMILY`), so the
> loop advances and demands the next rung's report too. If `C1` fails and `C2`
> passes, pass **both**, in order:
>
> ```bash
> cd C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 && PYTHONPATH=$PWD C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe scripts/select_phase1_v5_1.py --contract configs/v5_1_programme_contract.json --repo-root C:/Users/clclc/BusinessProject/Chat-Bubble-v51 capacity --plan configs/phase1_v5_plan.json --report C:/Users/clclc/tgsa-v5-stage0/probe_C1/probe_report.json --report C:/Users/clclc/tgsa-v5-stage0/probe_C2/probe_report.json --output C:/Users/clclc/tgsa-v5-stage0/frozen_band.json
> ```
>
> Symmetrically, a report *after* the first pass is refused — `probe reports
> exist after the first passing rung; the registered ladder must stop at first
> pass` — so never include `C3` once `C2` has passed. `--report` is
> `action="append"` on both the `capacity` and `exposure` subcommands
> (`scripts/select_phase1_v5_1.py:79,85`), so repetition is supported; this was a
> documentation defect, not a missing capability.
>
> **Why no test caught this.** For the registered plan
> `freeze_band_after_probe` raises `development-only probe reports cannot freeze
> a real band` (`tgsa/phase1_v5.py:1075`) *before* the ladder loop is entered, so
> no development-mode run can reach the missing-report check. Every
> `freeze_band_after_probe` test in `tests/test_phase1_v5.py` uses `C1` as the
> passing rung, so the "`C1` failed, a later rung passed" path has no coverage at
> all. Found by the 2026-09-05 pre-flight audit; same shape as the missing
> `--frozen-band` defect found on 2026-09-04.

`--report` takes the *report* path, but `_load_registered_report` also reads two
**siblings in the same directory** and fails if either is missing or fails:
`verification.json` (which is why §3's verify step must write exactly that name)
and `v5_1_registration_verification.json` (written automatically by the qualify
wrapper). For the registered plan it additionally requires
`operator_attestation.accepted is true`, so a band cannot be frozen from a run
made without the attestation.

The band binds `plan_sha256` and `selected_rung.id`, so a qualification run on a
different rung, or against an edited plan, is refused.

### 4.1 Run the four exposures

```bash
cd C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 && PYTHONPATH=$PWD C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe scripts/phase1_v5_1_qualify.py --contract configs/v5_1_programme_contract.json --plan configs/phase1_v5_plan.json --stage qualification --rung <FROZEN_RUNG> --exposure E1 --frozen-band C:/Users/clclc/tgsa-v5-stage0/frozen_band.json --output-dir C:/Users/clclc/tgsa-v5-stage0/qual_E1 --repo-root C:/Users/clclc/BusinessProject/Chat-Bubble-v51 --attest-no-prior-outcome-inspection
```

Then verify it, writing to the sibling name the selector expects:

```bash
cd C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 && PYTHONPATH=$PWD C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe scripts/verify_phase1_v5.py --plan configs/phase1_v5_plan.json --report C:/Users/clclc/tgsa-v5-stage0/qual_E1/qualification_report.json --output C:/Users/clclc/tgsa-v5-stage0/qual_E1/verification.json
```

Repeat both for `E2`, `E3`, `E4` with matching fresh output dirs. All four are
required (`require_complete_exposure_grid: true`), and they are independent
invocations — this is the level at which the 3-worker concurrency applies.

### 4.2 Select

By the **frozen resource-order rule**, never by the most favourable arm
contrast. `plan.qualification.selection_policy`, in order:

1. `minimum_sample_presentations`
2. `maximum_unique_train_examples`
3. `lexicographic_exposure_id`

Selection reads **verified reports only** (`evidence_policy.selection`) — the
same sibling-file checks as §4.0 apply to each of the four.

```bash
cd C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 && PYTHONPATH=$PWD C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe scripts/select_phase1_v5_1.py --contract configs/v5_1_programme_contract.json --repo-root C:/Users/clclc/BusinessProject/Chat-Bubble-v51 exposure --plan configs/phase1_v5_plan.json --frozen-band C:/Users/clclc/tgsa-v5-stage0/frozen_band.json --report C:/Users/clclc/tgsa-v5-stage0/qual_E1/qualification_report.json --report C:/Users/clclc/tgsa-v5-stage0/qual_E2/qualification_report.json --report C:/Users/clclc/tgsa-v5-stage0/qual_E3/qualification_report.json --report C:/Users/clclc/tgsa-v5-stage0/qual_E4/qualification_report.json --output C:/Users/clclc/tgsa-v5-stage0/selected_exposure.json
```

---

## 5. Costed wall-clock estimate

### Hardware and method

NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB. torch `2.12.1+cu126`, CUDA 12.6,
cuDNN 91002, Python 3.11.15, Windows. Interpreter:
`C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe` — note the
*default* `python` on PATH is a CPU-only torch and would run this ~an order of
magnitude slower; **use the venv explicitly.**

Timings came from development-only plan variants (`development_only: true`,
`study_id` prefixed `tgsa-phase1-v5-timing-devonly-…`) that keep the real
capacity ladder, batch size 64, dataset families/weights/vocabulary and the real
3,000/6,000 dev/test splits — so sequence length (max 44 tokens) and evaluation
cost are representative — and shrink only the step budget. Per-cell durations
were derived from consecutive `completed_at_utc` stamps; the first cell of each
invocation carries a one-off CUDA warm-up (+2.7 to +4.1 s) and was excluded.
Per-step cost was fitted over a 300 → 1,500 step lever for C1/`cascade`
(0.04712 s/step) and a 100 → 300 lever for the rest, scaled by the ratio
K = 1.0607 between the two levers for C1 — the short lever underestimates by
~6 %.

### Measured per-step cost (single worker, seconds/step)

| rung | probe `cascade` | qual `cascade` | qual `history_control` | qual `learned_delta` |
|---|---:|---:|---:|---:|
| C1 (96×2) | 0.04712 | 0.04712 | 0.05908 | 0.06046 |
| C2 (96×4) | 0.09557 | 0.09053 | 0.11970 | 0.12590 |
| C3 (192×4) | 0.09620 | 0.09416 | 0.12296 | 0.12240 |

Two structural findings, both material to planning:

- **Cost tracks depth, not width.** C2 (96×4) and C3 (192×4) are within 1 % of
  each other, while C1 (96×2) is half. At this scale the GPU is kernel-launch
  bound, not FLOP bound — the model is far too small to occupy a 4070. Doubling
  `embed_dim` is therefore effectively free; doubling `num_layers` is not.
- **The two control arms cost 25–39 % more per step than `cascade`.** The spread is +25.4 % to +39.1 % across the three rungs; the widest is C2 `learned_delta` at 0.12590 against 0.09053.

Fixed overheads: dataset generation 4.85 s at 50,000 train examples, 16.35 s at
200,000; evaluation ≈ 2.5 s per cell (1,500 seen + 3,000 dev + 6,000 test at
eval batch 256).

### Estimates

**Prerequisite probe** (3 seeds × 20,000 steps, `cascade` only):

| rung | per-rung | cumulative if it freezes here |
|---|---:|---:|
| C1 | 0.79 h (47 min) | **0.79 h** |
| C2 | 1.60 h (96 min) | **2.39 h** |
| C3 | 1.61 h (97 min) | **4.00 h** |

**Full qualification** (36 cells, 1.08 M steps), by frozen rung:

| frozen rung | qualification |
|---|---:|
| C1 | **16.71 h** |
| C2 | **33.65 h** |
| C3 | **33.99 h** |

**Stage 0 total.** The concurrency factor applies to the **qualification only** —
the probe cannot be sharded (see below), so its serial time is added unscaled:

| probe freezes at | probe (serial) | qualification ÷ 2.05 | total, 3 workers | total, 1 worker |
|---|---:|---:|---:|---:|
| C1 | 0.79 h | 8.11 h | **8.90 h** | 17.50 h (0.7 d) |
| C2 | 2.39 h | 16.33 h | **18.72 h** | 36.04 h (1.5 d) |
| C3 | 4.00 h | 16.50 h | **20.50 h** | 37.99 h (1.6 d) |

### Concurrency

Because the workload is launch-bound, concurrent invocations help. Measured:
3 concurrent processes slow each cell from 15.24 s to 22.26 s but run three at
once — **2.05× net throughput** (3 × 15.24 / 22.26 = 2.0539). **6 concurrent processes fail with CUDA OOM**
(each process reserves ~1.2 GB of CUDA context on an 8 GB card; 2 of 6 died).
**Use at most 3, and shard by exposure** (E1–E4 are independent invocations),
which also keeps each evidence root separate as `evidence_policy` requires. With
E3/E4 (40,000 steps) started first and E1/E2 (20,000) filling in behind, three
workers give a balanced schedule and the measured ~2.05× holds end to end.

**The probe cannot be sharded.** One invocation runs all three seeds
sequentially inside a single process (`for arm in arms: for seed in seeds:` with
`arms=('cascade',)`), and `phase1_v5_1_qualify.py` exposes no per-seed or
per-cell flag — only `--contract/--plan/--stage/--rung/--exposure/--frozen-band/`
`--output-dir/--repo-root/--no-resume/--attest-…`. The ladder is sequential by
design too, so the rungs cannot be run in parallel either. Probe time is
therefore serial and must not be divided by the concurrency factor.

Caveat: these are point estimates from a quiet machine (the GPU already had
~800 MiB and intermittent third-party load). Treat them as ±15 %, and note the
4070 Laptop will thermally throttle over a multi-hour run, which pushes the real
figure toward the upper end.

---

## 6. Resume / checkpoint path — CONFIRMED WORKING, BIT-EXACT

`training.checkpoint_interval_steps: 500`. Verified end to end, development-only:

1. Ran probe C1, 1,500 steps × 3 seeds, to completion as a reference
   (226.4 s); recorded each cell's `model_state_sha256`.
2. Re-ran into a fresh root and **hard-killed** the process (`TerminateProcess`,
   which does *not* run the runner's exception handler) once a periodic
   `training_state.pt` appeared. It was killed with `next_step = 500`, leaving a
   cell directory containing only `training_state.pt`.
3. Re-invoked the identical command. It resumed at step 500
   (`resumed_from_step: 500` in the cell manifest) and completed.
4. **All three cells' `model_state_sha256` were identical to the clean reference
   run.** Resume is bit-exact, not merely functional.

Mechanism, for the operator's confidence:

- state is **fingerprint-bound** (`execution_fingerprint = canonical_sha256(identity)`
  over plan hash, stage, rung, exposure, dataset manifest, arm, seed, git and
  environment); a mismatched state is refused rather than silently reused
- the deterministic batch stream is re-seeded from `start_step`, so the training
  order after resume matches an uninterrupted run
- completed cells are detected and returned without recompute, so re-running an
  interrupted 36-cell sweep only redoes the unfinished cell
- state is also saved on **any** `BaseException`, so a Ctrl-C or an OOM
  checkpoints on the way out
- `--no-resume` refuses to overwrite a non-empty cell directory

**Operational consequence:** the 8.90–20.50 h sweep at three workers (17.50–37.99 h at one) does not need to survive in
one sitting. Worst case, a crash loses < 500 steps of one cell (≈ 24–63 s).

---

## 7. Hard constraints for whoever executes this

- **Do not pass `--attest-no-prior-outcome-inspection` on anyone's behalf.** It
  is the operator's attestation.
- **A scientific `NO_GO` is a successful run and exits 0.** Do not treat it as a
  failure; do not retry it into a `GO`. A *software* failure must not be dressed
  up as a `NO_GO` — check `software_success` and `status` in the report.
- **Do not invent a fourth capacity rung.** If C3 fails the probe, the task
  family is rejected and the question returns to task design.
- **Do not weaken, skip, xfail or delete any gate, assertion or verifier.** The
  plan validator will refuse most of it anyway.
- **Never pool Phase 2A and Phase 2B evidence** (`sequential_teacher_forced_recursive_v1`
  vs `parallel_independent_offsets_v1`). Not applicable inside Stage 0, but it
  governs everything downstream.
- Fetch with `--tags` before any verifier run: `tgsa/v5_1_programme.py` resolves
  the six pinned SHAs with `cat-file -e` and `merge-base`, never by branch name,
  and a v5.1 CI run has already failed on exactly this after v4 was retired.
- Keep the worktree **clean**: `verify_repository_contract` fails closed on a
  dirty tree *before* creating any output, and it runs at the top of every cell
  invocation.
- Write every evidence root **outside** the repo (the commands above use
  `C:/Users/clclc/tgsa-v5-stage0/…`) so the clean-worktree gate keeps passing.
  Use short paths — this machine hits Windows `MAX_PATH` on deep checkouts.

---

## 8. Pre-flight checklist

```bash
cd C:/Users/clclc/BusinessProject/Chat-Bubble-v51 && git fetch origin --tags --prune && git status --porcelain
```

Expect empty output. Then:

```bash
PYTHONPATH=C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/scripts/verify_v5_1_programme_contract.py --contract Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1/configs/v5_1_programme_contract.json --repo-root C:/Users/clclc/BusinessProject/Chat-Bubble-v51 --output C:/Users/clclc/tgsa-v5-stage0/preflight.json
```

Expect exit `0` and `"passed": true` with all ten checks green. Confirm the GPU
is idle (`nvidia-smi`) and that the venv reports `cuda_available: true`.

Then validate the plan against its schema. Do this explicitly: `fd6e20f` fixed a
typo in the schema's `required` array (`famili_roles` → `family_roles`) that had
left `family_roles` silently unenforced, so this check only became meaningful at
that commit.

```bash
cd C:/Users/clclc/BusinessProject/Chat-Bubble-v51/Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1 && PYTHONPATH=$PWD C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe scripts/validate_json_artifact.py --schema evidence/phase1_v5_plan.schema.json --document configs/phase1_v5_plan.json
```
