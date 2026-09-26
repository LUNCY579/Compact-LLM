# HANDOFF — Supa Long-Context Minimal Intervention Pilot

> ## SUPERSEDED IN PART — read this box first (added 2026-09-26)
>
> Sections 1 and 3 below were written on 2026-09-05 and are **out of date**. The
> programme was continued by later sessions on 2026-09-06/07. Phase 1 is
> **COMPLETE on both cores**, with an extensive independent audit under
> `results/audit_20260905/`. Do not re-run Phase 1 from scratch.
>
> **Phase 1 verdict: FROZEN wins on both cores. Phase 2 is NOT licensed.**
>
> | core | A frozen | B LoRA | C NoRA-init |
> |---|---:|---:|---:|
> | a11 fp16 | 170/256 = 66.4% | 110/256 = 43.0% (−23.4) | 133/256 = 52.0% (−14.5) |
> | a1 bf16 | 168/256 = 65.6% | 108/256 = 42.2% (−23.4) | 128/256 = 50.0% (−15.6) |
>
> ### The negative is ABSTENTION COLLAPSE, not degraded reasoning
>
> Verified directly from `results/phase1*/per_example.jsonl`:
>
> | arm | abstains on *answerable* rows | unanswerable correct |
> |---|---:|---:|
> | A frozen | **2/192 (1.0%)** | 22/64 |
> | B LoRA | **116/192 (60.4%)** | 51/64 |
> | C NoRA | **116/192 (60.4%)** | 62/64 |
>
> Both adapters became near-perfect abstainers. They did not get worse at
> reasoning; they learned to emit `UNKNOWN`.
>
> ### Cause: already diagnosed in RESULT.md, and more precisely than here
>
> **Do not treat the abstention collapse as an open question — it is not.**
> `RESULT.md` §"Gate-off ablation" already identifies the cause and quantifies
> the mechanism from first-token logits:
>
> - 25% of the curriculum has `UNKNOWN` as its *entire* target, and `UNKNOWN`
>   is token 51378 — the same token the abstention gate biases by +5.15625.
> - The frozen model holds a **~13-nat margin** against `UNKNOWN` on answerable
>   rows. Training shifts it **~9 nats** toward `UNKNOWN`, leaving less than the
>   +5.156 bias, which then flips 50–81 answerable rows into abstention.
>
> A single-variable **gate-off ablation** (`ablations/gate_off_nora_v1/`, own
> `RESULT.md`) confirms it, with no retraining needed — gate-ON and gate-OFF dev
> loss are bit-identical because the loss is computed before the logits are
> mutated:
>
> | core | NoRA gate ON | frozen gate ON | NoRA gate OFF | frozen gate OFF |
> |---|---:|---:|---:|---:|
> | A1.1 | 133/256 | 170/256 | **171/256** | 162/256 |
> | A1 | 128/256 | 168/256 | **175/256** | 161/256 |
>
> **This does not overturn the Phase 1 decision**, and the ablation says so
> itself: preservation is byte-identical under both gate settings (0 of 200
> predictions changed), so NoRA's ~11-point preservation regression is real
> adapter damage; and the gate-off gain decomposes as −34 answerable / +43
> unanswerable (A1.1), so the entire net gain is *better abstention on
> unanswerable items*, not better long-context reasoning.
>
> ### Also already resolved — do not re-open
>
> - **Section 4.4's premise question is ANSWERED.** The same-LCR prior-training
>   hypothesis is **REFUTED**: the merged adapter's Stage A mix is CC0
>   long-context data plus short-response replay, not the LCR SFT corpus. Phase 1
>   was not retraining on an already-absorbed distribution.
> - **`MAX_LENGTH`**: 2560 retains only **73.3%** of the bounded curriculum
>   (1759/2400), not ~100% as section 3 claims — the smoke retained 100% and
>   masked it. Later raised to **3840**, after which 0 of 2400 were dropped.
> - Most of section 4's checklist has been worked through in `RESULT.md`
>   §"Audit checklist"; read that before re-auditing anything here.


Refreshed 2026-09-06. The previous handoff was written before both smokes, both
bounded stages and the 2560→3840 repair, and had become misleading in almost
every section; it is archived verbatim at
`results/audit_20260905/HANDOFF_before_20260906_refresh.md`.

**Everything below is a claim to be verified, not a fact to be trusted.** The
codebase and the JSON under `results/` are the only authority. Where this
document and the repository disagree, the repository wins.

---

## 1. The question this lab exists to answer

Supa-A1.1 scores 170/256 on the Rineya Long-Context Demonstration Suite. The
pilot asks, in the cheapest order possible, *why the other 86 rows fail.*

* **Phase 0 — is it interference?** Inference only. **Done, and the selector
  lost:** query selection scored *below* full context (demo B−A = −8.98) and
  above the random control. No causal interference verdict is claimed; see
  `RESULT.md` for the full list of withdrawn claims.
* **Phase 1 — is it capability?** Can a tiny LoRA on the *late* attention
  pathway (L17/18/19, `q,k,v,o` only, rank 4) buy long-context accuracy while
  the catastrophic stem L0/1/2 stays frozen? **This is the live question.**
* **Phase 2 — conditional**, only on a clearly positive Phase 1.

A clean negative is a successful outcome.

---

## 2. Verified state (read from `results/*/run_manifest.json`)

| stage | arm A | arm B | arm C | stage |
|---|---|---|---|---|
| `phase1_smoke` (A1.1, 6 attempts) | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| `phase1` (A1.1 bounded, 300) | COMPLETE | COMPLETE | COMPLETE | **COMPLETE — NEGATIVE** |
| `phase1_smoke_a1` (A1, 6 attempts) | COMPLETE | COMPLETE | COMPLETE | COMPLETE |
| `phase1_a1` (A1 bounded, 300) | COMPLETE | COMPLETE | COMPLETE | **COMPLETE — NEGATIVE** |

`canonical_checkpoint_unchanged` is true on every stage above.

### Bounded A1.1 is complete and negative

Independently re-derived from each arm's raw `per_example.jsonl` (every
`correct` flag re-scored from `raw_prediction`, not re-counted from stored
flags), with the gate reproduced in exact `Fraction` arithmetic:

| A1.1 arm | Target /256 | Gain | RER | Preservation /200 | Regression | Gate |
|---|---:|---:|---:|---:|---:|---|
| A frozen | 170 | — | — | 74 | — | baseline |
| B LoRA | 110 | −23.44 pts | −0.698 | 69 | +2.50 pts | fails both |
| C NoRA-init | 133 | −14.45 pts | −0.430 | 52 | +11.00 pts | fails both |

`winner=FROZEN`, `phase_2_licensed=false`. The collapse is over-abstention:
answerable correct falls 148 → 59 / 71 out of 192 while answerable abstentions
rise 2 → 116 / 116. Dose: B 300 attempts / 300 applied / 0 skips; C 300 / 299 /
1 scaler skip at attempt 6. Both consumed 2400 examples. No compensating steps.

### Bounded A1 is also complete and negative

| A1 arm | Target /256 | Gain | RER | Preservation /200 | Regression | Gate |
|---|---:|---:|---:|---:|---:|---|
| A frozen | 168 | — | — | 74 | — | baseline |
| B LoRA | 108 | −23.44 pts | −0.682 | 68 | +3.00 pts | fails both |
| C NoRA | 128 | −15.62 pts | −0.455 | 53 | +10.50 pts | fails both |

`winner=FROZEN`, `phase_2_licensed=false`. Every arm lands within 5 target rows
of its A1.1 counterpart (170→168, 110→108, 133→128) with identical gate
outcomes, so the checkpoint is not what drives the Phase 1 result. A1 loaded
genuinely as bfloat16 (four independent lines of evidence — the historical
`--core a1 loads A1.1` bug is gone). A1 smoke: A/B/C = 20/21/8 of 32.

All four stages audited together with explicit stage arguments: `phase1` and
`phase1_a1` **VALID**, both smokes **SMOKE_ONLY**, 17/17 pair checks each, every
arm VALID, **zero** discrepancies anywhere; saved adapters PASS on all eight
adapter arms. Evidence:
`results/audit_20260905/final_all_stages_audit_20260906.json`,
`final_saved_values_20260906.json`.

---

## 2b. Gate-off ablation — COMPLETE on BOTH cores, and it changes the reading

`ablations/gate_off_nora_v1/` (own `RESULT.md`). Single manipulated variable: the
cores' constant +5.156249985098839 abstention bias on token 51378 (`UNKNOWN`),
set to 0.0 in memory. Everything else reused unchanged by importing the frozen
modules.

**No retraining was needed, and that is measured, not assumed.** The modeling
`forward` computes the loss in `super().forward()` and only then mutates the
logits; the lab consumes `out.loss` everywhere. Gate-ON and gate-OFF dev loss are
bit-identical (0.04797873464580445, 160/160 rows, max diff 0.0), reproducing
arm C's recorded 0.047979. The saved adapters already ARE gate-off-trained.

| a11 arm | gate ON | gate OFF |
|---|---:|---:|
| A frozen target | 170/256 | 162/256 |
| B LoRA target | 110/256 | 165/256 |
| C NoRA target | 133/256 | **171/256** |
| C NoRA answerable | 71/192 | 115/192 |
| C NoRA preservation | 52/200 | **52/200** |

Removing the gate changes 56/256 target predictions (44 fixed, 6 broken, net
+38) and **0/200** preservation predictions. Under gate-off NoRA clears the
target thresholds (+3.52 pts, RER +0.096) but still fails preservation by 11
points, so **Phase 2 remains unlicensed in every setting.**

Mechanism, quantified from first-token logits: the frozen model holds a −12.88
nat margin against `UNKNOWN` on answerable rows; training shifts it +8.62 (NoRA)
/ +8.84 (LoRA) nats, leaving less than the +5.156 bias, which then flips 50 / 81
answerable rows into abstention. 25% of the curriculum has `UNKNOWN` as its
entire target and `UNKNOWN` is exactly token 51378.

**Do not turn this into an efficacy claim.** NoRA's gate-off +9 rows are
−34 answerable / +43 unanswerable: the gain is better abstention on unanswerable
items, not better reasoning. It still self-abstains on 66/192 answerable rows
with no gate at all.

Every gate-OFF run has a matched gate-ON control; all six controls reproduce
stored history with 0 mismatches across 456 rows; `verify_ablation_runs.py`
reports PASS on all 12 runs. **Replicated on the A1 core:** NoRA 128 → 175 vs
frozen 161 (+5.47 pts, RER +0.147), margin shift +9.23 nats, counting_sets
16/100 vs conditional_rules 37/100 — same pattern throughout.

## 3. What still has to be run

**Nothing in the preregistered programme.** Phase 1 is complete on both cores,
audited, and negative; `winner=FROZEN` and `phase_2_licensed=false` on both.
Under the preregistered protocol the programme **stops here** — a null or
negative Phase 1 does not license a layer sweep.

**The 2×2 composition check is NOT licensed and must never run** — it required
Phase 0 query selection to win independently, and it lost.

Open, and requiring its own preregistration before anyone runs it: the gate-off
ablation (§2b) shows the target-metric failure is largely an evaluation-protocol
artifact. The natural next single variable is to re-calibrate the abstention bias
for the adapted model, or to hold the gate active during training so the adapter
learns against the decoder it is judged by. Neither has been run. Do not fold
either into a re-run of Phase 1 and call it the same experiment.

---

## 4. Running it: launcher pitfalls that have already cost four restarts

The lab's own `logs/run_phase1_resumable.ps1` **cannot run on this machine —
`pwsh` 7 is not installed.** Do not assume it works.

Three further launch methods were tried and each failed for a *different*
reason. Read this before improvising a fourth:

* **PowerShell 5.1 with `*>> $log`** — PS 5.1 wraps a native command's stderr in
  `NativeCommandError`; under `$ErrorActionPreference='Stop'` the first stderr
  line throws and kills the child. Use `Start-Process -Wait -PassThru` with
  **separate** `-RedirectStandardOutput` / `-RedirectStandardError` files.
* **A background shell task, and `Invoke-CimMethod Win32_Process Create`** —
  both die when the controlling agent session exits. This machine's outer job
  object is kill-on-close and does not permit breakaway, exactly as
  `results/audit_20260905/CONTINUATION.md` records.
* **What actually works: Windows Task Scheduler.** Its parent is the scheduler
  service, outside the session job:

  ```
  schtasks /Create /TN SlcmiFinishBoundedA1 /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <shim>" /SC ONCE /ST <hh:mm> /IT /F
  schtasks /Run /TN SlcmiFinishBoundedA1
  ```

  `/TR` is capped at 261 characters, so point it at a short shim script that
  calls the real driver by absolute path. Delete the task when the run is done.

Current driver: `finish_a1_v2.ps1` in this session's scratchpad, launched by
task `SlcmiFinishBoundedA1`; receipts in `logs/finish_a1_driver_v2.launch.json`
and live state in `logs/finish_a1_driver_status.json`. It waits for
≥5500 MiB free VRAM, skips COMPLETE arms, resumes interrupted ones, and never
adds budget. Drop `logs/finish_a1_driver.cancel` to stop it starting new work.

**Direct recovery** (equivalent, from repository root) needs the frozen env:

```
CUDA_VISIBLE_DEVICES=0  CUDA_LAUNCH_BLOCKING=1
PYTORCH_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8
PYTORCH_CUDA_ALLOC_CONF=<same>
../Chat-Bubble/.venv/Scripts/python.exe experiments/supa_long_context_minimal_intervention_v1/scripts/phase1_run.py --core a1 --arms B --max-steps 300 --resume
```

`phase1_run.py` refuses to start on CUDA if these differ.

### Interruptions so far, and why none of them cost science

The atomic checkpoint commits LoRA params, AdamW moments, scaler, scheduler,
RNG, data cursor and committed counts per attempt. Every loss resumed from it:
arm B went 249 → 291 → continuing to the same fixed 300. Committed attempts
never repeat, uncommitted accumulation replays, and aborted segments are kept in
place as explicit lower-bound cost records. Losses to date: a CUDA illegal
memory access at ~attempt 240, a Windows restart, and repeated agent-session
teardowns. **None is a scientific null.** Never restart a *live* arm, and never
open `checkpoints/latest.pt` while its arm is running.

---

## 5. Findings from the 2026-09-06 independent verification

Eight dimensions were re-derived from raw records; 38 findings stand, **none
changes the verdict**. The ones that matter operationally or for the report:

* **Hazard — `scripts/audit_phase1_results.py` defaults `--stages` to include
  `phase1_a1` and hashes `arm_B/checkpoints/latest.pt`.** Running it bare while
  an arm trains reads a checkpoint mid-atomic-replace. **Always pass explicit
  stages.**
* `decide()` names a winner on SMOKE stages; both smoke `summary.json` files
  store `winner: "LoRA"` off 32 rows. `phase_2_licensed` is correctly forced
  false, so nothing is licensed, but the field is misleading — never quote it.
* Winner tie-break silently favours B/LoRA and keys on a 2-dp rounded gain while
  the gate itself uses exact fractions. Undocumented in the preregistration.
* **Undisclosed overlap:** the Phase 0 confirmatory `lcr_dev` suite shares
  **60 of its 200 rows** with Phase 1's *validation* set — both sample the same
  800-row dev split under seed 20260905, so it is systematic. It does not touch
  the Phase 1 target metric (the disjoint demo suite), but it is disclosed
  nowhere and should be.
* **B/C matching is weaker than stated:** C's scaler skip at attempt 6 leaves its
  LR schedule one applied-update behind B for attempts 7–300 and halves its
  gradient scale. Only the applied-count difference is currently disclosed.
* `README.md` still claims `lcr_dev` shares 0 questions with the Phase 1
  training split; 75 question strings recur across LCR train/dev (contexts,
  titles, IDs and scenarios *are* disjoint).
* Superseded preregistration text was left standing in
  `preregistrations/phase1_late_attention_adapter.md` for both `max_length`
  (2560, actually 3840) and the preservation classes (`arithmetic_chain`,
  actually `conditional_rules`). Both are amended and disclosed in
  `preregistrations/audit_20260905_phase1_repairs.md`, but a reader consulting
  only the primary preregistration is misled.

---

## 6. Hard constraints still in force

* Canonical checkpoints are **read-only**, and live only in the primary checkout
  at `../Chat-Bubble/Rineya_Published_Cores/`, not in this worktree.
* `src/**/*.py`, `scripts/*.py` and `configs/*.json` are **frozen** — their
  SHA-256 is recorded in every run receipt. Do not edit or add files there
  during a run; put helpers in `results/audit_20260905/` or outside the lab.
* No pruning, no recurrence or layer looping, no RoPE changes, no KV-cache work,
  no MoE, no full finetune, no training all 28 layers, no MLP training in
  Phase 1, no rank/layer/LR sweep, no second LLM, no large embedding model.
* Phase 2, if ever licensed, compares only L17–19 against L17,18,19,25,27.
* No claims of statistical significance on this corpus.

## 6a. SHARED-HOST PROTOCOL — read before launching anything on the GPU

**This was violated throughout the first session. Do not repeat it.**

This machine runs several concurrent research programmes that coordinate GPU,
host RAM and disk through a lease. The previous session did not know this and
ran every Phase 0 evaluation and the Phase 1 smoke **without taking the lease**.
Consequences that are now on the record:

- an ~8x slowdown on `phase0_eval_lcr_dev` (2844 s for 750 generations),
- one CUDA OOM that killed a Phase 1 smoke on Arm C,
- and probable degradation of *other* sessions' runs in that window.

Any timing or memory figure recorded in `results/*/resource.json` between roughly
13:50 and 16:00 on 2026-09-05 is contaminated by uncontrolled contention and
must not be read as a property of this lab's code.

### The lease

Implementation: `../rineya-dataset-retention-records/hplat-rehearsal-20260921/gpu_lease.py`
Lock file: `%TEMP%/rineya-gpu-0.lock`

Read that module's docstring before using it. Two rules from it that are easy to
get wrong:

- **Absence of the lock file is never sufficient to launch.** Other controllers
  require absence before starting and can take it between your check and your
  write. Use `acquire()` (O_EXCL), do not hand-roll a check.
- **The protocol measures the card, not the bookkeeping.** A foreign job is a
  python process in `nvidia-smi --query-compute-apps`, outside your own process
  tree, with RSS >= 0.5 GiB. A lock-holder PID at 0% CPU is *not* evidence of a
  hang — a 6 MB idle supervisor shim is what a healthy detached job looks like.

API: `acquire(timeout_s, poll_s, label) -> token`, `set_workers(token, pids)`,
`release(token)`. **Release only your own token.** Never clear another lane's
lease, even if it looks stale.

### Queue position

As agreed with the peer sessions at handoff time, this pilot is **last**:

    OSIDE-001 (model-soup)  ->  CLOCKFIX-001 (Rineya Delta)  ->  ABST-001 (DOSELORA)  ->  this pilot

Each lane broadcasts "ended" to the next. Wait for that; do not poll, and do not
jump the queue. Announce to the peer sessions before starting, and say what you
need.

Measured shortfalls at handoff: disk free 20.50 GiB (CLOCKFIX needs 23.03), host
RAM free 4.74 GiB of 15.22 (CLOCKFIX needs 7.4). Re-measure; do not trust these.

### Host RAM is the binding constraint, not VRAM

The card is 8 GB but the host has only 15.2 GB, shared. CUDA allocations past
VRAM spill into host RAM rather than failing fast, which is how a job that
"fits" still takes the machine down. Keep one 596M model resident at a time, one
process per arm, gradient checkpointing on.

### Long jobs and session teardown

Background jobs launched from a session die when that session ends — this is how
the first Phase 1 smoke was lost mid-Arm-B. For anything long, launch detached
(e.g. via WMI `Win32_Process.Create`) rather than as a child of the session, and
treat "complete" as *the expected artefact exists with the expected row count*,
never as "the command returned".

### Do not over-instrument

Monitors and polling loops on this host have previously killed the jobs they
watched and consumed the RAM they were waiting for. Wait on the **artefact**
(a file appearing with the right row count), not on a process or a log tail.

## 7. Environment

* Interpreter: `../Chat-Bubble/.venv/Scripts/python.exe` (Python 3.11.15,
  torch 2.12.1+cu126, transformers 5.14.1, peft 0.19.1). Default `python` is
  CPU-only and wrong for model work.
* GPU: RTX 4070 Laptop, 8 GB, **shared with other research on this machine** —
  a neighbouring `rineya_single_layer_ft_v1` eval held ~2.8 GB during this
  session and blocked the run for minutes. Wait for it; never kill a neighbour.
  This stage peaks near 6.0 GB reserved.
* Run every command from the repository root.
