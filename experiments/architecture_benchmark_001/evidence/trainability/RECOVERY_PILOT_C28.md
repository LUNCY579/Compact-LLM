# Bounded recovery pilot, C28 — the destroyed arm recovers fast, but not toward the teacher and not on the task

Exploratory, at implementation `fc44701e`, pinned CPU runtime, official
Supa 1.7.0-devfreeze weights. Run **outside** the hashed experiment root, so it produces no
receipt and opens no gate. Open splits only — no sealed or DEV access, no efficacy claim.

**This run was interrupted at step 40 of 120** when the session ended. The partial
trajectory is preserved verbatim as
`evidence/trainability/PARTIAL_recovery_pilot_C28_step40.log` rather than discarded, and it
is labelled INTERRUPTED, not complete.

## Setup

| | |
|---|---|
| arm | C28 (`G1_CED_C28`, 14 encoder + 14 shared-memory decoder layers) |
| train split | `data/train-smoke.jsonl`, 25 rows, all 596,311,424 parameters trainable |
| optimiser | AdamW, lr 1e-4, weight decay 0, grad clip 1.0, seed 17, gradient checkpointing |
| held-out split | 8 `data/calibration.jsonl` rows, 1,001 target tokens, **never used for a gradient step**, `ensure_disjoint` enforced against train and pilot |
| generation | greedy, 24 new tokens max, 6 open pilot prompts |
| reference | frozen source held-out CE = **5.6679** on the same 8 rows |

Note the optimiser is AdamW, which the registered 25-step engineering recipe deliberately
is not. That recipe is an engineering smoke and was left untouched; this is a separate
exploratory probe.

## Result

| step | tokens | held-out CE | KL→source | top-1 agree | pilot hits | terminated | distinct | train CE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | **12.2003** | 8.9176 | 0.021 | 0/6 | 0/6 | 6 | — |
| 20 | 2,585 | **2.5368** | 3.5077 | 0.297 | 0/6 | 6/6 | 1 | 2.6281 |
| 40 | 4,968 | **2.2796** | 5.1657 | 0.287 | 0/6 | 6/6 | 4 | 1.5546 |

Generation, same prompts, expected answer `3`:

    step  0   ``fredou间距تك pred usamovereito以下-added itdagioventions fleck itfers…
    step 20   "The current version is 19:15 AM AM, but I will wait for the "   (all 6 identical)
    step 40   "The current task is to finish the coding sprint."

## What this establishes

**The architecture is not broken as an optimisation problem.** From CE 12.20 — worse than
uniform (11.93) — to 2.54 held-out in 20 steps and 2,585 supervised target tokens. This
contradicts the earlier reading of the initialization numbers, and it means the destroyed
initialization is a *starting point*, not a dead end. Fluency returns too: token salad at
step 0, grammatical English by step 40, and termination goes 0/6 → 6/6.

## What this does NOT establish — three separate claims, only one of which is supported

The brief for this work distinguished recovery, improvement over the untrained conversion,
improvement over the official source, and superiority to an equally-trained ordinary
control. They come apart sharply here.

1. **Recovery from the untrained conversion: supported.** 12.20 → 2.28 held out.
2. **"Better than the source": NOT supported, despite 2.28 < 5.6679.** The held-out rows and
   the training rows come from the *same* synthetic corpus
   (`Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0`, 16,504 template
   families, highly redundant). The frozen source was never adapted to that corpus, so the
   comparison rewards format adaptation rather than capability. Any model fine-tuned on 25
   rows of it would be expected to gain. **Without the matched control this number means
   nothing**, and the control is the next run.
3. **"Recovering toward the teacher": contradicted.** KL to source falls 8.92 → 3.51 by step
   20, then **rises to 5.17** by step 40 while CE keeps improving. The model is moving *away*
   from the teacher's distribution as it specialises. So this is distribution shift toward
   the training corpus, not distillation toward the source.
4. **Task capability: not recovered.** Pilot hits are **0/6 at every checkpoint**, against
   the frozen source's 6/8 on the same family. CE improved by ~10 nats and the arithmetic
   task did not move at all. Step 20 is worse than it looks: all six prompts produced the
   *same* output (`distinct = 1`), so the termination rate of 6/6 is a degenerate pass. This
   is exactly why the `G1_QUALITY` gate added in HOTFIX-003 checks distinct-output count
   alongside termination.

## Reading

Held-out CE falling by 10 nats while task accuracy stays at zero and KL to the teacher
*increases* is a textbook case of the failure the brief warned about: training-loss
reduction is not reasoning improvement. The honest summary is that the CED arm is
**trainable and fluency-recoverable**, and that nothing here shows it is **useful**, or that
it beats an ordinary architecture given the same training.

The decisive missing measurement is the matched control — the frozen source trained on the
identical 25 rows with the identical recipe. If the source reaches comparable held-out CE
*and* retains its arithmetic ability, then C28's recovery is unremarkable and CED has paid
46% of its KV budget for a capability loss. That run is in progress; its result belongs in
this file before any conclusion is drawn.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY experiments/rab001_exploration/recovery_pilot.py C28            120 1e-4 20 8
$PY experiments/rab001_exploration/recovery_pilot.py source_control   40 1e-4 20 8
```

Cost on this machine: ~68 s/step including the per-checkpoint held-out and generation
evaluation, CPU only, ~10 GB peak RSS with AdamW state for 596M parameters.
