# The matched comparison: CED loses at both doses, and it loses harder at the good one

Exploratory, pinned CPU runtime (python 3.11.15 / torch 2.10.0+cpu / transformers 4.51.3),
official Supa 1.7.0-devfreeze weights. Run **outside** the hashed experiment root — no
receipt, no gate opened. Open splits only; no sealed or DEV access; no efficacy claim.

This file was revised twice as evidence arrived. Both revisions are recorded in §5 rather
than silently applied, because one of them **withdrew a headline claim of mine** and the
other **fixed a scoring bug of mine that had inverted a conclusion**.

## Setup

Identical on every arm: AdamW, weight decay 0, grad clip 1.0, seed 17, gradient
checkpointing, all parameters trainable, the same 25 `train-smoke.jsonl` rows in the same
order, 40 steps, **4,968 supervised target tokens**. Held-out set is 8 `calibration.jsonl`
rows (1,001 target tokens) **never used for a gradient step**, `ensure_disjoint` enforced
against train and pilot. Generation is greedy, 24 new tokens max, on 6 open pilot prompts.

Scoring reports two readings side by side: `numeric` (leading integer, primary — these
answers are integers by construction) and `strict` (full-string equality).

## The full 2×2

| arm | dose | held-out CE @0 | @20 | **@40** | KL→orig @40 | **task @40** | distinct @40 |
|---|---|---:|---:|---:|---:|---:|---:|
| source | lr 1e-4 | 5.6679 | 1.7295 | **1.6213** | 5.4721 | **0/6** | 5 |
| source | **lr 1e-5** | 5.6679 | 2.1091 | **1.5165** | 3.4908 | **6/6** | 6 |
| C28 | lr 1e-4 | 12.2003 | 2.5368 | **2.2796** | 5.1657 | **0/6** | 4 |
| C28 | **lr 1e-5** | 12.2003 | 4.2490 | **2.7216** | 3.0280 | **0/6** | 3 |

## 1. CED loses at both doses, and the gap widens at the good one

| dose | source CE | C28 CE | CED penalty |
|---|---:|---:|---:|
| lr 1e-4 | 1.6213 | 2.2796 | **+0.66 nats** |
| lr 1e-5 | **1.5165** | 2.7216 | **+1.21 nats** |

At lr 1e-4 the comparison was confounded, because that dose destroys the source's task
ability. At lr 1e-5 — where the source keeps **6/6** arithmetic — CED is **1.21 nats worse**
and still scores **0/6**. So the conclusion does not merely survive the correction; it is
stronger at the dose anyone would actually use.

There is no measured axis on which CED wins. It also costs 46% of the KV budget and fails
strict FP32 parity, so it loses on quality, loses on parity, and buys a saving it cannot
spend.

## 2. CED learns the format and never the arithmetic

C28 at lr 1e-5, step 40, all six prompts:

    expected  3  →  "10."          expected  9  →  "10."
    expected  5  →  "19."          expected 11  →  "10."
    expected  7  →  "10."          expected  8  →  "The current one."

It has acquired the corpus's terse "number + full stop" style and emits `10.` almost
regardless of the question — 3 distinct outputs across 6 prompts. Contrast the source at the
same dose: `3.` `5.` `7.` `9.` `11.` `8.`, all correct, 6 distinct.

This is why the nondegeneracy check matters. Termination alone reads 6/6 for *both* arms at
step 40; only the distinct-output count separates a model answering questions from a model
reciting one answer. `G1_QUALITY` (HOTFIX-003) checks both for exactly this reason.

## 3. The dose result, separately

lr 1e-5 is better than lr 1e-4 on **both** axes for the source: lower held-out CE
(1.5165 vs 1.6213) *and* arithmetic intact (6/6 vs 0/6). At lr 1e-4 the source's answers were
`1.` `3.` `4.` `6.` for `3 5 7 9` — wrong. So the capability loss seen earlier was a
**learning-rate effect, not a defect in the corpus**, and the corpus does not need replacing.

## 4. What is established, and what is not

**Established.** CED loses the matched comparison at both doses, by more at the better dose.
CED is trainable and fluency-recoverable from a worse-than-uniform start (CE 12.20 → 2.72)
but does not acquire the task. lr 1e-4 on 25 rows destroys arithmetic in a working model;
lr 1e-5 preserves it. Held-out CE alone cannot certify a run: at lr 1e-4 it improved 4.05
nats while capability went 6/6 → 0/6. KL-to-teacher alone cannot either: it *rises* with
training even when capability is fully preserved (source lr 1e-5: 0.0000 → 3.4908), so it
tracks corpus drift rather than quality.

**Not established.** Nothing beyond 40 steps — every run stopped there. Nothing about a
recipe with replay, a frozen-teacher regularisation term, or a warmup/decay schedule; those
are untried and are the obvious next levers. Nothing about generalisation: the task probe is
6 prompts of one arithmetic family, a smoke test, and the held-out CE set is 8 rows. Single
seed (17), single deterministic CPU run per cell, **no repeats**, so neither the 1.21-nat gap
nor the 0.66-nat gap is replicated. No sealed or registered DEV evaluation was touched.

**Provenance caveat.** The C28 lr 1e-4 cell comes from a run **interrupted at step 40 of a
planned 120**; the other three cells ran their full 40 steps. All four are therefore compared
at step 40, which is where the interrupted run stopped — matched by construction, not by
truncating one side after the fact. Two further C28 lr 1e-5 attempts were killed by session
teardown before producing data and are not included.

## 5. Revision history of this file

**Revision 1** claimed *"CED loses the matched comparison, and the training data destroys the
capability being measured."* The first half held; the second was **withdrawn**.

**Revision 2** withdrew it for two reasons, both my own errors:

1. *The scorer was wrong.* Task hits were scored by full-string equality, so a model
   answering `"3."` instead of `"3"` scored 0/6 while being **entirely correct**. That is how
   a run with perfect arithmetic was published as a total capability loss. `recovery_pilot.py`
   now reports numeric and strict hits side by side and saves every sample.
2. *It was the dose, not the data.* lr 1e-5 preserves capability and reaches better CE, so
   the corpus is usable.

**Revision 3 (this one)** adds the missing C28 lr 1e-5 cell, which was the whole point of the
correction: revision 1's conclusion had been established only at the destructive dose. It now
holds at both, more strongly at the good one.

Both bugs were caught because a control behaved impossibly — not by code review. That is the
argument for keeping the identity controls and dual scoring in every run.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY experiments/rab001_exploration/recovery_pilot.py source_control 40 1e-4 20 8
$PY experiments/rab001_exploration/recovery_pilot.py source_control 40 1e-5 20 8
$PY experiments/rab001_exploration/recovery_pilot.py C28            40 1e-4 20 8
$PY experiments/rab001_exploration/recovery_pilot.py C28            40 1e-5 20 8
```

~11–29 s/step depending on contention; 451 s for the 40-step C28 lr 1e-5 run including
per-checkpoint held-out and generation evaluation. CPU only, ~10 GB peak RSS with AdamW state
for 596M parameters. Artifacts:
`RECOVERY_PILOT_source_control.json`, `RECOVERY_PILOT_source_control_lr1e-05_s40.json`,
`RECOVERY_PILOT_C28_lr1e-05_s40.json`, `PARTIAL_recovery_pilot_C28_step40.log`,
`dose_test_source_lr1e-5_fixed_scorer.log`.
