# 96× the budget does not rescue 13 displaced layers — and CE closes to 0.12 nats while capability stays 13× apart

Every earlier conclusion in this programme about CED trainability rested on ~5,221 supervised
target tokens. The registered ceiling is 1,000,000 per complete matched recipe. This spends
**500,099 per arm — 1,000,198 for the complete matched pair, inside the ceiling under both
readings** — and settles the question.

It also produces the most extreme cross-entropy/capability decoupling measured anywhere in this
work.

Exploratory, CPU FP32, matched within the run. Outside the hashed root — no receipt, no gate,
open splits only, no efficacy claim.

## Design, and the matching

| | C28 | source_control |
|---|---|---|
| displaced layers | 13 | 0 |
| KV ratio | 0.536× | 1.000× |
| supervised target tokens | **500,099** | **500,099** |
| optimizer steps | 468 | 467 |
| TRAIN rows consumed | 3,743 | 3,743 |
| epochs | 0.376 | 0.376 |
| rows revisited | 0 | 0 |
| device / dtype | cpu / float32 | cpu / float32 |
| wall clock | 5.14 h | 6.50 h |
| completed | **true** | **true** |

Identical token count, identical row count, identical recipe (AdamW lr 1e-5, wd 0, clip 1.0,
accum 8 at micro-batch 1, 3% linear warmup then cosine to 10% of peak, all parameters
trainable), identical splits, same seed, same device and dtype. **0.376 of an epoch, so no row
is seen twice** — a null result here cannot be explained as memorisation of a cycled set.

`source_control` was interrupted three times by session restarts and resumed from checkpointed
state. The schedule is a pure function of token count, verified to reproduce the recorded LR at
all nine checkpoints to 1e-12, so the resumes cannot have shifted it; and the TRAIN split's
sha256 is stored in the resume state (unchanged, mtime 02:03:34, earlier than every run).

## Result

| tokens | C28 dev CE | C28 reply CE | C28 task | ctl dev CE | ctl reply CE | ctl task |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.8880 | 12.2181 | 0/60 | 5.5782 | 2.6492 | 20/60 |
| 2,500 | 12.1293 | 11.6981 | 0/60 | 4.9501 | 2.6207 | 19/60 |
| 7,500 | 9.1729 | 8.8553 | 0/60 | 3.7469 | 2.5052 | 18/60 |
| 25,000 | 3.7350 | 4.0615 | 0/60 | 1.8741 | 2.1629 | 23/60 |
| 50,000 | 1.9167 | 3.4322 | **4/60** | 1.0841 | 1.8914 | **26/60** |
| 125,000 | 0.9301 | 2.2481 | 3/60 | 0.6127 | 1.2840 | 16/60 |
| 250,000 | 0.6266 | 1.5154 | 3/60 | 0.4417 | 0.9064 | 23/60 |
| 375,000 | 0.5111 | 1.2509 | 3/60 | 0.3794 | 0.6948 | 26/60 |
| **500,000** | **0.4727** | **1.1039** | **2/60** | **0.3569** | **0.6091** | **26/60** |

**Final: C28 2/60 (95% CI [0.9%, 11.4%]) against the control's 26/60 ([31.6%, 55.9%]).
Intervals disjoint.** C28's lifetime peak is 4/60 at 50k; the control's is 26/60.

**The budget hypothesis is falsified.** At 96× the budget every prior conclusion rested on, C28
does not recover. The 40-step negative result stands and is now dose-resolved across two orders
of magnitude.

## The decoupling, which is the more useful finding

    CE gap at init     7.3098 nats      task 0/60 vs 20/60
    CE gap at 500k     0.1158 nats      task 2/60 vs 26/60
    -> 98.4% of the likelihood gap closed; the capability gap did not move

At 500k the two models are **0.116 nats apart in held-out cross-entropy and 13× apart in
measured capability.** And C28's final dev CE of 0.4727 is **5.11 nats *better* than the frozen
source's 5.5782** — a model that beats its own teacher by five nats on held-out likelihood while
answering 2 of 60 questions the teacher answers 20 of 60.

No likelihood-shaped endpoint could tell these two models apart. Both candidate endpoints in the
registered G3 shape — held-out CE and KL-to-teacher — were already measured to decouple from
capability in this programme; this is the sharpest demonstration, at the largest budget, with
the arms matched to the token.

**The damage concentrates in the answer.** The reply-region CE gap is 0.4948 nats against an
overall gap of 0.1158 — **4.27× concentrated**. At initialization that concentration was
1.31–1.52×, so training does not merely fail to fix the answer region: it *sharpens* the
disparity by driving the shared boilerplate to parity while leaving the answer behind.

## What the models actually produce

    want '64.'          C28 '$50.'                     control '60 units.'
    want '40.'          C28 '$40.'                     control '30.'
    want '45 minutes.'  C28 '1 hour and 15 minutes.'   control '45 minutes.'
    want '90 minutes.'  C28 '1 hour.'                  control '1 hour.'

C28 is fluent, terminating (60/60) and well-formatted. It learned the answer *shape* completely
and the arithmetic not at all.

## A claim of mine this falsifies

Partway through, with the control only at its 125k checkpoint (16/60, down from 26/60 at 50k), I
wrote that "both arms over-train on the SFT corpus and lose the held-out capability, so ~50k is
the right dose, not 500k." **That was premature and is now wrong.** The completed control
trajectory is

    20, 19, 18, 23, 26, 16, 23, 26, 26

It ends at its peak. The 125k dip is inside the noise — 16/60 is [17.0%, 38.1%] against 26/60's
[31.6%, 55.9%], overlapping. There is no evidence of over-training in the control at this dose.

## What this establishes, and what it does not

**Established.** At a token-matched, row-matched, recipe-matched 500,099-token comparison on
family-disjoint data with no row seen twice, 13 displaced layers do not recover. Held-out CE
converges to within 0.12 nats of an ordinary control while task capability stays disjoint at
95%. The likelihood gap is not the capability gap.

**Not established.**
- **One seed.** Seed 17 only. The three-seed frontier at 40 steps bounds run-to-run variance at
  a small budget; nothing bounds it here.
- **One arm pair.** `ced_last4` (3 displaced) matches the control at 40 steps and was *not* run
  at 500k, so this says nothing about whether small displacements benefit from more budget.
- **Probe breadth.** 60 prompts, three numeric classes, 24 generated tokens. Retention of other
  capabilities is unmeasured.
- **Not the ceiling.** 500,099 per arm, not 1,000,000. A 2× larger budget is affordable under
  the registered cap for a single arm but not for a matched pair.
- **No sealed or registered DEV evaluation, no receipt, no gate.**

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY experiments/rab001_exploration/build_splits_large.py 10000
for a in C28 source_control; do
  $PY experiments/rab001_exploration/long_budget.py $a 17 500000 1e-5 8 40 60 cpu float32
done
```

~5–6.5 h per arm on CPU (27.0 and 21.4 supervised targets/s respectively). Resumable: re-invoke
with identical arguments. Artifacts:
`evidence/trainability/LONGBUDGET_{C28,source_control}_seed17_b500000_lr1e-05.json`.
