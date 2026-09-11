# CORRECTION (supersedes the "data destroys capability" reading below)

**Two things below are wrong, and the error inverted a headline conclusion.** Both are fixed
here rather than quietly edited out, and the superseded text is left in place under the rule.

### 1. My scorer was wrong, and it hid a preserved capability

Pilot hits were scored by full-string equality against the expected answer. After adapting to
this corpus the model answers in the corpus's own style — `"3."` rather than `"3"` — so a run
whose arithmetic was **entirely correct** was recorded as **0/6**.

Re-scored by leading integer, at lr 1e-5 (the registered engineering learning rate):

| expected | got | strict match | numeric match |
|---|---|---|---|
| 3 | `3.` | no | **yes** |
| 5 | `5.` | no | **yes** |
| 7 | `7.` | no | **yes** |
| 9 | `9.` | no | **yes** |

4/4 on the saved samples. `recovery_pilot.py` now records `exact_hits` (leading integer,
primary for this integer-answer pilot) and `strict_hits` (full string) side by side, and
saves every sample rather than the first four.

### 2. It was the DOSE, not the data

The claim "this training data destroys the capability being measured" is **withdrawn**.
Comparing the two doses, same 25 rows, same 40 steps, same 4,968 supervised tokens:

| | lr 1e-4 | lr 1e-5 |
|---|---|---|
| held-out CE | 1.6213 | **1.5165** |
| arithmetic answers | `1.` `3.` `4.` `6.` for 3/5/7/9 — **wrong** | `3.` `5.` `7.` `9.` — **correct** |
| capability | destroyed | **preserved** |

The gentler dose is better on **both** axes: lower held-out CE *and* intact arithmetic. So
lr 1e-4 on 25 rows is simply too high for a 596M model; the corpus is usable.

### 3. What this does to the CE/capability anti-correlation claim

The anti-correlation is real **at lr 1e-4** and is **not** a property of the corpus. At
lr 1e-5, CE and capability improve together. The narrower surviving claim:

> An excessive learning rate can drive held-out CE down while destroying task accuracy, so
> CE alone cannot certify a training run. A capability-retention check is still required —
> but the corpus does not need replacing, and recommendation 0 (fix the endpoint) stands on
> the weaker and still-sufficient ground that CE *can* decouple from capability, not that it
> always does.

### 4. What is still open

The 4/4 above is from the four saved samples of one run. A full 6-prompt re-score at lr 1e-5
for **both** the source control and C28 is running with the fixed scorer; the matched
CED-versus-ordinary comparison at the *good* dose does not exist yet, so **§1's conclusion
that "CED loses the matched comparison" is currently established only at lr 1e-4** — the
dose now known to be destructive. That comparison must be redone before the conclusion is
relied on.

---

# The matched comparison: CED loses, and the training data destroys the capability being measured

Exploratory, pinned CPU runtime, official Supa 1.7.0-devfreeze weights. Run **outside** the
hashed experiment root — no receipt, no gate opened. Open splits only; no sealed or DEV
access; no efficacy claim.

Identical recipe on both arms: AdamW lr 1e-4, weight decay 0, grad clip 1.0, seed 17,
gradient checkpointing, all parameters trainable, the same 25 `train-smoke.jsonl` rows in the
same order, 40 steps, **4,968 supervised target tokens each**. Held-out set is 8
`calibration.jsonl` rows (1,001 target tokens) never used for a gradient step,
`ensure_disjoint` enforced against train and pilot. Generation is greedy on 6 open pilot
prompts.

## The three-way result

| | source untrained | **source trained** | C28 untrained | **C28 trained** |
|---|---:|---:|---:|---:|
| held-out CE (nats) | 5.6679 | **1.6213** | 12.2003 | **2.2796** |
| held-out KL to original source | 0.0000 | 5.4721 | 8.9176 | 5.1657 |
| held-out top-1 agreement | 1.000 | 0.332 | 0.021 | 0.287 |
| **pilot task hits** | **6/6** | **0/6** | 0/6 | 0/6 |
| terminated | 6/6 | 6/6 | 0/6 | 6/6 |
| distinct outputs | 6 | 5 | 6 | 4 |

Full trajectories: `RECOVERY_PILOT_source_control.json`,
`PARTIAL_recovery_pilot_C28_step40.log`.

## 1. CED loses to the matched ordinary control

At identical training, identical data and identical token budget, the ordinary architecture
reaches held-out CE **1.6213** against CED's **2.2796**. CED is **0.66 nats worse**.

This closes the question the earlier pilot could not. C28's held-out CE of 2.28 looked like
it beat the frozen source's 5.67, and that reading is now fully explained as corpus
adaptation: the ordinary architecture adapts to the same corpus *better*. There is no
measured axis on which CED wins. It also costs 46% of the KV budget and fails strict FP32
parity, so it loses on quality, on parity, and gains only a saving it cannot spend.

## 2. The far more serious finding: this training destroys the capability being measured

**The frozen source went from 6/6 to 0/6 on the pilot arithmetic after 40 steps.**

    expected  3  ->  got "1."
    expected  5  ->  got "3."
    expected  7  ->  got "4."
    expected  9  ->  got "6."

It learned the corpus's short-numeric answer *format* — terse, digit, full stop — and lost
the *arithmetic*. Held-out CE improved 3.5x (5.6679 -> 1.6213) over exactly the same 40 steps
in which task accuracy fell to zero. Catastrophic forgetting at 25 rows, lr 1e-4, under 5k
supervised tokens.

### Held-out CE and task capability are anti-correlated here

This is the methodological headline, and it is worse than a null result.

| arm | ΔCE (held-out) | Δtask |
|---|---|---|
| source | −4.05 nats (much better) | 6/6 → 0/6 (destroyed) |
| C28 | −9.92 nats (much better) | 0/6 → 0/6 (unchanged) |

Every arm improved sharply on the metric and either lost or failed to gain the capability.
So on this corpus **CE is not a valid proxy for capability, and optimising it actively
harms the capability**. Any recovery plan whose endpoint is held-out CE or teacher KL —
which is what the registered G3 shape implies, with its `recovery_tokens_per_arm` and
CE/KL objective — would have reported success while destroying the model.

Note also that both arms' KL to the *original* source **rises** with training (source
0.00 → 5.47, C28 8.92 → 5.17). Neither is converging on the teacher; both are converging on
the training corpus. At step 40 the fine-tuned source is *further* from its own original
distribution (5.47) than the CED arm is (5.17).

## 3. What is and is not established

**Established.** CED is worse than the matched ordinary control at equal training. The CED
arm is trainable and fluency-recoverable from a worse-than-uniform start. Fine-tuning on 25
rows of this corpus at lr 1e-4 destroys arithmetic in a working model. Held-out CE is
anti-correlated with task accuracy under this setup.

**Not established.** Nothing about a *well-chosen* recovery recipe: lr 1e-4 on 25 rows for
40 steps is a deliberately small probe, and a lower learning rate, a replay mixture, or a
frozen-teacher distillation term might preserve arithmetic. That is a hypothesis, not a
result, and it is the obvious next experiment. Nothing about longer training — both runs
stopped at 40 steps. Nothing about generalisation beyond one narrow arithmetic family of 6
prompts, which is a smoke test. Single seed (17), single deterministic CPU run, no repeats,
so the 0.66-nat CE gap is unreplicated.

**The C28 run was interrupted at step 40 of a planned 120** when a session ended; the source
control ran its full 40. The comparison is therefore matched at 40 steps by construction, not
by truncation of one side.

## 4. Consequence for the experiment

The registered four-arm design cannot answer its question with this data and this endpoint:

1. The ordinary control D40 is degenerate before training even starts.
2. The CED arms start worse than uniform.
3. Training on the available corpus destroys the capability the pilot set measures.
4. The proposed CE/KL endpoint moves in the *opposite* direction to capability.

Fixing (4) is a precondition for any further training work: the endpoint must be a task
measure with a retention check, not a likelihood. `G1_QUALITY` (HOTFIX-003) is the
initialization-side half of that; the training side needs an equivalent — a registered
regression check on capability the optimiser is not allowed to trade away.

## Reproduction

```bash
PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
cd C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification
$PY experiments/rab001_exploration/recovery_pilot.py source_control 40 1e-4 20 8
$PY experiments/rab001_exploration/recovery_pilot.py C28            40 1e-4 20 8
```

~22 s/step for the source control (890 s for 40 steps including per-checkpoint held-out and
generation evaluation), CPU only.
