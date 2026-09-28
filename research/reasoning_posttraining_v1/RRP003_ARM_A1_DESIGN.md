# RRP-003 arm A1: the v2 response contract and its reasoning, together

Status: **DESIGN ONLY. Not registered, not granted, not run.** Drafted while the host was
RAM-blocked, so that GPU time goes to the arm rather than to deciding what the arm is.

Registered predictions are in §6 and are written **before** the arm runs. I expect this arm to
**pass P1 and fail P2**, and §6 says so in advance so that outcome cannot later be presented as
a success.

---

## 1. What the evidence says the arm must target

| observation | source |
| --- | --- |
| v2 accuracy 0.0174 against a 0.4391 floor — worse than a prompt-copying shortcut | CL-040 |
| 91 of 230 predictions fail to parse; 87 for one mechanical reason | CL-041 |
| repairing the format moves validity 139→205 and accuracy **4→4** | CL-033 |
| after repair the errors become `answer_requires_status` (34) and `schema` (31) | CL-033 |
| v1 accuracy 0.2553 against a 0.1411 floor — it **clears** its floor | CL-028 |
| 0 of 131 answerable v2 worlds solved | measured this session |

Read together these say something specific. The model has **some** capability in v1's
plain-string contract over the *same reasoning families*, and **none** in v2's JSON contract. The
repair experiment rules out syntax as the cause: once the JSON parses, the model is emitting the
wrong *keys* and answering worlds that required a status. So the deficit is three stacked things,
and an arm that fixes only one will not move the number:

1. **syntax** — strict JSON, quoted array elements
2. **schema** — the right keys for this family (`path`/`cost`, `actions`/`cost`, `order`, …)
3. **contract** — emit `{"status": …}` when the world is unanswerable

And underneath all three, **the planning itself**, which v1 says is partly present.

## 2. The arm

Supervised fine-tuning on Forge v2 `data/rl/train` (16,000 rows), prompt → gold JSON target,
identical inference contract to the baseline. Deliberately the **simplest thing that addresses
all three layers at once**, because the repair experiment already showed that addressing one in
isolation buys nothing.

- **Trainable surface**: blocks 8–27 (the qualified `trainable_tail = 20`), fp32, CpuAdamW,
  gradient checkpointing, TF32 off — every knob identical to the qualified RRP-001 inventory, so
  the arm changes the *data*, not the recipe. One variable at a time.
- **Window**: the binding question. v2 train is p50 3,325 tokens and only 33% fits 2048.
  Training at 4096 is **not qualified** (CL-027 — only inference is). So either
  (a) qualify a 4096 **training** window first, with a backward pass and optimizer state, or
  (b) train on the 2048-fitting subset and label the arm explicitly short-context.
  **(a) is required before this arm is honest**, because (b) trains only on the short third of
  the distribution and the object of study is long-context evidence use.
- **Exposure**: sized from the qualified per-update measurement (~3.7 s/update), not guessed.

## 3. Why not the alternatives

| alternative | why not first |
| --- | --- |
| Train on v1 instead | v1 already clears its floor, so the headroom is smaller and the strata are confounded with family (CL-019). v2 has the three-way status that spans families. |
| Format-only repair as training signal | Already falsified: the repair moved format 139→205 and accuracy 0→0 (CL-033). |
| RL on the verifier | The verifier passes adversarial audit (4,390 attacks, 0 leaks), which is a prerequisite, but the readiness gate and on-policy pilot are unattempted and every grant pins `on_policy_rl_rollout_tokens: 0`. Not first. |
| A larger teacher / distillation | Barred by every grant's `excludes` and needs authorization. |

## 4. Guards this arm inherits and must not weaken

- **Preservation is a gate, not a score.** P3 on RRP2-PRES-DEV at a −0.03 margin. An arm that
  buys v2 accuracy with ordinary-response or abstention regression has not improved the model.
  The bank is read as pass/fail only, never for ranking.
- **Three reads of RRP2-DEV-V2 for the entire ladder.** This arm spends **one**.
- **RRP2-DEV-V1 once**, after selection, as the cross-grammar check — never for selection.
- The source checkpoint is frozen: parameter digest before and after every evaluation.
- Training needs a **new registered experiment and grant**. Every existing grant lists
  `training` in `excludes` and pins `processed_training_tokens_all_attempts: 0`, so no current
  lane can be stretched to cover it (and CL-045 is what stretching a lane looks like).

## 5. What P2 actually demands — decomposed, because the headline number is misleading

Measured on the 230-row evaluated subset (ANSWER 131 / IMPOSSIBLE 66 / UNKNOWN 33):

| predictor | correct | accuracy |
| --- | --- | --- |
| perfect abstention + **zero** reasoning | 99/230 | 0.4304 |
| `copy_prompt_sentinel` — **the floor** | 101/230 | **0.4391** |
| P2 threshold (floor + 0.05) | 113/230 | 0.4891 |

**So the v2 floor is very nearly "abstain perfectly, solve nothing".** The prompt prints the
status literal, so perfect abstention is free. That makes P2 concrete:

> **P2 = learn the abstention contract AND genuinely solve ≥ 14 of 131 answerable worlds
> (10.7% of them).** The source currently solves **0**.

This reframing matters in both directions. It makes P2 reachable rather than absurd — it is not
"go from 0.02 to 0.49". It also means **passing P2 licenses only "solves about 11% of the
planning worlds"**, and any report must say that rather than implying general competence.

## 6. Registered predictions, written before the arm runs

1. **P1 passes.** Abstention is nearly free once the contract is trained, so accuracy should
   move from 0.0174 towards ~0.43.
2. **P2 fails.** I do not expect ≥14 of 131 answerable worlds to be solved by contract training
   alone, because the contract is not what makes a 12-step plan correct.
3. **Format validity rises sharply** — the 87 mechanically-repairable parse failures are exactly
   what SFT on gold JSON should fix.
4. **`wrong_sentinel_on_unanswerable` becomes the interesting channel.** Once the model emits
   status keys at all, whether it picks UNKNOWN vs IMPOSSIBLE correctly becomes measurable for
   the first time; on the baseline it was unmeasurable because it emitted no status key at all.
5. **Preservation risk is real and concentrated in abstention.** Training a model to emit
   `{"status":"IMPOSSIBLE"}` may increase abstention on the preservation bank's *answerable*
   rows. That is precisely what P3's `false_abstention` channel and the 40 minimal pairs
   measure, and it is the most likely way this arm does harm.

If prediction 1 holds and 2 holds, the honest headline is **"the arm learned the response
contract, which is not the task"** — informative, bounded, and a direct pointer to what the
second arm must attack: the planning itself, not its presentation.
