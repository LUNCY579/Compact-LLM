# Matched pilot: a narrow binding signal, acquisition and retention still incomplete

**Neither fixed arm meets the pilot's acquisition criterion, and both forget old answers.**
Independent CPU rescoring reproduces every completed raw output and checks stage artifact
hashes. This is one exploratory seed on visible synthetic development and reused selection
data. It establishes neither comprehensive model improvement nor learning without forgetting.

The training task owns the model artifacts at
`C:/Users/clclc/BusinessProject/ChatBubble-slft/research/composable_skills_v1/runs/pilot004`.
The complete independent result is [PILOT004_FINAL_AUDIT.json](PILOT004_FINAL_AUDIT.json).

## Fixed comparison and observed results

Both arms train FP32 rank-16 LoRA on Supa blocks 8–27, with the same 384 new-task rows,
28 updates, seed, token-based learning rates and old-example forward exposures. The control
uses new-task CE; the protected arm adds 0.25 times old-task KL, with CE coefficient one.
The teacher cache uses source-correct old training anchors, separate from development worlds.

| Endpoint | Frozen base | CE | CE + old-skill KL |
|---|---:|---:|---:|
| Acquisition, /384 | 14 | **194** | **193** |
| Family floors cleared | 0/3 | **2/3** | **2/3** |
| Old development total, /168 | 148 | 148 | 149 |
| Old source-correct answers lost, /148 | — | **4** | **2** |
| Old newly correct answers | — | 4 | 3 |
| Binding, /112 | 111 | 110 | 112 |
| Binding source-correct losses | — | 2 | 0 |
| Integer arithmetic, /28 | 28 | 28 | 28 |
| Clock arithmetic, /28 | 9 | 10 | 9 |
| Clock source-correct losses, /9 | — | **2** | **2** |
| All four binding variants correct, /28 worlds | 27 | 27 | 28 |
| Correct UNKNOWN on insufficient binding, /56 | 56 | 54 | 56 |
| False UNKNOWN on answerable binding, /56 | 0 | 0 | 0 |

Equal total accuracy is not equal capability. CE loses four old successes and gains four.
The KL arm loses two and gains three. Its clock total equals the base's nine, but it loses
two previously correct clocks and gains two different ones. Whole-answer 24-hour clock and
integer format sensitivity rescues **zero** answers in every arm; this diagnostic does not
explain away the losses or change the primary score.

The two arms' clock losses are **different sets**: only dev world 0027 is shared. CE also
loses world 0021; KL repairs that case but newly loses world 0000. Thus protection is not
monotone even within the source-correct clock set. These identities were independently
checked from the immutable audit, rather than inferred from equal loss counts.

The acquisition bank also contains preexisting successes: **both trained arms lose 5 of the
base's 14 correct transition answers** while adding many others. Calling this a new-skill
bank does not make those original successes irrelevant to the overall retention goal.

## Acquisition criterion fails in both arms

The fixed exploratory criterion was at least 200/384 and all three published family floors.

| Family | Base | CE | KL | Published floor | CE / KL exact 95% lower bound |
|---|---:|---:|---:|---:|---:|
| Ledger, /128 | 14 | 38 | 35 | 0.234375 | 0.2194 / 0.1984 — both fail |
| Prerequisite, /128 | 0 | 82 | 84 | 0.078125 | 0.5511 / 0.5672 — both pass |
| Register, /128 | 0 | 74 | 74 | 0.0703125 | 0.4877 / 0.4877 — both pass |

The procedure clearly changes measured one-step behavior, but the designated complete
acquisition test fails. Compared with CE, KL loses five acquisition answers and gains four:
the net difference of one spans nine discordant rows. Its old-development difference of
one spans seven discordant rows (three losses, four gains). Neither supports an equivalence
or superiority claim; descriptive grouped paired p=1.0 for both net contrasts.

## The promising observation is narrower than the aggregate

KL retains the two insufficient-binding answers lost by CE, at almost identical effective
update magnitude. Those two answers belong to **one world**, not two independent replications.
This is a candidate mechanism signal worth testing on a more informative binding bank, not
a demonstrated general retention improvement. The source already scores 111/112 on this
roster task, which does not cover the harder two-hop alias failures in historical experiments.

Conditional precision is weak. The source-correct units include 28 binding worlds, 28
arithmetic worlds, and nine clock examples. Even zero damaged worlds would give unadjusted
one-sided iid 95% upper loss risks of 10.14%, 10.14%, and 28.31%. Correlated/reused data and
multiple exploratory comparisons further limit population interpretations.

## Which rules were actually tested

The official floors are preserved. Additional CPU analysis reports subsets where an ablated
rule changes the reference; these are arithmetic/rule ablations, not replacement shortcuts.

| Decisive subset | Rows | Base / CE / KL correct |
|---|---:|---:|
| Ledger status matters to the answer | 11 | 1 / 3 / 3 |
| Ledger fixed transfer operation fails | 53 | 4 / 21 / 19 |
| Register fixed add operation fails | 67 | 0 / 32 / 33 |
| Register modulus changes answer, among recorded add ablations | 3 | 0 / 0 / 0 |
| Prerequisite precondition matters | 0 | unmeasured |
| Prerequisite delete/add ordering matters | 0 | unmeasured |

There are 15 non-posted ledger events but four are no-ops even if status is ignored, leaving
11 decisive cases. Prerequisite's score does not demonstrate its named precondition or
ordering rules. The new prospective development bank addresses these missing cases;
it has not yet been model-qualified.

## Independent mechanics and cost audit

- Both arms use exactly 54,172 new processed tokens, 5,254 supervised tokens and 18,873 old
  forward tokens. Every learning rate and old-example ID list matches across the 28 steps.
- Old presentations are 87 binding, 16 arithmetic and 9 time examples, with 210/48/54 target
  positions. This objective is token-weighted, not balanced across old skill families.
- Both train 7,208,960 FP32 parameters. Frozen-base checks pass. Teacher hidden-cache/head
  reconstruction has exactly zero observed maximum logit difference after matching GEMM shape.
- First-step CE, KL=0, gradient norm and learning rate match. Step norms agree within
  3.51e-12. No step-one tensor snapshot exists, so whole-vector identity is not asserted.
- Effective weight delta norms are 1.46723908 for CE and 1.46446008 for KL, a ratio of
  0.998106. Cosine is 0.993582. The best scalar approximation leaves 1.28% of KL delta
  energy unexplained: the treatment changes direction slightly, not just overall magnitude.
- Supervised training takes 116.672 versus 141.344 seconds: 21.15% extra in this local pair.
  This is measured run cost, not a general hardware speed benchmark. All five pilot stages
  total 1,286.719 seconds, below the declared 2,700-second pilot budget.
- Three earlier failed engineering attempts are retained: ownership check, tokenizer return
  type, and mismatched LM-head GEMM shapes. Their recorded stage time totals 62.234 seconds.
  Only the third produced source-anchor observations; no candidate training was performed.

See [training invariants](PILOT004_TRAINING_AUDIT.json) and
[effective adapter geometry](PILOT004_ADAPTER_GEOMETRY.json). No paid/cloud jobs were run.
The training task has released the GPU reservation.

## Next discriminating work

Keep the current result fixed. Before claiming a better frontier, test stronger acquisition
and a separately specified anchor-coverage comparison: simple roster versus richer fresh
two-hop/salience binding and adequately covered clock examples at matched anchor budgets.
Choose anchors on frozen-source training/calibration behavior, never candidate development
failures. Do not infer that a larger KL coefficient alone is the solution.

The prospective rule bank also needs source/specialist qualification and a joint-supervised
positive control before a joint zero can identify a composition failure. Sequential extension
should protect both original-base skills and newly acquired skills, retaining original and
stage-A teachers with explicit dependency hashes. Any smaller-model replication requires
its own qualification; the available 454M Aegis is a candidate, not a demonstrated transfer.

The user's comprehensive research goal remains **active and unachieved**. This pilot reduces
uncertainty and supplies tested instruments for the next experiment; it is not a release.
