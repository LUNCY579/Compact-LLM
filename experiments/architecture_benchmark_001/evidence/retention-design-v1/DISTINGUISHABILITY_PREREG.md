# Making the arms distinguishable: what has to change before any "gain" can be measured

Pre-registered 2026-09-13, at implementation digest `a329eb7b`, **before H56 or P56 has been
trained or scored**. Every threshold below is fixed here and is not to be moved afterwards.

## The problem, stated exactly

From `DUAL28_GATED.md`:

> H56, P56 and D56 are indistinguishable at init by necessity — P56 is the control that isolates
> context from parameter count and only becomes informative once trained; D56 has no memory at all
> and still scores 47/47. And the cost is unpaid: 596M -> 1.156B parameters, 28 -> 57 KV banks.

That is correct, and it has **three independent causes**. Only one of them is about training.

**(1) At initialisation the arms are the same function.** The read output projection is zero, so
the memory branch adds exactly 0 to the residual stream. H56 *is* B28 plus 560M of dead weight.
This is not a weak measurement — it is an algebraic identity. No test whatsoever can separate
H56 from P56 from D56 at init. Expecting otherwise is a category error.

**(2) The retention gate is one-sided and has a ceiling.** `G1_RETENTION` scores an arm against a
frozen 47-item reference of rows the *source* answers correctly. The best attainable score is
47/47 = the source. **A capability gain cannot register on it at all.** The gate answers "did
surgery break anything", which is the question it was built for, and it is the wrong instrument
for "did surgery make the model smarter". Reporting 47/47 as a headline conflates "did not break"
with "is good".

**(3) No arm has been trained with a dose that moves the model.** D28-001's P2 is 25 steps over
8 fixture paragraphs — about 1,600 supervised tokens. This programme has already measured that the
3,159-token engineering lane changed **0 of 150** answers on B28. P2 is a null dose. Whatever it
reports, it cannot separate the arms.

## The headroom nobody has been measuring

The panel is 150 rows. The source answers 47. **103 rows are unexploited headroom**, and they are
not evenly spread:

| reply class | rows | source correct | **headroom** |
|---|---:|---:|---:|
| percentage_reasoning | 50 | 21 | 29 |
| ratio_proportion | 50 | 16 | 34 |
| time_duration | 50 | **10** | **40** |
| **total** | **150** | **47** | **103** |

`time_duration` is where the source is weakest — 10 of 50 — and it is exactly the class that needs
carrying information across positions (start time, elapsed, offset, wrap past midnight). If shared
causal memory does anything for intelligence, this is the class where it should show first.
That is a directional prediction recorded before the run, and it is falsifiable: gains concentrated
in `percentage_reasoning` instead would indicate a generic finetuning effect, not a memory effect.

`run._retention_classes` has computed a per-class `gained` count all along — *"Reported, never
subtracted: a new right answer does not restore a destroyed one."* The number exists. No threshold
has ever been attached to it. That is what this document fixes.

### The headroom is not homogeneous, and that is a confound

`HEADROOM_AUDIT.json` asks of each of the 103 rows whether the source's own text already asserts
the right answer, read in the units the target actually uses (clock targets carry
`minute_of_day`, durations carry seconds):

| stratum | rows | percentage | ratio | time |
|---|---:|---:|---:|---:|
| **wrong_answer** — the model computed something else | **77** | 23 | 34 | 20 |
| **format_only** — right answer present, grammar rejected it | **24** | 5 | 0 | 19 |
| runaway — never terminated | 2 | 1 | 0 | 1 |

Two consequences, both load-bearing:

**The panel understates the source.** 24 rows are answers like `It ends at 6:55 PM.` that are
arithmetically correct and rejected on presentation. **A grammar fix alone reaches 71/150 without
changing a single weight** — nearly the entire finetuning gain, for free, at zero inference cost.
The "47/150" baseline is a grammar-limited floor, not a capability measurement.

**A finetune can score by learning format instead of arithmetic.** Gains in the `format_only`
stratum cost the model nothing in reasoning. So gains must be reported **split by stratum**, and
only the `wrong_answer` stratum counts as capability. Note `ratio_proportion` has **zero**
format_only rows — every gain there is necessarily genuine, which makes it the cleanest class in
the panel and a useful check on the other two.

### The bar an architecture has to clear

Re-reading `RETENTION_DOSE_gpu_lr1e-05_b5200_seed17.json`, which this programme has been citing
only for its loss side: the same run took the panel from **47 to 73 correct** — gained 32, lost 6,
net **+26** — at 5,233 supervised tokens, 41 optimizer steps, 811 s, from ordinary full-parameter
finetuning of B28. The gate verdict on it is **FAIL**, purely because of the 6.

So `H56` is not competing against 47/150. It is competing against **73/150 at 1.00x parameters and
1.00x KV cache**, and it has to get there while carrying 1.94x the parameters and 2.04x the cache.

## Four changes that make the arms distinguishable

**1. Score both directions.** Retention on the 47 (loss) *and* acquisition on the 103 (gain),
reported separately and never netted against each other. A gate that netted them would let an arm
destroy a reference answer and buy the verdict back with an easier row.

**2. Compare paired per-row, not by aggregate rate.** H and P train on identical data in identical
order from one seed, so every row is a matched pair. The test is **McNemar exact, two-sided,
alpha 0.05**, on the discordant rows (`b` = H right & P wrong, `c` = P right & H wrong). Aggregate
"H got 52, P got 50" is not evidence; two arms can reach the same count on different rows.

**3. Add the control that is actually missing: `B28ft`.** H vs P isolates *context* from
*parameter count*, which is necessary but not sufficient. It does not ask the question a model
owner asks: **is the new pathway better than just training the weights we already have?** So
B28ft trains the last 8 transformer blocks of the unmodified B28 — 125,847,552 parameters versus
H's 119,571,072, i.e. the control gets **5.2% more trainable budget than H** — on the identical
recipe, data and order. Deliberately conservative: if H beats a control with more trainable
parameters and no added inference cost, that is a strong result. If H does not beat it, the
architecture earns nothing, however clean the H-vs-P contrast looks.

**4. Price the gain.** Any gain is reported against what it costs:

| arm | parameters | x B28 | KV banks | KV MiB @ 8192 tok | x B28 |
|---|---:|---:|---:|---:|---:|
| B28 (source) | 596,049,920 | 1.00 | 28 | 896.0 | 1.00 |
| B28ft (control) | 596,049,920 | 1.00 | 28 | 896.0 | 1.00 |
| H56 / P56 | 1,156,087,424 | 1.94 | 57 | 1824.0 | **2.04** |
| D56 (depth control) | 1,036,516,352 | 1.74 | 56 | 1792.0 | 2.00 |

H56 costs **1.94x the parameters and 2.04x the KV cache**. A gain of a few rows bought at double
the cache is not a product improvement; it is a research signal. Stating the multiplier next to
every gain keeps that honest.

## Pre-registered decision rules

Retention thresholds are unchanged and inherited from `protocol.json` `G1_RETENTION`: zero
correct-to-incorrect per class, termination rate >= 0.8 per class. New acquisition thresholds:

| # | statement | threshold, fixed now |
|---|---|---|
| A1 | the dose is non-null | >= 10 of 150 answers changed vs source, in **both** arms |
| A2 | context effect exists (H vs P) | McNemar exact two-sided p < 0.05 with b > c |
| A3 | architecture beats matched budget (H vs B28ft) | McNemar exact two-sided p < 0.05 with b > c |
| A4 | the gain is not bought with damage | H's lost count <= B28's at the same dose (6) |
| A5 | directional prediction | majority of H's `wrong_answer`-stratum gain falls in `time_duration` |
| A6 | the gain is capability, not formatting | gains reported split `wrong_answer` / `format_only`; only the former is credited |
| A7 | the architecture is worth its cost | H's corrected-correct total exceeds B28's 73/150 |

A4 is stated against B28's measured 6 rather than the gate's zero-loss rule on purpose. At this
dose *every* arm loses reference answers, so a zero-loss threshold would fail all of them and
separate nothing; the question that discriminates is whether the added pathway is **more or less**
destructive than plain finetuning at equal dose.

**Minimum detectable effect, computed before the run.** With an exact two-sided sign test the
smallest discordance that can reach p < 0.05 is **6 rows split 6-0**:

| b-c | discordant n | exact p | significant |
|---|---:|---:|---|
| 5-0 | 5 | 0.06250 | no |
| **6-0** | **6** | **0.03125** | **yes** |
| 7-1 | 8 | 0.07031 | no |
| 8-1 | 9 | 0.03906 | yes |
| 9-2 | 11 | 0.06543 | no |
| 10-2 | 12 | 0.03857 | yes |

So this panel **cannot** certify an effect smaller than about 6 net rows out of 150. If H beats P
on 4 rows and loses on 0, that is p = 0.125 and the honest report is "no effect demonstrated",
not "a small effect". Recording that limit now prevents reading a 4-0 as a win later.

## What is NOT claimed by any of this

- This is exploratory instrumentation, outside the hashed root. It opens no gate, writes no
  receipt, and licenses no training on a released model.
- A1 failing makes A2-A5 **unreadable, not negative.** A null dose is not evidence of safety and
  not evidence of absence of effect.
- Passing A2 would show shared *context* contributes, on 150 rows of three arithmetic-word-problem
  classes, at one dose, on one source. It would not show the architecture generalises, and the
  103-row headroom is a narrow and easy-to-overfit target.
- 150 rows is a small panel. A2 and A3 are single tests at alpha 0.05 with no multiplicity
  correction, because they are pre-registered here as the only two comparisons that will be run.
  If further comparisons are added later, the correction must be applied and stated.

## Environment note: `Supa/` in this worktree

D28-001's `guards.binding()` requires `source.resolve() == REPO / "Supa"`. No checkout in this
repository has a `Supa/` directory — the official core lives at
`Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze`. The first P1 attempt is preserved as
`outputs/p1-official-v1/blocked.json` with
`official source must be this checkout's Supa/ installation`.

This is the same class of finding as `MERGED_BRANCH_PORTABILITY.md`: a hard-coded path convention
from the authoring environment. It was resolved by creating `Supa/` as **NTFS hardlinks** to the
seven assets named in `source_lock.json`, not by editing their guard. The files are byte-identical,
`audit_source` returns the same digest `2a9c9664...`, `is_symlink()` is false for the directory and
every asset, and the hardlinks consume no additional disk. The guard's intent — that the receipt
binds to specific, hashed official weights — is preserved exactly; only the path layout changed.

`Supa/` is untracked and must never be committed (1.2 GB of weights). It is not in `.gitignore`,
so do not run `git add -A` in this worktree.
