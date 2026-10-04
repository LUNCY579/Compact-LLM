# RDD-001 protocol — freeze before native outcomes

## Scientific scope

**Primary question:** at comparable new-skill competence, which frozen-teacher
recipe damages fewer previously reliable behaviors? Secondary questions concern
harder generalization, sampling coverage, and effective parameter-update size.

The source paper separates rollout policy, KL direction and learning rate; its
findings motivate controls rather than a presumption that on-policy training,
forward KL, low learning rate, or small deltas must win on Rineya. Its principal
experiments were full-parameter fine-tuning, not this q/v LoRA treatment. This
experiment does not establish universal optimal learning rates or mechanisms.

## Frozen treatment factors

| Code | Prefix source | Token objective |
|---|---|---|
| A | Fixed teacher bank | KL(teacher || student) |
| B | Current student | KL(teacher || student) |
| C | Fixed teacher bank | KL(student || teacher) |
| D | Current student | KL(student || teacher) |
| SFT | Reviewed reference solution | Response-token CE |

Each treatment crosses two registered learning rates and every registered seed.
`screen` includes A/B/SFT; `full` includes all five. The default seeds are
17, 29 and 43. Native selection rejects an incomplete stage, missing seeds,
duplicate arms, drifted checkpoints, different sources/teachers, or unpaired
initial adapters. A one-seed engineering pilot must use its own registered plan;
it is not sufficient for a replicated method ranking.

All arms share the exact loaded source, frozen teacher, tokenizer and template,
LoRA target modules/rank/alpha, AdamW settings, clipping, response cap, sampling
temperature, replay exposure and weight, task schedule, evaluation budget, and
checkpoint schedule. Within each seed, the exact same zero-B adapter initialization
is restored before every treatment. Student stochastic rollouts necessarily
change as its policy changes. No dynamic teacher, privileged solution conditioning,
recurrence, new reward model, extra retention regularizer, or merge coefficient is
introduced into the first matrix.

A teacher can be stronger by task competence without being larger in parameter
count. Its reviewed student-prefix feedback and an independent teacher-dev
competence/advantage screen are required. Neither a model name nor low KL proves
that its supervision is useful. The native tokenizer contract is stricter than
merely matching vocabulary size.

## Objectives and exact sampling semantics

At each shared prompt-plus-response prefix h, p is the student distribution and q
the detached teacher distribution. FKL is sum_v q(v)[log q(v)-log p(v)]; RKL is
sum_v p(v)[log p(v)-log q(v)]. Every term normalizes over the **entire vocabulary**.
The loss temperature is 1; generation temperature is separately fixed by the plan.

Losses average response tokens within each sequence, then average acquisition
sequences within an update. Independent retention CE is a separately normalized
mean multiplied by the registered replay weight. This is not global token-weighted
batch reduction. Response lengths and token exposure must be reported.

Response positions are selected from `prompt + response[:-1]`, beginning at the
last prompt position. Prompts are never CE labels. EOS is supervised when it is a
sampled final token or is appended exactly once to a verified SFT solution.
Interior EOS, empty responses, malformed tokens, and silent truncation are rejected.

Student generation is under no-grad. All student rollouts in an update bind to the
current optimizer version and are consumed before the next update. There is no
score-function term through sampled prefixes: these are **semi-gradient** updates,
not claims of an exact total sequence-level reverse-KL gradient.

Teacher arms share one fixed response per TRAIN task and seed. The frozen teacher
then scores the same prefixes the student scores; no private answers or rejected
student attempts are inserted into the teacher prompt. Capped rollouts remain
included and explicitly counted. Failed model calls fail the arm; successes are
not silently selected. SFT is a different comparator, not mislabeled full KL.

The first-order autograd implementation chunks over token positions and recomputes
probabilities in backward. It supports different hidden widths and different
frozen heads. Teacher hidden states/head can stay on CPU. This saves probability
history storage but does not remove frozen-weight, transformer-activation,
optimizer, transfer, or full-vocabulary projection costs. Higher-order gradients
and trainable output heads are unsupported.

## Data and leakage boundaries

Required roles: TRAIN, TEACHER_DEV, DEV_ACQUISITION, DEV_RETENTION, DEV_HARD;
RETENTION_TRAIN is also required when replay is enabled. The actual lowercase
schema names appear in `data.py`. Only TRAIN and RETENTION_TRAIN admit gradients.
TEACHER_DEV is only a qualification screen, not treatment checkpoint selection.

Globally namespaced parents cannot cross roles. Exact keys and normalized prompts
are audited, including collisions within a role. Every training reference must
pass its declared exact/final-integer grader. Held-out rows carry no solutions.
Existing protected evaluation banks are not opened, imported, reclassified,
selected upon, or trained upon. Supply approved parent/prompt-hash exclusions.
These checks **do not prove absence of semantic near-duplicates**.

Use fresh, owned or appropriately licensed verifiable tasks. Start with bounded
procedural arithmetic or retrieval-and-transformation. Hard tests should alter
operation structure, dependency depth or distractors, not just constants. Include
fresh retention sources representing the actual skills intended to be preserved,
including long-context behavior and abstention where those claims are intended.
Source labels alone cannot certify this coverage; it requires content review.
Use an adequate registered context cap; do not truncate long retention tasks to
fit a short-context pilot and then claim long-context preservation.

The separate confirmation catalog is committed by SHA-256 before any native
training. Its content is opened only after eligible DEV selection is locked and
explicit access is requested. Confirmation parents/keys/prompts are checked against
the entire development/training catalog and supplied historical exclusions. An
interrupted or failed confirmation access stays spent; it is not a retry budget.

## Estimands, selection, and uncertainty

The main design matches task/update exposure, **not compute**. Report the
acquisition-versus-retention curve across checkpoints rather than interpreting
small/no-learning updates as success. Native DEV selection requires positive
registered acquisition gain and an acceptable drop on **every** retention source
in **every** seed. Defaults are +0.03 acquisition and at most 0.02 retention drop;
these are prospective engineering policy values, not statistically powered margins.
Review and register them before outcomes; do not retrospectively weaken them.

Choose one configuration and the same checkpoint step across all seeds. Among
eligible choices, minimize mean damaged retention cases, then maximize worst-seed
acquisition gain, then mean hard-task accuracy, then prefer the earlier checkpoint
and lexicographic configuration name. No best-seed cherry-picking. A candidate
can be declined entirely. Confirmation applies the same scoped gates; it does
not authorize deployment or a general-intelligence claim.

Report total accuracy plus baseline-correct retained, damaged and newly rescued
cases per source. The baseline-correct denominator is fixed by source predictions;
when it is zero, reference retention is undefined, not perfect. Parent-cluster
bootstrap intervals concern paired changes within a seed; they are **not** a
replacement for training-seed variability, adequate sample sizes, or power analysis.
The primary confirmation statement remains confined to the registered suites.

Sampling reports the standard unbiased pass@k estimate from n attempts and a real,
gold-free majority-of-visible-output-strings selector, including its selected
accuracy. Candidate availability is not deployed selection success. There is no
learned verifier, oracle deployed selector, or fixed-FLOP system benchmark here.

## Cost and update geometry

Trace generation attempts/tokens, forward input-token volume for each role,
gradient-bearing input tokens, supervised/replay response tokens, capped rollouts,
wall time, and qualification/evaluation versus training phases. These are measured
counts, **not FLOPs**. Teacher-bank collection cost is recorded separately for every
consumer; also record which consumer created it. Do not charge it only to the first
arm when drawing a method-cost conclusion. Report both unamortized and stated
amortized scenarios. CPU offloading trades latency and host memory for residency.
A future equal-compute campaign requires a new protocol.

Geometry is computed on effective q/v updates, including
`scale * (B_final @ A_final - B_initial @ A_initial)` for a warm-started factor state.
Native v1 installs zero-B adapters. Report L2, mean absolute change and fractions
below 1e-8, 1e-6 and 1e-4. The denominator is adapted matrices, not all frozen model
parameters. Threshold-small does not mean exactly zero, and LoRA-factor movement
is not effective weight movement. Low learning rate is not equivalent to scaling
an already-trained delta.

## Deliberate limits and complementary follow-ups

No native efficacy result, 344M experiment, remote teacher/API, model conversion,
quantized teacher, arbitrary architecture support, automatic training resume,
success-filtered dataset, reinforcement learning, or delta composition is delivered.
GD-001 can consume separately parity-qualified exports; it is not silently merged.
After a clean standalone acquisition/retention result, test composition with an
independently qualified skill and evaluate centralized reasoning at matched total
generation budgets. These are follow-ups, not conclusions of this implementation.
This branch complements existing research and does not obsolete or replace it.
