# RVS-001 research analysis and hypotheses

Status: proposed research, no native outcome. Source keys refer to [SOURCES.md](SOURCES.md).

## 1. Paper finding and interpretation boundary

Brown et al. study independent repeated attempts and distinguish coverage from the ability to identify correct candidates. Their historical SWE-bench Lite result rises from 15.9% with one attempt to 56% with 250. The reported selection methods do not always track increasing coverage; the tested Pythia family also has zero CodeContests coverage at 10,000 samples. These observations motivate measurement, not a universal scaling guarantee. Benchmark-test access must not be silently treated as deployment access. [P1]

The Rineya proposal is a separate, much smaller experiment. It does not assume the paper's scaling curve, runtime, task distributions, or improvement magnitude transfers to Titan. It does not claim a current frontier result.

## 2. Why this matters for Rineya now

The pinned centralized campaign has a frozen Titan source and deterministic acquisition inputs, but its native E0 is blocked and its current-source quality effects remain unmeasured. Later registered tracks include post-training objectives, bounded sampling exploration, modular donors/composition, and separate inference studies. [R1, R2]

RVS should help decide whether the next useful intervention is better proposals, better selection, or cheaper inference. It is not a shortcut around native qualification or an amendment to the first E1 training contrast.

## 3. Three measurements, not one headline

For problem i and a fixed stochastic policy, let p_i be the probability that one attempt is correct. With independent attempts from that same conditional distribution:

    coverage_i(k) = 1 - (1 - p_i)^k
    C_k = mean_i coverage_i(k)

Do not substitute aggregate benchmark accuracy for each p_i. A model that always solves half the questions and never solves the others differs from a model that solves each question half the time.

Illustrative arithmetic, not Titan predictions:

| Per-attempt probability | Attempts | At least one correct |
| --- | ---: | ---: |
| 0.05 | 16 | about 56.0% |
| 0.001 | 16 | about 1.59% |

Let A_k be the probability a candidate-only selector returns a correct candidate, and R_k its conditional success when the pool contains a correct candidate. Then A_k = C_k * R_k. The gap C_k - A_k is selection loss for that same pool distribution. R_k is undefined when C_k is zero, not automatically zero.

A selector that synthesizes a new answer, calls a solver to create one, or emits a correct abstention absent from the pool is a different system. Either include every allowed output action in the defined candidate pool or report its outcome separately; the candidate-only identity cannot be applied blindly.

Independent RNG streams and unchanged contexts are the proposed sampling mechanism. Repeated identical outputs can occur under IID sampling; duplication alone is not proof of statistical dependence. Reusing previous answers in later prompts, adaptive revision, shared mutable environments, or broken RNG state creates a different design and must be labeled accordingly.

## 4. Generation, selection, and efficiency diagnoses

| Observation within the tested budget | Working interpretation | Proposed next step |
| --- | --- | --- |
| Little coverage increase | More attempts expose little additional success | Inspect task competence, training/data, retrieval, or proposal strategies |
| Coverage increases but selected accuracy does not | Useful candidates exist but are not recovered | Improve and independently calibrate selection |
| Both improve but cost is excessive | Recovery works at an unattractive operating point | Optimize budgets, caching, batching, or distillation |
| Both improve with acceptable cost and preservation | Candidate system is worth further study | Freeze a policy and evaluate on newly protected parents |

None of these outcomes proves an absolute capacity limit, universal efficacy, or broad model superiority. A flat 1-16 curve only characterizes the tested policy, tasks, and budget. Do not extrapolate a fitted five-point curve to 10,000 samples as an established law.

## 5. Chance coverage and evidence controls

A four-option uniform guesser has coverage 1 - (3/4)^16, about 99%, after sixteen guesses. It has not learned to reason. Short symbolic outputs, common targets, empty strings, and UNKNOWN can similarly inflate apparent pool success.

Compare against task-appropriate random and constant-answer controls. Such controls may use only legal answer-space information or TRAIN-derived frequencies, not a DEV gold-answer dictionary. Keep causal evidence-removal and counterfactual-evidence controls. On long context, separately inspect answerable and unanswerable rows, evidence positions, and adversarial distractors. Existing construction-time controls are input checks, not proof that native Titan uses the evidence. [R1]

Correct final answers also do not certify valid intermediate reasoning. A lucky terminal answer cannot automatically qualify an entire rationale or tool trajectory for training.

## 6. Post-training integration

For each independently qualified checkpoint, compare frozen-policy greedy accuracy, sampled pass@1, coverage@k, and selected accuracy@k. A training change may increase first-attempt reliability, expand practically reachable solutions, or do one while harming the other. These are distinct outcomes.

E3's proposed RSFT/GRPO/MaxRL comparisons can reuse the evaluation design later. E4's bounded NGU is a separate sampling intervention. WSD is a training schedule, not repeated inference. Do not simultaneously change objective, schedule, decoding, selector, and checkpoint in the first RVS test. [R2]

A future distillation chain is:

    independent TRAIN attempts -> reference/replay audit -> eligible successes
    -> bounded training export -> separate training experiment
    -> held-out single-attempt quality and preservation evaluation

All attempts count toward collection cost. Cap or weight successes per parent under a frozen rule so easy parents do not silently dominate. Keep TRAIN, replay-training, selection, DEV, CHECK, and FINAL roles distinct. Sampling gains alone do not establish stronger weights; that requires a subsequent controlled training result.

## 7. Composable skills and family portfolios

After donor qualification, compare sixteen total samples from one frozen model with sixteen total samples allocated across qualified variants, such as four attempts from each of four variants. Do not compare sixteen with sixty-four and describe the gain as complementarity.

Measure additional verified problems solved, actual selection recovery, and all checkpoint/adapter-switching cost. Parameter-space orthogonality does not logically guarantee complementary solutions. Output diversity alone is not useful if every new output is wrong.

A portfolio gain without a merge gain would motivate investigating parameter composition. A cheap merged checkpoint matching the portfolio would strengthen the merge's practical case. Failure of both would limit that portfolio on the tested task bank, not disprove every composable-skill method. E5's competent-donor boundary remains in force. [R2]

## 8. Harness trajectories and RHD

Treat an attempt as an entire interaction trajectory, not just its final answer. Reset the environment for independent attempts. Preserve tool observations, errors, termination, seeds, and budgets. Do not allow arbitrary generated code execution in the initial bounded arithmetic/retrieval environment.

RHD provides student-first collection, private PASS/REPLACE review contracts, replay-audited TRAIN compilation, and reviewer-free evaluation interfaces. These are useful integration points, not proof of native resampling or successful distillation. Teacher-corrected and student-only trajectories remain separate arms. [R3]

## 9. Uncertainty, adaptive compute, and abstention

Only after fixed-budget results, investigate a staged 1 -> 2 -> 4 -> 8 -> 16 policy. Calibrate stopping decisions using allowed features on separate calibration parents. Do not stop using hidden correctness. Agreement is not a guarantee: a wrong answer can dominate every sample.

Answerability confidence and candidate-correctness confidence are separate predictions. An abstention detector cannot be relabeled as a verifier without qualification. Keep false abstention, correct abstention, and confident wrong answers as separate endpoints. Adaptive budgets need separately registered inference-cost and evaluation estimands.

## 10. Verification hazards

Evaluation oracle, deployment selector, and training auditor are separate roles. The oracle may use references; the selector must not. A checker that solves the whole problem requires a direct-solver baseline. Returning the solver's answer is tool-assisted solving, not candidate selection.

Verifier false positives can accumulate with sample count. If all candidates are wrong and independent false-accept probability is 1%, the chance of at least one false acceptance among 32 is 1 - 0.99^32, about 27.5%. This illustrative event is not itself the final system error rate. Measure accepted-answer precision and false-accept counts at each budget, including adversarial, malformed, and missing-evidence cases.

Benchmark hidden tests, private gold answers, privileged dataset fields, and hidden source-family labels must never enter the production selection view. Public tests may be incomplete; a test pass is not universal program correctness. A learned judge or larger teacher has its own cost and qualification obligations.

## 11. Architecture and cost comparisons

Repeated sampling is initially weight-frozen and complements, rather than replaces, recurrence, longer reasoning, retrieval, and architecture work. Later compare those alternatives at matched total resource budgets. More visible reasoning tokens must be a separate arm because the initial pilot keeps the non-thinking task contract.

Compare small-model sampling with a separately qualified stronger single-attempt baseline only after tokenizer, context, precision, and cost are bound. Parameter count alone does not determine system cost. Account for prefill, decode, verification, tools, batching, switching, and latency.

## 12. Priority and stopping interpretation

First: qualify the inference path and measure the bounded fixed-model screen. Next: pursue whichever bottleneck is observed. Only then consider portfolios, adaptive allocation, long-context expansion, or distillation. These are conditional experiments, not an instruction to launch every direction.

The lasting deliverable is a decision-quality coverage-versus-selection-versus-cost record. A documented negative screen is useful; an oracle score presented as deployable intelligence is not.
