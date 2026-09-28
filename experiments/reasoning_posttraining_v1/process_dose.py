"""RRP-008: the dose ladder for one-step transitions, registered separately from arm A1.

RRP-007 established two things across three seeds. The new skill replicates almost exactly —
289/384 every time, every family clearing its floor. Retention does not: the D3 gate (panel
>= 62 AND references retained >= 37 of 47) passed on seeds 17 and 29 and **failed on seed 23**
at panel 58 / retained 35, and the pre-registered rule is that a majority does not rescue a
failing seed.

So the bottleneck is forgetting, not acquisition, and the obvious question is whether it is a
function of dose. This lane answers it by holding everything except the number of parents
fixed:

    D1    128 parents/family     384 rows    ~54k processed tokens
    D2    256 parents/family     768 rows   ~108k processed tokens
    D3    512 parents/family   1,536 rows    216,896 -- ALREADY MEASURED as RRP-007 seed 17
    D4  1,024 parents/family   3,072 rows   ~434k processed tokens

**The ladder stops at 1,024 for a data reason, not a budget one.** After excluding the 384
scored parents, `prerequisite_plan` has 1,027 available against 1,872 for each of the other
two families, so a balanced draw above 1,024 per family does not exist. Going higher would
mean either unbalancing the families -- confounding dose with mixture -- or taking more steps
per parent, which is a different intervention and needs its own registration.

Why a separate experiment id rather than another RRP-007 grant: RRP-007 is "arm A1", a single
dose. Filing a four-point ladder under it would misattribute the ladder to the arm and make
the arm's own record ambiguous, which is the kind of lane-stretching CL-045 is about.

The ceiling below is RRP-007's shape with the processed-token and wall-second headroom D4
actually needs. Pre-registration, endpoints, predictions and analysis rules are in
`forge-v2-prereg-001/RRP008_DOSE_PREREG.md`, written before any rung was built.

This module registers the lane. The corpus is built by `process_arm.build` with
`--experiment-id RRP-008-PROCESS-DOSE`; there is deliberately no second builder, because two
builders is how the trained task and the scored task drift apart.
"""
from __future__ import annotations

from . import budget

EXPERIMENT = "RRP-008-PROCESS-DOSE"

# Sized from measured quantities for the largest rung, D4:
#   processed tokens -- 3,072 rows x ~141 tokens is ~434k; 600k leaves one retry.
#   wall seconds -- ~212 update groups at the measured 3.7 s/update is ~785 s, plus load and
#     three 24-row probes. budget.start() charges timeout + 20 against this, so a 2,000 s
#     timeout charges 2,020 and must fit.
#   disk -- engine.py checkpoints at group 2 and at the final group only, reserving
#     sum(numel) * 12 + 1 MiB each; ~3.52 GiB per checkpoint for the 314,618,880-parameter
#     tail, so two fit inside 8 GiB.
#   adaptive_model_selection_rounds -- charged once per journal on the first train launch.
#     It is 1 and not 0; a 0 here refuses the lane's own launch.
LIMITS = {
    "total_owned_work_wall_seconds": 3600,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 2400,
    "processed_training_tokens_all_attempts": 600000,
    "evaluation_and_qualification_model_calls": 240,
    "generated_tokens_all_candidates": 30720,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 8 * 2**30,
    "adaptive_model_selection_rounds": 1,
    "model_worker_launches": 4,
    "retries": 1,
    "cloud_or_teacher_spend": 0,
}
EXCLUDES = ("RL", "larger_teacher", "cloud", "release", "architectural_change")

budget.register(EXPERIMENT, ceiling=LIMITS, phases=("train", "resume"), excludes=EXCLUDES)
