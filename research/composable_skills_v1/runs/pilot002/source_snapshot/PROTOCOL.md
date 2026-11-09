# CS-PILOT-001: protect old inputs while acquiring transition skills

2026-09-19. Exploratory local pilot, specified before any model call by this package.
This is an empirical attempt at better acquisition/retention, not a claim of a general
no-forgetting algorithm. The user's active research goal authorizes this bounded local work.
Historical records, checkpoints, protected evaluation banks and registries are read-only.

## What changed in the research question

Titan-Sparse shows strong acquisition and partial retention repair. It still forgets.
COMPOSE-001 did not make fixed addition of independent skills reliable. COMPOSEREQ's
joint zero is inconclusive because no component learned discriminative insufficiency
handling on that instrument. Failing after an explicit rule does not rule out an
instruction-following failure. Neither near-orthogonal deltas nor disjoint masks imply
functional independence in a nonlinear network.

The prior LWF trainer caches frozen-base outputs on the **new transition training
examples**, and anchors the same new-answer positions. It does not anchor a separate
old-skill input distribution. Its failure does not close independent-anchor distillation.
Its `(1-lambda)*CE + lambda*KL` also changes the new-task CE coefficient. This pilot
keeps CE at coefficient 1 and puts the preservation penalty on independent old inputs.

The recently completed LORA-596 run learned 70/384 transitions and cleared 1/3 floors.
It lost 0/47 permissive arithmetic successes, 3/41 strict successes, and 9 source-correct
long-context rows (3 gains). Its actual BF16 training and linear step-based schedule differ
from its FP32, token-schedule intent. Those are bounded observations, not evidence that
LoRA cannot acquire the skill. The independent supervisor audit is stored at
`C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/research/continual_skill_audit_v1/LORA596_AUDIT.json`.

## Fixed comparison

- Exact Supa 1.7 source SHA `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
- FP32 base, FP32 rank16 LoRA alpha32, zero dropout, attention+MLP matrices of blocks8–27.
- Original RRP-015 384 rows, 54,172 processed / 5,254 supervised tokens, exact 28 groups.
- Torch AdamW, betas(.9,.95), epsilon1e-8, no weight decay, clipping1.
- Peak LR1e-4, token-position WSD20/60/20 with square-root decay to0.1 peak.
  LR is deliberately a LoRA pilot choice; not asserted equivalent to dense1e-5.
- Control: CE(new); treatment: CE(new)+0.25*KL(base||student on old inputs).
- Both execute the same four old forward examples per group. Only treatment backpropagates
  their KL. Report computation separately: old inputs are not free, and this is replay.
- Teacher forcing on source-correct anchor answer plus EOS. Full-vocabulary KL at T=1.
  Cache source hidden states and reconstruct logits through the unchanged output head.
- Neither treatment nor control reads development targets during training.

## Instruments

Fresh anchors and visible development: 28 worlds per split; 112 binding rows (matched
answerable/missing × identity-swapped),28 integer arithmetic,28 clock-time rows. Anchor
and development have disjoint worlds, exact semantic problems and rendering templates.
Seeds301/303 fixed before baseline; seed302 was rejected solely for an exact arithmetic
overlap. Half of binding contexts have repeated irrelevant filler. This is a narrow context
stress test, not representative long-context reasoning. No broad dialogue, coding or
reasoning-retention claim can follow from it.

Qualify source competence: ≥32 correct anchors overall, ≥8 in each family and ≥8 each
on answerable and missing binding rows. Retain all
source-correct anchors in the cache; the fixed shuffled cycle exposes112 rows over28groups.
KL is a mean over old target positions, not family-balanced. Record actual per-family
presentations and target-token counts in both arms so this weighting remains visible.
Report source-correct missing/answerable coverage. Score every development row, including
source-incorrect rows, and separate losses from gains. Report joint correctness across all
four binding variants and conditional forgetting. These synthetic families are development,
not independently sealed confirmation. Related rows are clustered by world for uncertainty.

Acquisition uses already selection-admissible SELECT-VIEWS-001, all384 rows. Training-parent
overlap must be zero. Its published per-family floors remain30/128,10/128,9/128, exceeded
by the exact two-sided95% binomial lower bound. Stop-generation failures count as incorrect.
Historical long-context/abstention banks stay report-only and are not used to choose an arm.

## Readout and subsequent decisions

Report both arms. A lower forgetting rate with weak acquisition is not a solution.
Provisional useful-acquisition criterion: all3 family floors and ≥200/384 canonical
answers. This threshold is exploratory and not a release gate. Compare treatment/control
old-correct loss counts at the achieved acquisition; if acquisition differs materially,
do not attribute retention to a better frontier without an acquisition/norm-matched control.
Zero observed forgetting still has uncertainty; p>0.05 does not establish preservation.
No alpha sweep or post-outcome hyperparameter change in this pilot.

If this direction helps, the next independent run adds a matched norm/accuracy control,
then independent training draws. Only after two independently qualified skills exist
should a sequential A→B or LoTTO-inspired support reservation2×2 test begin. Joint
composition must require both skills, not just a mixture of A-only and B-only questions.
A frozen-base fallback can preserve the original branch by construction; routing mistakes
and joint skill behavior remain measurable system-level failure modes.

## Bounded execution

At most2700seconds of supervised stage wall time,56 total optimizer updates,2GiB new
artifacts,0 automatic retries,0 cloud jobs. One shared RTX lease at a time. Require6GiB
free host RAM and6GiB GPU at launch; stop under1.5GiB host free. Source, code, runtime,
bank and teacher-cache hashes are bound. Every stage writes new output and its failure
receipt; never replace a failed attempt with a success. Code changes require a new plan.
FP32 adapter storage preserves the tiny update separately; merging/export qualification is
separate. No historical lane's native qualification is inherited by this HF eager pilot.

## Literature informing the hypotheses

- [Composable Sparse Fine-Tuning](https://aclanthology.org/2022.acl-long.125/): modular sparse
  deltas can compose in its cross-lingual tasks; Titan's skills still need qualification.
- [Lottery Ticket Adaptation](https://arxiv.org/abs/2406.16797): constrained sparse adaptation
  motivates training compatible updates rather than post-hoc addition alone.
- [MoCL](https://aclanthology.org/2024.naacl-short.39/): continual module addition/composition
  motivates a skill-library alternative if fixed merging remains destructive.
- [A-GEM](https://arxiv.org/abs/1812.00420): old-task functional constraints motivate a later
  optimizer-update projection control, not weight-delta orthogonality as a guarantee.
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673): compare preservation
  at useful acquisition, rather than crediting an adapter that barely learns.
