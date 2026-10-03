# Rineya continual skill research: shared successor protocol

2026-09-19. Working proposal, not retrospective preregistration. The aim is a measured
improvement in learning, retention, composition and cost. A finite benchmark cannot prove
universal preservation or unchanged intelligence at every smaller model size.

## Ownership and immediate experiment

The active `multi-reasoning-slft` task owns new training code and a bounded local GPU pilot
in `ChatBubble-slft/research/composable_skills_v1`. This task owns independent CPU scoring,
method review and cross-scale evaluation in `experiments/continual_skill_audit_v1` and this
research directory. Coordinate GPU ownership before execution. Preserve historical artifacts,
published cores and unrelated working-tree changes. Neither task may treat the other's
recommendation as a measured outcome or as a replacement for the applicable experiment rules.

The first comparison is frozen Supa versus matched FP32 tail20 LoRA CE versus CE plus
old-skill KL anchoring. Use identical new-task rows, token grouping, scope, rank, seed,
optimizer and token-based schedule. Keep CE coefficient at one; log the separate old loss
and its reduction denominator. KL is averaged over the declared prediction positions and
then by the declared family weighting. Budget extra anchor forwards, backward passes,
teacher cache and storage explicitly: parameter efficiency is not total cost efficiency.

The old-skill bank needs answerable/insufficient pairs, entity swaps, arithmetic and clock
examples. The baseline must demonstrate competence in each reported preservation family.
Unknown answers on answerable cases are failures. Distillation of a teacher's outputs can
also preserve mistakes; track ground-truth correctness separately from teacher agreement.
Disjoint worlds and rendering templates support a narrow development test. Repetitive
filler is not representative long-context coverage.

## Tests that distinguish the hypotheses

| Hypothesis | Necessary control | What would count against it |
|---|---|---|
| Old-skill anchoring prevents interference | Identical CE arm; per-family old-correct loss at matched acquisition | Retention improves only because acquisition falls |
| Restricted updates preserve skills efficiently | Full-FT/scaled-delta references; actual effective weight increments and precision | Smaller parameter count but same losses or weaker learning |
| Independent skill deltas compose | Each specialist first succeeds on its own component; add and merge controls | Combined arm loses specialist successes, even if another score rises |
| Routing isolates skills | No task-ID oracle at deployment; frozen-base and random-router controls | Router errors damage old behavior or joint tasks fail |
| Compression preserves the method | Student-only, compressed-base and distilled-student controls at the same inference budget | The apparent gain depends on the larger teacher at inference |

The historical composition bank has no demonstrated discriminative insufficiency competence
for its tested 596M arms. Joint zero therefore cannot identify composition versus capacity,
instruction following, data mismatch or optimization. Failure even with an explicit rule
does **not** rule out instruction-following failure. Add a reachable in-domain specialist
and a supervised positive control before interpreting the joint endpoint.

## Candidate methods, in experimental order

1. **Old-skill anchors during learning.** The current pilot tests whether independent old
   examples fix the coverage gap in previous LWF, which used only new-task answer positions.
   Compare hard-answer rehearsal and teacher-distribution anchoring only after the matched
   CE comparison works. The KL coefficient controls a tradeoff; it is not a retention guarantee.
2. **Constrain the actual update.** If anchoring learns but still damages measured families,
   inspect gradient conflict per family. A-GEM motivates a cheap projection baseline, but
   protecting an average can sacrifice one family. Project the proposed optimizer increment
   against separate old-family constraints and verify losses after the step. Projecting raw
   gradients before Adam is insufficient because its moments and preconditioning change
   the step. A first-order constraint also needs a finite-step check. Keep this as a separate
   method ablation; do not change the pilot mid-run. [A-GEM](https://arxiv.org/abs/1812.00420)
3. **A frozen base with a small skill bank.** Evaluate sparse deltas and low-rank modules with
   explicit activation/routing costs. Preserve module identity, base hash and precision.
   Parameter disjointness or orthogonality does not imply function independence. Sparse
   composition has demonstrated utility in specific language/task transfer settings; that
   supports testing the method, not assuming arbitrary reasoning skills will compose.
   [Composable Sparse Fine-Tuning](https://aclanthology.org/2022.acl-long.125/)
4. **Interference-aware merge and periodic consolidation.** Compare naive addition with a
   sign-conflict-aware merge on development data. TIES addresses small-update and sign
   interference; it remains an empirical baseline, not a no-forgetting guarantee. Train a
   compact student on verified examples from all acquired skills, their combinations and
   old-skill anchors. Consolidation is accepted only after the same retention checks.
   [TIES-Merging](https://arxiv.org/abs/2306.01708)

These are complementary hypotheses, not an instruction to run a large uncontrolled sweep.
Fix a small number of arms and adaptive rounds before each model-execution stage.

## Evaluation contract for sequential and composed skills

After every acquisition stage, measure each skill on the same bank and report loss against
three references: the original base, the preceding stage, and the union of earlier successes.
The third prevents a later model from forgetting a newly learned skill while still retaining
the original base's much smaller correct set. `metrics.sequence` implements all three.

Report individual losses, gains, net change, answer validity, termination, and sensitivity
and specificity for abstention. Preserve scenario/world groupings for uncertainty. Compare
composition against **each standalone specialist**, and test reachable joint tasks separately;
`metrics.composition` implements the component comparisons, not joint-task proof.

Do not select on spent report-only instruments. New evaluation worlds must be disjoint from
new training, old-skill anchors, calibration and hyperparameter selection. Group derivative
views together. Freeze a candidate and its inference artifact before opening a fresh
confirmation bank. Report every preregistered seed, failed run and adaptive look.

Zero observed loss and noninferiority are different claims. As an illustrative planning
target, bounding conditional loss below 2% with one-sided 95% confidence requires at least
149 independent source-correct units with zero losses. For six prespecified families using
Bonferroni alpha=0.05/6, it requires 237 per family. These are sample-size calculations,
not chosen product tolerances, and do not apply directly to correlated variants or reused
selection data. The source-correct count, not the total bank size, is the denominator.

## Generalization to smaller models

Local header inventory now identifies an available smaller candidate: Aegis 1.1 has
454,471,424 stored tensor elements and 19 Qwen3 layers, versus Supa's 596,049,920 and 28.
Both have width 1024, intermediate width 3072 and vocabulary 151936. Weights are present,
but have not been loaded or fully hash-verified for this experiment. These are checkpoint
metadata, not quality or runtime evidence. See `CROSS_SCALE_CANDIDATES.json`. Distinct
training histories prevent this pair from isolating capacity alone. The shared embedding
matrix accounts for 34.23% of the smaller checkpoint versus 26.10% of Supa, so further
depth-only reduction has a fixed storage component that must be measured explicitly.

Keep a common task specification and scoring interface while requalifying each base model.
Do not copy deltas between different tensor shapes, layer meanings or tokenizers. Transfer
the learning procedure and verified examples; train new model-specific deltas or distill
the behavior. Test at least the current 596M base and a smaller independently qualified
base/student before claiming cross-scale applicability. A stronger model can diagnose task
reachability but cannot alone establish a causal capacity effect across different pretraining.

For every scale, compare learned skill and retention at matched training budgets and at
matched achieved acquisition. Also report total stored bytes, active parameters, adapter
and teacher-cache bytes, peak RAM/VRAM, wall time, generated tokens, prefill/decode latency
and energy only if actually measured. Fixed LoRA rank is not a fixed fraction of a smaller
model. Keep both a fixed-byte and a fixed-parameter-fraction comparison when feasible.

Finally evaluate the actual deployment precision. A module that works in FP32 but loses
its useful increments on BF16 export is not a deployable improvement. Either preserve a
small higher-precision residual path and measure its cost, or train/qualify under the target
precision. Compression and continual learning are separate interventions until their
combined model passes the same sequential and composition checks.

## Decisions after the pilot

- If the baseline cannot perform an old family, report it as uninformative retention and
  fix the instrument or train a demonstrably competent specialist before testing composition.
- If neither trained arm acquires the new skills, diagnose optimization strength and answer
  validity before interpreting low forgetting. Do not call this a capacity limit.
- If anchoring preserves more at comparable acquisition, freeze a short, separate
  multi-seed replication before adding a second skill.
- If anchoring only reduces acquisition, the next discriminating test is matched-acquisition
  strength and per-family update constraints, not a broader quality claim.
- Only after sequential acquisition and composition succeed should the same procedure be
  tested on a smaller student. Keep the research goal active until model evidence supports it.
