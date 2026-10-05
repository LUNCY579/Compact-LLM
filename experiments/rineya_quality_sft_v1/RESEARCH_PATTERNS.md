# Conditional quality patterns and opposing explanations

Scope: independent offline reanalysis of retained historical aggregate reports.
No model/data generation, bank admission, native execution or promotion is implied.
The active research objective remains open until useful gains are demonstrated
on the intended0.6B source with preservation and harder transfer. Deterministic
execution is not a universal deterministic improvement law.

## A counterexample changes the dose hypothesis

Same ungated Supa/last20/WSD(.95)/1e-5/seed17/corpus:

| Actual targets | Worlds / updates | Arithmetic correct /150 | Source successes retained /47 | First LR / peak |
|---:|---:|---:|---:|---:|
|1397|10|47|39|.7154|
|5233|41|74|41|.1788|
|15734|118|64|38|.0596|

Increasing exposure from the first to the middle point improves BOTH measured
accuracy and preservation. More exposure beyond the middle point worsens both.
This falsifies a monotonic claim that less training necessarily preserves more.
It does not prove the middle horizon optimal: world coverage, update count,
relative warmup pacing and integrated LR exposure all change together.

The first clipped gradient norm is identical in the three retained scalar traces,
but the short horizon's first LR is four times the middle horizon's. A plausible
opposing explanation is early-step overshoot rather than simply too little
learning. It remains an inference. LR sums/gradient norms are not measured
parameter-update distance, and Adam/clipping prevent that substitution.

The independent `pattern_audit.py` projection validates complete final records,
source/corpus/recipe identities and arithmetic retained+lost/gained accounting.
It omits per-row outcomes, prompts and prediction bodies. Preserve partial and
incomparable attempts as explicit exclusions, never silently pool them.

## Which positive directions survive counterchecks?

- WSD/.95 beats cosine/.999 on arithmetic totals in four matched draws:
  +7,+4,+13,+6. Source-success retention changes +1,-2,+3,0. The acquisition
  direction repeats; balanced-quality dominance and schedule-only attribution
  do not. Warmup,beta2,schedule and exploration harness differ.
- Response-focused MQ SFT beats whole-sequence by +6,+6,+5 /51 across three
  orderings of ONE238-world bank. All input exposure is24910tokens/13updates;
  response targets1944 versus whole targets24672. This is ordering stability,
  not independent-draw replication or balanced-grounding evidence. The global
  target-count ratio12.69 is a supervision-emphasis clue, not an actual gradient
  or parameter-dose measurement.
- Structural compensation increases known-answer counts but can weaken UNKNOWN
  likelihood. ABST depth20's pooled worsening189/256 hides a family reversal:
  prerequisite0/64 worse, other families189/192 worse. Changing vocabulary at
  depth20 changes no sign; depth18 reverses189/256 signs. These compressed-student
  findings do not establish an ordinary0.6B SFT improvement.

Primary local sources: MODEL_QUALITY_STATUS.md;
docs/model-quality/audit-2026-09-15/chatbubble-results-audit.json and mq-audit.json;
research/rineya_small_vocab_depth_v1/ABST001_RESULT.md;
research/reasoning_posttraining_v1/rrp005/A3_RUNBOOK.md. Exact input hashes and
derived values belong in the external aggregate audit receipt, not in new bank
selection. Existing exposed instruments remain development/history.

## Opposite-direction falsifiers, chosen before new outcomes

1. **Supervision content versus effective dose.** Keep the approved first
   whole/response screen unchanged. If response improves acquisition, compare
   a separately registered attenuated response LR3e-6 against response1e-5 on
   the SAME all200 rows/order/groups/absolute warmup clock. Observe actual
   gradient/update norms and the same quality/retention gates. Neither lower
   loss nor smaller updates alone resolves the mechanism. This is not a new
   arm in the current execution permit.
2. **More repeated exposure versus more distinct situations.** If TRAIN fitting
   rises without harder transfer, prospectively hold token exposure and rule
   difficulty fixed while changing independent-world diversity/repetition.
   Targets/parent groups must be independently admitted; no DEV-success-based
   selection. Sibling-answer collision tests need explicit chance controls.
3. **Answer bias versus evidence-sensitive solving.** Keep complete paired
   present/missing/conflicting public evidence; compare learned supervision
   against an independently reviewed teacher-blind decision/bias control.
   Gains require answerable correctness AND preserved safe refusal, per family.
   A tracking/KL improvement alone cannot count as competence.

No opposite branch launches automatically. Each needs its own reviewed design,
actual admitted data, resource/qualification/cost limits and an explicit one-cell
execution decision. Existing first-screen quotas, flags, budgets and thresholds
remain unchanged. The current bank owner still governs construction/admission.

## External research motivates tests, not Rineya claims

The self-distillation study separates rollout,coupling and KL axes; its larger
model findings cannot select a Rineya winner. [Zuin et al.,2026](https://arxiv.org/html/2609.39494v1).
This supports isolating causes instead of inferring safety from teacher alignment.

A study of weak supervision links transferable learning to faithful reasoning
and saturation dynamics; it reports reasoning-trace SFT interventions on3B
models. [Rahman et al.,2026](https://arxiv.org/abs/2604.18574).
This motivates verified short-process versus final-answer supervision if Rineya
fits examples but fails fresh compositions; it does not prove0.6B benefit.

Difficulty-aware small-model post-training research distinguishes acquisition
during SFT from consolidation during RL. [He et al.,2026](https://arxiv.org/abs/2606.04466).
This supports checking learnability before adding RL; no RL/teacher spending or
new training permission is opened here.

Published rationale supervision results also oppose a blanket small-model
capacity ceiling: selected770M T5 experiments benefited from auxiliary rationales.
[Hsieh et al.,2023](https://aclanthology.org/2023.findings-acl.507/).
Generated rationales are not independently verified process labels, so this
does not admit teacher text or demonstrate Rineya transfer.

Conversely, small-model distillation research reports a learnability gap for
long reasoning and benefits from shorter/mixed demonstrations.
[Li et al.,2025](https://aclanthology.org/2025.findings-acl.1301/).
Its long/short teacher conditions also change teacher identity. A Rineya test
must hold verified solution content/difficulty and source teacher provenance
fixed before attributing an effect to length. Shorter is a hypothesis, not law.
