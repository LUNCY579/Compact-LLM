# HCI-FUT-001 — Effective context use and human–AI delegation

**Status: DEFERRED_DOCS_ONLY. No interface change, model evaluation, participant recruitment, telemetry collection, or execution approval.**

## What the idea is about

Eliav's paper proposes a feedback loop between increasingly capable AI assistance, greater delegation, and reduced independent practice. Its limitations acknowledge indirect measures and the absence of a longitudinal test of the complete loop. Treat this as a hypothesis, not proof that using AI damages a user's cognition. [Paper, Sections 8, 10–11](https://arxiv.org/html/2603.26707v1).

## Why retain it

The useful Rineya question is not whether a person has a particular token capacity. It is whether an assistant helps people complete tasks accurately while retaining understanding, checking evidence, and choosing how much work to delegate.

Do not equate time before switching screens with a universal human reasoning limit. Do not compare a configured model context window with human comprehension as though they were a validated common-scale measurement. These are methodological boundaries for this proposed study, not personal judgments about the user.

## Possible future integration: two distinct evaluation tracks

### Track M — model-only context evaluation

This is our proposed Rineya evaluation design inspired by the broader concern; it is not a test of human cognitive change and not a new attention architecture.

Build a parent-grouped problem bank that varies evidence position, distractor count, contradictory or missing evidence, and the number of facts that must be combined. Keep all variants from a parent problem in one data role. Explicitly define what counts as a contradiction and a correct abstention; do not rely on an ambiguous grader. Validate tokenizer/template length and prohibit silent truncation.

Report task accuracy by difficulty and length, evidence selection, answer correctness given the selected evidence, calibration/abstention, termination, and generation cost. A model that fits a long prompt but cannot use its evidence has not demonstrated effective context use. Keep adaptive development results separate from fresh confirmation results.

Potential integration points are the FMA and NSA directions named in the centralized [method inventory at 5e5fe36](https://github.com/LUNCY579/ChatBubble/blob/5e5fe36ebc6c2725c61c4c4cf78d3f85344cba80/experiments/post_training_mimo_centralized_v1/METHOD_INVENTORY.csv). That inventory records unresolved native/efficacy boundaries. It provides candidate evaluation consumers, not existing proof of this benchmark or permission to reuse protected evaluation material. Bind and inspect actual evaluator interfaces when the task is activated.

A model-only benchmark could be scoped independently later, without waiting for a human study. This record does not launch it now or move it ahead of active RSD/CLARE work.

### Track H — optional human–AI interface study

A future prototype could compare direct answers, evidence-linked answers, and user-selected hint/explanation modes. Do not assume one mode is universally better. Keep an accessible direct-answer path and clear user control; do not insert forced friction or automatic quizzes into ordinary product use under this note's authority.

Before involving participants, obtain the appropriate institutional research review and informed consent. Register a target participant population, task domain, one primary endpoint, secondary endpoints, task-order controls, and an analysis plan. Size a confirmatory study from pilot uncertainty rather than an invented universal participant count. Randomize treatment or counterbalance matched task sets; account for carryover, repeated measures, prior knowledge, and participant-level dependence.

Suggested measures are immediate task success, time and effort, ability to identify a planted assistant error, explanation accuracy, and later unaided transfer to a different problem. Do not interpret one session's recall as evidence of long-term cognitive change. Record adverse trade-offs, including increased burden, inaccessible interaction, and worse task completion.

Collect only consented task events needed for the registered question. Do not capture unrelated browsing, screen recordings, audio, personal messages, or background application activity. Define retention/deletion rules before collection. Keep participant data and private interaction logs out of Git and model training unless separately and explicitly authorized.

No UI source path is asserted here: the current product entry points and event schema have not been inspected for this task. Verify them before proposing a patch rather than fabricating an integration API.

## Revisit only when

For Track M: there is a concrete long-context evaluation need, an owner, an agreed data budget, and a qualified native model/scorer path. For Track H: there is a real product question, an inspectable prototype, appropriate review and consent, and an approved study budget. These triggers are separate.

Success must be evaluated against the registered task goal and its costs. If a design improves immediate answers but worsens unaided transfer or verification, report the trade-off. Do not present the paper's broad hypothesis as confirmed by a small pilot or infer individual cognitive decline from interaction metadata.

## Evidence and boundary

Both tracks are proposed adaptations. No model, user interface, study instrument, participant dataset, benchmark result, or validated psychometric measure was created by this documentation change. See [the index](README.md) before future activation.
