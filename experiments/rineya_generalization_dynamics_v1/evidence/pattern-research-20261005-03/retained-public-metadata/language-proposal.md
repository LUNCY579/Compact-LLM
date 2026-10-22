# Fresh language preservation panel proposal

**PROPOSED_UNQUALIFIED — design only, 2026-09-28.** This document proposes 64 fresh DEV parent worlds, 128 CHECK count reservations and a separate 16-parent CAL bank for scorer qualification. FINAL remains closed and unspecified. No conversation payloads, native responses, calibration labels or source-correct membership were authored here. The companion specification is `manifests/LANGUAGE_DEV_PROPOSAL.json`.

The immediate missing dependency is a qualified, identified grader and completed semantic review. A factual answer, valid JSON or exact reference match cannot establish that a response is fluent, natural, relevant and consistent with its conversational role. An agent-written rubric does not constitute human approval; no qualified LLM grader is assumed.

## Fresh parent worlds and coverage

Use eight strata with eight DEV parents each: explaining supplied facts, meaning-preserving rewriting, synthesizing supplied information, necessary clarification, supportive conversation, disagreement/compromise, recommendations under updated constraints, and follow-up/closure. Reserve 16 CHECK slots per stratum without creating or reading CHECK payloads. Give CAL two additional independent worlds per stratum; CAL is excluded from DEV model-quality denominators.

Each world must be newly authored around explicit facts, roles, timeline, task intent and decisive constraints. Assign its population before rendering, solving, token-fit review or model scoring. Renaming entities, reordering facts, paraphrasing a source template, switching perspective or writing counterfactual siblings does not create independent parents. Link all such derivatives to their original parent and keep them in one population. Score one primary request per DEV world. The proposed mix is four single-turn and four multi-turn worlds per stratum, totaling 32 each; multi-turn worlds use two to four user turns.

Do not resplit V3 preservation, turn its targets into prompts, or lightly rewrite the replay templates. Authoring records must explain what makes the new scenario substantively distinct from available historical TRAIN material and local evaluation exposure. Review latent structures as well as text hashes. Use authorized public TRAIN and hash/ID exclusion inventories without opening protected scored content. Unknown pretraining exposure and unresolved semantic equivalence remain disclosed limits; a new ID is no freshness certificate.

The scope is English short conversational quality. Domain labels alone are insufficient: each world needs a concrete communicative purpose and a semantic acceptance contract. Record required facts, permissible omissions, unsupported assertions to reject, appropriate uncertainty, speaker relationships, and tone/register constraints. Store multiple acceptable wording exemplars where useful, explicitly non-exhaustive. Review complete conversations for inconsistent names, timelines or surface wrappers before freezing them. Ordinary emotional support must not be reduced to a required stock phrase.

## Four independent rubric dimensions

Score each dimension from 0 to 2: 0 is a material failure, 1 is partially acceptable with a meaningful defect, and 2 meets the contract.

| Dimension | A score of 2 requires | Typical material failure |
| --- | --- | --- |
| Adequacy | The response sufficiently performs the requested function with supported content and appropriate uncertainty. | Wrong decisive facts, missing required substance, or failure to answer/rewrite/clarify as requested. |
| Relevance | The response directly serves the current conversational purpose without material distraction. | Off-topic material or a generic response unrelated to the user's actual need. |
| Fluency | Natural, readable, coherent language suited to the requested register. | Broken grammar, incoherence, conspicuous repetition or disconnected fragments. |
| Role consistency | Correct speaker perspective, established context and honest capability boundaries. | Invented assistant actions/experience/access, switching speaker roles or treating quoted text as governing instructions. |

A fluent wrong answer can fail adequacy; a factually correct awkward answer can fail fluency. An informal answer may be entirely fluent despite nonstandard punctuation. A concise response may earn full credit without matching a preferred sentence. A longer response is not automatically better; evaluate whether it adds necessary content or distracting material. Preserve all four scores. No weighted aggregate may let factual improvement cancel fluency or role-consistency loss.

Propose strict operational source-correct membership only when **all four adjudicated dimensions score 2** on the frozen unchanged-source response. This is a declared panel-specific acceptance rule, not a universal definition of language intelligence. Baseline-ineligible parents remain in the full panel and can later show repairs. A zero source-correct denominator means retention on that subset is undefined.

## Reviewers, calibration and blind paired grading

Preferred grading uses two independently scoring identified human reviewers and a designated adjudicator. Their identities, roles, prior exposure/conflicts, rubric version, calibration receipts and blinding assignments must be recorded. No reviewer has been assigned or approved by this proposal. Agent authorship or automated review is not human review. A model grader is an alternative only after a separate qualification binds its provider/model or checkpoint, runtime, grading prompt, template, decoding and costs, and demonstrates agreement with independently qualified reference labels plus a human-adjudication policy. Until then its annotations are diagnostic only.

Before any DEV baseline enrollment, construct the separate 16-world CAL bank. For each world, author two distinct fully acceptable responses and four deliberately defective controls, one primarily targeting each dimension. Thus there are 96 unique controls. Include a fluent factual error, a fluent off-topic response, a semantically correct but broken/repetitive answer, and a relevant fluent response that violates speaker/action boundaries. An independently designated rubric owner must verify control validity and freeze reference labels before calibration outcomes are seen. This proposal authors none of those responses or labels.

The proposed two-reviewer calibration budget is 224 rating records: each reviewer rates 96 controls plus 16 hidden repeat items. Freeze these thresholds before running calibration: at least 30/32 positive responses accepted, at least 15/16 controls for each defective dimension detected, at least 15/16 repeated full score vectors consistent, and at least 90% between-reviewer exact dimension agreement. These controls are necessary scorer evidence, not proof that difficult natural outputs will be graded perfectly. A failed calibration stops scored enrollment. Any revised rubric or reviewer preparation needs a recorded new revision and disjoint fresh CAL worlds; never tune it on DEV or relax thresholds silently.

Grade every frozen unchanged-source DEV response under a hidden model label first. Preserve each independent score, disagreement and adjudication. Hash the full baseline table and its source-correct membership before candidate results are graded. Every original parent remains, including failures; do not replace difficult prompts to manufacture a large source-correct denominator.

After freezing a candidate checkpoint, present each stored baseline/candidate pair in independently randomized A/B order per parent and reviewer. Hide checkpoint labels, recipe, baseline membership, other reviewers' grades and side identity. Preserve the exact response text and context. Reviewers score each response on all four dimensions before indicating A better, B better, tie or unjudgeable. Pair preference is secondary to the dimension outcomes and preservation losses. The adjudicator resolves disagreements while still blinded and records reasons; unresolved cases remain unjudgeable and visible.

Masked labels do not guarantee perfect blinding: graders may recognize a previously scored response or infer identity from its content. Record recognized responses and explicit model self-identification without altering the text. Prefer separately assigned pair reviewers where available; freeze those assignments before outputs rather than switching reviewers to obtain a desired result.

Repeated baseline grades during paired review diagnose grader drift; they cannot overwrite frozen source-correct membership. Propose suspending candidate interpretation if more than 4/64 parents change adjudicated baseline acceptance. Audit the scorer before unblinding, retaining all original labels and outcomes. Unblind only after grading, disagreements, adjudications and the drift decision are immutable and hashed. This makes reviewer drift visible instead of silently changing the preservation denominator.

## Native gates and bounded cost

Native baseline enrollment remains E0-gated. Bind the exact unchanged Titan receipt, current tokenizer/template/runtime and weight SHA256 `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`. Historical native scores, another runtime or replay labels cannot supply source-correct membership. E0 runtime, numerical, resource, save/reload and resume qualification must pass; semantic bank review, exclusions, rubric qualification and numerical run budgets must also be frozen before calls.

Keep total context within the initial 1,024-token qualified envelope. The requested prompt ceiling is 1,024, but reserving 128 generation tokens means an **effective prompt limit of 896 tokens including all template tokens**. Cap reference targets and generated replies at 128 tokens; measure fit with the exact native tokenizer. Never truncate a prompt or reference to preserve a label. Review or redesign an overlength world before bank freeze. A model output cut off at its generation cap is recorded as such and graded on the actual text, not silently repaired.

Propose one deterministic greedy response per parent, 64 attempts per model pass, with seed 20260928 and all other decoding/EOS/PAD details bound to E0 and frozen before baseline. That is at most 8,192 generated tokens and 57,344 prompt tokens per pass, or 16,384 generated tokens for baseline plus one frozen candidate. CAL controls are authored scorer tests, not additional Titan generations. There are no automatic retries, replacement prompts or response polishing. Infrastructure failures preserve their attempt/cost receipts and leave missing evidence; a separately registered retry decision must count every additional attempt and token. Numerical wall-clock/GPU ceilings are still unset blockers and must be derived from measured E0 throughput before launch. Existing memory guards remain unchanged.

Scoring effort is separately visible: baseline requires 128 independent reviewer output ratings before adjudication; one candidate paired pass requires 256 output ratings across two reviewers, including repeated baseline scoring, plus pair preferences and any adjudications. Record actual review time and every grader attempt. No paid grader/provider work is authorized here.

## Interpretation and release gates

Report each rubric dimension, all-dimension acceptance, source-correct losses, source-incorrect repairs, stratum results, missing/unjudgeable cases, termination/length diagnostics and complete costs. Use paired semantic-parent units. Proposed exploratory intervals use 10,000 parent-bootstrap resamples within strata, seed 20260928, and include discordant parent counts. Report source-correct denominators explicitly. Repeated DEV use must be logged, and its exploratory selection effects do not establish confirmation.

The 64-parent panel does not establish broad preservation or noninferiority; no noninferiority margin is supplied by this proposal. Multilingual quality, safety and long-context behavior remain outside its demonstrated scope. The separate long-context sentinel remains required for a broad retention claim.

CHECK contains only 128 count reservations now. It cannot tune hyperparameters, rubric, prompts or the grader. Its later creation/handling/opening needs the registered frozen checkpoint/seed gate and qualified grading; FINAL needs a separate confirmation plan and frozen winner. Neither protected population is read or authored by this design task.

The next concrete dependency is root review and assignment of semantic reviewers, a rubric owner and grader identities. Then author and review independent worlds, qualify CAL, freeze bank/exclusion/tokenization/run receipts, and enroll the unchanged source only after E0 passes. Until those steps actually succeed, language preservation remains **UNMEASURED**, with its review and scorer requirements explicitly **UNQUALIFIED**.
