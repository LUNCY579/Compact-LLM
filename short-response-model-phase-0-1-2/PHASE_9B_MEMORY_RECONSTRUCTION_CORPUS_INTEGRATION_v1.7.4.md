# Phase 9B Memory Reconstruction Corpus Integration — v1.7.4

## Purpose

v1.7.4 makes the dedicated `Phase9B_Memory_Reconstruction_Corpus_v1` a first-class Phase 9B research asset and carries the approved reconstruction behavior forward into Phase 10 and Phase 11.

The primary question is no longer only whether a student can answer from retrieved evidence. Phase 9B asks whether it can decide whether a retrieved memory is **currently applicable**, adapt it when assumptions changed, reject it when its preconditions fail, detect unresolved equal-authority conflict, and fail closed when current state is insufficient.

## Corpus audit

The normalized repository corpus contains:

- 300 external memory records;
- 600 human-accepted canonical reconstruction rows;
- 60 topic families;
- 440 train / 80 dev / 80 sealed-test rows;
- all five reconstruction decisions: 96 ACCEPT, 204 ADAPT, 120 REJECT, 60 CONFLICT, 120 INSUFFICIENT;
- 120 counterfactual state pairs;
- 1,800 curated preference pairs (three deliberate negatives per canonical row);
- 2,400 evaluator candidates (approved + raw replay + over-rejection + wrong-source variants).

Ten scenario families are represented evenly at 60 rows each: state match, premise invalidation, counterfactual adaptation, goal mismatch, multi-memory applicability, micro-adaptation, equal-authority conflict, insufficient current state, semantically relevant but inapplicable retrieval, and retrieval injection.

All 600 canonical examples contain non-empty `current_state`. All 300 master memory records contain `source_state` and `validity_conditions`. Topic families do not cross train/dev/test.

## Integration-time correction

The source package described counterfactuals as minimal edits. A strict audit found that 66/120 pairs changed small state-coherence fields in addition to the decisive causal field. v1.7.4 does not erase those legitimate changes or make a false one-field claim. The normalized package records:

- `decisive_fields`: the causal premise/goal edit;
- `changed_fields`: every concrete state field that actually changed;
- `supporting_changed_fields`: coherence fields changed with the decisive edit;
- `minimality_contract`: either a literal single decisive edit or a minimal decisive edit plus coherence fields.

This makes the counterfactual evidence auditable without changing its intended task semantics.

## Bounded validity-condition DSL

The dedicated corpus uses richer validity conditions than v1.7.3 understood. v1.7.4 adds a bounded, non-executable condition evaluator supporting:

- `all` equality premises;
- `any` alternative premises;
- `item` equality;
- named numeric predicates such as `candidate_price <= max_price`, VRAM feasibility, and inventory/reorder comparisons;
- `all_required_attributes_match`;
- legacy direct mappings/comparison fields.

Unknown free-form predicates are **never executed**. They return UNKNOWN, causing reconstruction to fail closed to INSUFFICIENT instead of guessing.

## Canonical reconstruction sequence

The conservative metadata baseline now applies the following order:

1. reject untrusted/instruction-like memories;
2. invalidate superseded memories for current-state requests;
3. detect changed high-value goals;
4. evaluate bounded validity conditions;
5. separate valid, invalid, and unresolved candidates;
6. detect conflict only among currently applicable memories;
7. ACCEPT exact state matches;
8. ADAPT when a valid principle survives changed state or invalid companions were filtered;
9. REJECT when all candidates are explicitly invalid;
10. INSUFFICIENT when applicability cannot be established from current state.

This deterministic path is a seed/audit baseline, **not** the learned reconstruction target generator.

## Training boundary

Only `approved_reconstruction` and `approved_answer` are ordinary supervised targets.

Curated preference negatives and evaluator candidates are deliberately wrong/naive alternatives. They are never concatenated into SFT. Preference alignment is an optional post-SFT/KD step.

Phase 9B materializes three canonical SFT views:

- explicit structured reconstruction;
- answer from approved reconstructed guidance;
- direct one-pass memory-to-answer distillation for Phase 10 compatibility.

## DEV-first / sealed TEST protocol

Phase 9B preserves the authored split.

- TRAIN: reconstruction SFT and optional progressive KD / preference alignment.
- DEV: compare memory-off, raw replay, prompted reconstruction, trained reconstruction, and direct candidate; measure counterfactual and negative-transfer behavior; freeze exactly one strategy.
- TEST: evaluate the already-frozen strategy only. TEST cannot choose or swap a reconstruction strategy.
- HUMAN GATE: only a named, frozen strategy can be materialized as the Phase 9B reference.

## Evaluation policy

Reference wording is an example, not a mandatory sentence. Functional behavior and grounding govern correctness.

The evaluator reports:

- decision accuracy by ACCEPT / ADAPT / REJECT / CONFLICT / INSUFFICIENT;
- source-ID precision/recall;
- safe conflict handling separately from explicit conflict recognition;
- scenario-family success;
- negative-transfer-expected success;
- complete counterfactual-pair success;
- decision sensitivity and answer sensitivity after decisive state edits;
- JSON/schema validity;
- raw replay rescue/harm;
- full reconstruction + answer token, latency, and VRAM cost.

The Phase 9B opportunity calculation remains proportional: measured reconstruction benefit must exceed its measured runtime cost. No universal absolute quality promotion floor was added.

## Phase 10 handoff

Phase 10 still performs architecture selection on the RAG DEV split and sealed final evaluation on the RAG TEST split. When an approved Phase 9B reference exists, Phase 10:

- requires the Phase 9B human gate;
- consumes the approved Phase 9B direct-compatible checkpoint;
- validates the integrated memory-reconstruction corpus before structural reduction;
- preserves Phase 9 as rollback provenance.

The dedicated test split is not used to pick a structural candidate.

## Phase 11 runtime handoff

Phase 11 now exposes reconstruction consistently in both the persistent service and the measured production benchmark:

- `off`: ordinary RAG;
- `prompted`: run the source-state protocol without claiming specialized reconstruction training;
- `trained`: use the approved Phase 9B behavior.

The benchmark records reconstruction validity, reconstruction latency, reconstruction prompt/generated tokens, and the complete two-pass end-to-end request cost. The answerability gate stays authoritative: reconstruction cannot convert unsupported evidence into permission to answer.

## Important limitation

This corpus is high-quality supervised/counterfactual/preference data for reconstruction cold start and alignment. It is not, by itself, a reproduction of MemHarness's outcome-linked GRPO training. Optional outcome/RL training should only be considered after supervised reconstruction is demonstrably useful under the project's measured opportunity policy.
