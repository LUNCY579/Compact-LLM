# Canonical TGSA × MTP programme path

## Decision

The programme keeps the existing branches, but assigns them non-overlapping scientific roles. No additional v5 branch is required and the two MTP histories must not be merged into one implementation namespace.

| Stage | Branch | Scientific role | MTP semantics |
|---|---|---|---|
| Phase 1 v4 | `research/tgsa-phase1-v4-competence-first` | Frozen competence and regime-selection boundary | none |
| Phase 2A | `research/tgsa-phase2-mtp-complementarity` | TGSA trajectory-coordinate mechanism identification | `sequential_teacher_forced_recursive_v1` |
| Phase 2B | `research/tgsa-mtp-rineya-complementarity-v1` | System factorial, exact verified decoding, and real-model integration | `parallel_independent_offsets_v1` |

Phase 2A is the canonical next scientific stage because the research priority is TGSA mechanism identification. Phase 2B is downstream and may validate system-level complementarity even if Phase 2A is negative, but it must not reinterpret a system interaction as evidence that TGSA's trajectory coordinate caused the result.

## Frozen Phase 1 boundary

The v4 registered regime family is rejected:

- `direct_lookup` and `pointer_chase` remained inert;
- `state_update` alone crossed from floor to saturation;
- equal total sample presentations produced different transition reliability at different effective epochs/data diversity;
- no TGSA mechanism hypothesis was tested;
- no further measurement is permitted under the v4 registration.

Therefore supplied old-band Phase-2 configurations are valid only for deterministic mechanics, gradient-path, initialization, schema, integrity, resume, and leakage tests. They are not admissible competence, mechanism, quality, complementarity, or speed evidence.

## Why the MTP implementations stay separate

### Phase 2A: sequential trajectory-conditioned future prediction

A later future horizon can consume:

1. the current causal hidden state;
2. the previous future-token embedding;
3. TGSA's existing trajectory or information-matched history coordinate.

Teacher-forced training and recursive inference therefore have distinct conditioning distributions. The design is useful for testing whether the trajectory coordinate contains incremental predictive information, but it is not the canonical deployment verifier.

### Phase 2B: parallel independent-offset prediction

Offsets `2..K` are predicted directly from the same causal hidden state. Future tokens are labels only; one predicted future token is not fed to the next auxiliary head. This supports a clean architecture × MTP factorial and exact target-model block verification, but a positive interaction does not isolate the trajectory coordinate.

Because the causal graphs differ, Phase 2A and Phase 2B outputs cannot be pooled, averaged, or described as replications.

## Required scientific identity

Every configuration, run manifest, checkpoint manifest, matrix, verifier result, and evidence fingerprint must identify:

```text
study_kind
mtp_semantics
required_ancestor_commit
source_branch
source_commit
development_only
confirmatory
registration_status
phase1_regime_status
obsolete_regime
task_scope_sha256
model_scope_sha256
objective_scope_sha256
competence_scope_sha256
unique_train_examples
sample_presentations
optimizer_steps
microbatch_size
gradient_accumulation
effective_epochs
device_fingerprint
environment_lock_sha256
study_fingerprint
cell_fingerprint
```

Allowed semantic identifiers in this programme are:

```text
sequential_teacher_forced_recursive_v1
parallel_independent_offsets_v1
```

A missing or mismatched identifier is an evidence-integrity failure, not a documentation warning.

## Competence and qualification gates

Two gates are distinct:

1. **External programme prerequisite** — confirms the predecessor scientific rationale and ancestry.
2. **Within-study qualification** — confirms that every primary cell on the actual Phase-2 task/model/objective scope is competent, non-saturated, sufficiently generalizing, and reliable across seeds.

An external Phase-1 report may never replace within-study qualification when Phase 2 changes the generator, families, difficulty band, depth, objective, tokenizer, split, or metric.

## Exposure contract

At minimum every study records:

```text
unique_train_examples
sample_presentations
optimizer_steps
microbatch_size
gradient_accumulation
effective_epochs = sample_presentations / unique_train_examples
```

Calibration varies data diversity and sample presentations as separate axes. Effective epochs are a required descriptor and calibration variable, not a universal transition law; architecture, objective, and task changes can move the transition.

## Initialization contract

Evidence reports separate booleans/metrics for:

- main causal-LM path neutrality;
- auxiliary-head output identity where intended;
- trajectory-control gradient connectivity;
- dormant-control parameter and initialization matching.

No single field called `initialization_exact` may collapse these distinct claims.

## Cache and conversion contract

The real-model path separates:

1. bitwise conversion identity on the same execution path;
2. cached-versus-uncached numerical agreement under a locked dependency/backend environment;
3. exact generated-token, verifier-decision, accepted-prefix, rollback, and final-sequence equality.

Tolerance must not be relaxed merely to make CI green. First lock PyTorch, Transformers, attention backend, dtype, device class, and deterministic settings; then diagnose any remaining drift.

## Promotion order

1. Keep the v4 result branch frozen.
2. Repair and keep both v5 branch CI suites green.
3. Write a new competence preregistration with learnable, non-saturated primary families, explicit depth/capacity controls, and separate diversity/presentation axes.
4. Obtain a genuine within-scope competence `GO`.
5. Run Phase 2A first for trajectory-mechanism identification.
6. Run Phase 2B for system-level quality and verified serial-round complementarity.
7. Reconcile the real-model additions against `research/tgsa-rineya-pretrained-retrofit` with a machine-readable port manifest and revalidated prerequisite artifacts.
8. Make wall-clock claims only after cache-aware end-to-end target/draft timing.

## Branch lifecycle

- `research/tgsa-phase1-v4-competence-first`: frozen evidence branch; no Phase-2 code merges.
- `research/tgsa-phase2-mtp-complementarity`: active Phase 2A mechanism branch and PR review surface; never merge into the frozen v4 branch.
- `research/tgsa-mtp-rineya-complementarity-v1`: active Phase 2B system branch; downstream of the programme boundary, not a replacement for Phase 2A.

Preserve both branch histories for provenance. Do not delete, squash together, or relabel their artifacts under one generic “TGSA+MTP” identity.
