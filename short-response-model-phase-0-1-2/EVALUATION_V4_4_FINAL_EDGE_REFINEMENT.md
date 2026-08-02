# Quality Evaluator v4.4 — final-edge refinement

## Scope

Evaluator v4.4 is a concentrated correction to v4.3. It preserves the existing architecture:

1. universal mechanical and decoding-integrity checks;
2. explicit case-level response contracts;
3. category-neutral prompt-function scoring;
4. proportional quality loss inside the shared relative-opportunity engine.

It does not restore category-driven expectations, reference-answer grading, catch-all semantic labels, or absolute promotion floors.

## Problems corrected

The final Phase 2B terminal review found six narrow failure families:

- explicit uncertainty such as `not necessarily` was not recognized;
- plain-language rejection such as `not acceptable` did not satisfy a required refusal;
- capability limitations expressed with `cannot verify` did not satisfy a visibility boundary;
- scope limitations such as `not a realistic one-step replacement` were treated as complete misses;
- repeating a known location could hide an invented status claim, for example `East hall is not fully ready`;
- `compression reduces accuracy` was rewarded as a valid overview even though it states a contingent trade-off as an unconditional fact.

## Implementation

### Narrow semantic cues

`scripts/common/observable_quality.py` now supports human-authored `any_cues` and `all_cues` rules. The available cues are intentionally narrow:

- `uncertainty`
- `rejection`
- `capability_limit`
- `scope_limit`
- `unqualified_compression_accuracy_loss`
- `unsupported_update_status`

A cue has no global scoring effect. It operates only when an individual benchmark contract or prompt-adequacy rule explicitly requests it.

### Contract refinements

The v4.4 Phase 2B probe recognizes:

- `not necessarily`, `may`, `might`, `possibly`, and similar uncertainty forms;
- `not acceptable`, `unacceptable`, `refuse`, and similar rejection forms;
- `cannot verify/identify/see/tell` as capability limits;
- `not realistic`, `not feasible`, and one-step replacement objections as partial scope boundaries.

### Unsupported update claims

Location-update acknowledgments receive high credit for confirming the new location. They no longer receive full credit merely for mentioning the location while inventing an unsupported status such as closed, unsafe, unavailable, delayed, or not ready.

### Compression trade-off precision

The new concrete minor label is:

`unqualified_compression_accuracy_loss_claim`

It applies only when a reply states that compression reduces accuracy unconditionally. Qualified language such as `can reduce accuracy` or `may involve an accuracy trade-off` remains valid.

## Regression evidence

The comprehensive calibration contains:

- 620 total cases;
- 150 official references;
- 31 reporting categories;
- 61 concrete scored labels;
- 150 human-adjudicated coherent replies;
- 150 real collapsed replies;
- 25 targeted regressions;
- 620/620 passing expectations.

The final terminal-log review re-scored 149 complete generation rows available in the pasted log. The first generation row was absent from that pasted excerpt, so the report does not claim a 150-row rescore.

## Governing principle

A short response should be rewarded for preserving useful meaning, facts, context, and boundaries. A rule must not reward a keyword when the proposition containing that keyword is wrong, and it must not reject a valid paraphrase merely because it differs from a reference sentence.
