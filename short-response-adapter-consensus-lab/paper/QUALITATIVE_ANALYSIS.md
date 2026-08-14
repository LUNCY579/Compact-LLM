# Qualitative Analysis Protocol

## Selection rule

Qualitative cases must be selected by deterministic rules from machine-defined
buckets before their prompt text is inspected. For each bucket:

1. construct the complete eligible example-ID set;
2. sort by SHA256 of `phase2-qualitative:<bucket>:<example_id>`;
3. take the first three cases, subject to domain/task diversity constraints
   declared before reading them;
4. report all selected cases, including awkward or uninformative ones;
5. record any replacement and objective exclusion reason.

This prevents hand-picked anecdotes from masquerading as frequency evidence.

## Required buckets

- Hard rescues Track A.
- Hard harms Track A.
- Hard beats probability mean.
- Probability mean beats hard.
- All experts agree and the final sequence is correct.
- All experts agree and the final sequence is wrong.
- A 4-3 vote.
- A 3-2-2 vote.
- Tie resolution followed by a correct sequence.
- Tie resolution followed by an incorrect sequence.
- Correct abstention.
- False abstention.
- Earliest hard/soft divergence followed by a hard-only success.
- Earliest hard/soft divergence followed by a probability-only success.
- A minority expert proposes the gold local token under teacher forcing.

## Case template

For every case record:

```text
Bucket:
Example ID:
Selection rank/hash:
Dataset and role:
Prompt:
Gold answer:
Decision class:
Individual expert proposal(s):
Top-1 probability / margin:
Vote counts and coalition:
Tie/EOS state:
Hard output:
Comparator output:
Official exact score:
Secondary semantic-audit label (if sampled):
Interpretation:
Alternative interpretation:
```

## Interpretation constraints

- One example can illustrate a measured pattern but cannot establish its
  prevalence.
- Free-running divergence does not identify the locally correct token unless a
  gold-prefix teacher-forced record exists.
- "Overconfident mistake" requires the expert's selected token to be wrong at a
  position with a valid gold target, not merely disagreement with the committee.
- Formatting errors and semantically correct exact-match failures must be
  labeled separately.
- Do not expose private or sensitive prompt content if provenance review finds
  any; the current deterministic benchmark is expected to be synthetic, but
  that expectation must be verified.

## Machine source

The initial candidate sets will be derived from:

- `results/phase2_mechanism/existing_results_analysis.json`;
- `results/phase2_mechanism/teacher_forced_dev_analysis.json`;
- the frozen raw JSONL and official dataset row keyed by `example_id`.

Selected cases and hashes will be written to a separate
`results/phase2_mechanism/qualitative_selection.json` before narrative text is
added here.

## Interim deterministic case readings

These examples were read only after the selection artifact had fixed three
cases per nonempty bucket. They illustrate measured patterns; their frequency
must be taken from the complete bucket counts in
`reports/PHASE2_FAILURE_ANALYSIS.md`.

### A hard-vote rescue driven by specialization

In `v22_general_insufficient_information_0098`, the prompt asks for a maximum
when one of three values is unknown. Track A answers `Beacon.`, while hard K=7
returns the exact abstention target. At the first token, all seven experts
propose different tokens. The abstention expert alone proposes `I` with top-1
probability 0.999; the hard tie rule (sum log probability, then smallest token
ID) selects it. This is a genuine sequence-level rescue, but it is not evidence
that plurality alone found the answer: the confidence-sensitive tie rule and a
specialized abstention proposal are both essential in this case.

### The same tie mechanism can help or harm

For `v22_personal_conflicting_evidence_0095`, the first-token vote is 2-2-1-1-1.
The tie rule selects `I`, producing the correct abstention. The abstention
expert's `I` probability is 0.999, whereas Track B's second vote for `I` is
0.279. In contrast, `v22_creative_dependency_order_0093` has the same
2-2-1-1-1 shape, but the correct output is `Delta.`. Track A proposes `Delta`
with probability 0.400, while the abstention expert proposes `I` with 0.997;
the tie rule selects `I` and the final answer is wrong. These paired examples
motivate the calibrated tie-break and tie-ablation experiments.

### Hard voting can suppress a dominant but wrong expert

In `v22_finance_constraint_satisfaction_0099`, Track A and Track B both propose
the correct first token `A` (probabilities 0.189 and 0.253). Every other expert
proposes a different token, including an abstention expert assigning 0.999 to
`I`. Hard voting selects the only token with two votes and completes `A.`;
probability averaging selects `I` and abstains incorrectly. This case is a
direct illustration of the robustness hypothesis: counting experts prevents
one extreme probability from outweighing two independent top-1 proposals.

### Majority agreement can still preserve a shared mistake

In `v22_finance_insufficient_information_0092`, the correct response is an
abstention. Track A, Track B, and the general expert all propose `F`, so hard
voting emits `Falcon.`. The abstention and social experts instead propose `I`,
and probability averaging returns the exact abstention answer because the
abstention expert assigns essentially all mass to `I`. Thus hard voting's
calibration robustness becomes a liability when several experts share the same
content error.

### Committees can break a correct specialist

In `v22_media_constraint_preserving_transform_0093`, Track A and Track B both
propose the correct option token `A` with probabilities 0.649 and 0.528.
General, procedure, and social instead propose `C`, giving hard voting a 3-2
plurality and the wrong output `Cedar.`. Probability averaging makes the same
mistake. This is a concrete correlated-error failure: adding experts does not
help when three adapters share a plausible but task-inappropriate lexical
continuation.

## Gold-prefix local-decision cases

Source: E-QUAL-TF. A separate deterministic pre-narrative pass over all 4,050
DEV examples and 18,760 target-token decisions selected three examples per
nonempty bucket. At least one qualifying token occurs in 1,095 examples where a
minority expert proposes gold but is outvoted, 294 where gold is a tied vote
leader but the confidence tie-break rejects it, 157 where that tie-break selects
gold, 90 with premature EOS, and five where hard fails to select gold EOS. The
complete 27-case record, prompts, token proposals, confidence, hashes, and
eligibility counts are in `reports/PHASE2_TEACHER_FORCED_CASES.md` and
`results/phase2_mechanism/qualitative_teacher_forced_selection.json`.

### A locally correct specialist can be outvoted decisively

In `v22_travel_insufficient_information_0081`, the first gold token is `I` for
the required abstention. The abstention expert alone proposes `I` at probability
0.999, while five experts propose `Delta`; both hard and probability pooling
select `Delta`. This is a direct minority-correct case, but it also shows why
specialist confidence alone cannot repair correlated majority error.

### Unique plurality can suppress a wrong high-confidence dissenter

In `v22_health_constraint_preserving_transform_0085`, Track A, Track B, and
General vote for the gold token `A`; two experts vote `F` and two vote `I`.
Hard's unique 3-2-2 plurality selects gold even though Abstention assigns 0.958
probability to the wrong `I`. This isolates the scale-robust part of hard voting:
the confidence tie-break is never consulted.

### The confidence tie-break can reject a tied gold leader

In `v22_writing_capacity_fit_0088`, `Beacon` and `C` each receive two votes,
including gold votes from Track A and Track B. The confidence tie-break selects
`C`, so hard and probability pooling are both locally wrong. Across DEV, gold is
a tied leader but loses this secondary decision in 294 examples, confirming that
hard voting is not purely ordinal at its tie boundary.

### EOS errors occur in both directions

In `v22_coaching_filter_then_rank_0081`, six experts select EOS when the gold
token is the final period of `Harbor.`, a clear local premature-stop decision.
Conversely, in `v22_math_constraint_preserving_transform_0083`, gold is EOS and
ties another token at two votes, but the hard tie-break selects `3`; raw
probability pooling selects EOS. These examples explain why restrictive global
EOS overrides performed poorly even though individual EOS failures are real.

### Hard and soft local rescues are bidirectional

In `v22_devops_constraint_preserving_transform_0080`, three experts give the
gold `A` a unique plurality while the 0.955-confidence abstention proposal pulls
probability pooling to `I`; hard is locally correct. In
`v22_consumer_filter_then_rank_0089`, period and EOS tie 3-3; hard resolves to
premature EOS while probability pooling selects the gold period. The full cache
contains 204 examples with a hard local rescue and 313 with a probability local
rescue. These counts are opportunity/bucket counts, not free-running sequence
effects.

The original free-running cases and these gold-prefix cases are complementary.
The completed same-split comparison in `selected_dev_free_run_analysis.json`
shows that the method ordering is essentially unchanged between gold-prefix and
free-running evaluation. These cases therefore illustrate where discrete
decisions differ, but they do not justify a claim that hard voting generally
repairs autoregressive trajectories. The cross-method interpretation and exact
paired effects are consolidated in `final_synthesis_facts.json`.
