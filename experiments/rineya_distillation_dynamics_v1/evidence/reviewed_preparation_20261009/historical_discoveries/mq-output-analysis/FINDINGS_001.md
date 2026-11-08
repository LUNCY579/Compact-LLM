# MQ output mechanics on open historical DEV

The corrected raw MQ outputs show changes in visible answer content and a small
exact-string formatting contribution. The questions are unavailable in the
selected records, so this review does not verify task-solving competence.

The original archive's `final_dev.json` record schema contains ID, family,
prediction, generated IDs, prompt-token count, scores, termination and cap state.
It contains no question, reference, format tag or parse tag. The already archived
`prepared-003/plan.json` DEV records contain declared reference strings and exact
accepted strings, plus prompt token IDs. This analysis used those in-record
references and did not decode token IDs or follow dataset paths.

The historical exact grader checks full stripped-string membership. Its corrected
answer lists include both bare and trailing-period forms. Diagnostic string replay
matches every archived score across all 64 baseline and paired-arm records in all
three orders. The historical grader and results remain unchanged.

| Population | Baseline correct | Response correct, orders 17/23/29 | Whole correct, orders 17/23/29 | Source-correct support |
|---|---:|---:|---:|---|
|Historical scored 51|3|13 /13 /13|7 /7 /8|3 source-correct cases retained|
|Full-pool overlap exclusions:14 remain|0|5 /5 /4|1 /1 /1|Undefined: zero source-correct cases|
|Consumed-only overlap exclusions:29 remain|2|7 /7 /6|3 /3 /3|2 source-correct cases retained|

The 13 historically unscored rows stay excluded from every accuracy calculation;
no retrospective gold is invented. The 14-row and 29-row filters are post-hoc
overlap diagnostics, not new prospective confirmation populations.

The 14-row paired response advantage is 4/4/3 by order. Its 11 positive paired
occurrences cover five distinct IDs, with this mechanical decomposition:

| Order | Changed numeric value | Opposite YES/NO polarity | Same option token, different surface form |
|---|---:|---:|---:|
|17|2|2|0|
|23|1|2|1|
|29|1|1|1|

The two surface-form occurrences are the same constraint-selection case:
`Option B.` versus `B.`. Both visibly select B; only the longer form meets the
historical exact contract. Without the original question, this review cannot say
whether the prompt expressly required that form.

The numeric occurrences compare different values against the declared references,
such as `18.` versus `12.` or `16.`, and `65 minutes.` versus `75 minutes.`. The polarity
occurrences compare YES versus NO. These are high-certainty differences in output
content/reference agreement, while semantic correctness remains
`NOT_ADJUDICABLE_WITHOUT_QUESTION`. They do not establish that a correct value was
obtained by solving the task instead of a source cue or a flawed reference.

All paired discordant outputs terminated; none hit the generation cap. The
machine report retains complete response, whole and baseline texts only for
paired discordant rows, with generated lengths, numeric strings, reference
matches and conflicting-number diagnostics. Full-cohort mechanical metadata is
retained without full output texts.

The complete 51-row population also retains one recurring counterexample:
the estimation-sanity case has `No.` under whole supervision and `Yes.` under
response supervision in every order, with the declared reference `No.`.
It is outside the 14-row subset, so filtering does not erase its original result.

Artifacts:

- `MQ_OUTPUT_DIAGNOSTIC_001.json`, SHA256
  `0bb6c53ece194fbd5782892be7d05a3fd30bec96d1c0073c1cc771215b86c8fe`.
- `MQ_OVERLAP_OUTPUT_DIAGNOSTIC_002.json`, SHA256
  `6bd0c2101b86166c5a08d706991afda6bce70b946bf58a38f31cc7fcddb8ab9d`.

Neither artifact grants native execution, training, new grading, bank access or
promotion permission. The historical source remains Supa1.7; these results do
not establish present Titan capability.
