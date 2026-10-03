# Model-quality review, 2026-09-17 UTC

Latest completed experiment and preservation inventory:
[RESULTS_AND_PROGRESS.md](RESULTS_AND_PROGRESS.md). The clean control acquired
234/384 transitions against Supa14/384, with arithmetic69/150, retained39/47,
long context162/256 and literal abstention10/64. The rehearsal arm was worse on
the experimental retention gate. These are one-draw exploratory results.

No broad model-quality winner or absence of catastrophic forgetting is established.
The current dataset branch already contains all model-quality and reasoning-branch
commits. The useful immediate integration is stronger measurement and data hygiene,
not a second merge of inherited code or an unqualified architecture experiment.

## Concrete defects found and addressed

1. The new selection instrument's parents could enter new process/rehearsal
   training. Both builders now require the immutable selection file, record its
   hash/role/geometry, and bind exclusions into plan/data contracts. The controller
   rechecks actual parent disjointness before a worker loads a model.
2. Requested family doses could silently shrink after exclusion. Insufficient
   available parents now refuse preparation instead of silently changing the dose.
3. An eight-word overlap guard missed six-word percentage questions mislabeled as
   abstention. Text filtering now also checks digit-blinded question stems. The
   source remains intact and every dropped row is reported. Historical consumed
   contact is documented; its score impact has not been measured.
4. Process-dose and mixture runs skipped the in-run native qualification used by
   another lane. They now require it before optimization or checkpoint restore,
   retaining all numerical and resource limits.
5. Superiority at a declared nonzero margin was tested against zero instead of the
   margin. The test now shifts its null to the margin. In an exact eight-cluster
   counterexample, the previous p=.0078125 incorrectly supported the margin claim;
   the corrected two-sided p=1 does not. The symmetry/independent-cluster
   assumptions are explicit; historical results are preserved.

The pre-existing floor erratum and instrument-specific metadata edits were read
and preserved. The pre-existing deleted Supa selection receipt remains deleted in
the user's checkout; the committed historical receipt is retained in the isolated
clone and Git history. No unrelated changes were reset, stashed, or committed.

## Published core identities

The exact dense source is Supa 1.7, 596,049,920 parameters. Its weight SHA256 is
`962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
Published Opallo 1.0 has weight SHA256
`e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b`,
freshly rehashed here. Historical Opallo results improve arithmetic but lose source
successes and long-context performance. Supa-A1.1 is a different gated source and
must not be silently substituted for Supa 1.7.

## Experiment arrangement

[RRP015_PREREG.md](RRP015_PREREG.md) freezes a selection-clean, one-seed paired
pilot before outcomes. The prepared control contains 384 transition rows / 54,172
processed tokens / 28 updates. The treatment contains the same 384 transitions plus
648 rehearsal rows / 108,389 processed tokens / 55 updates. Maximum rendered length
is 162 tokens in both. Both exclude 768 scored parents and the treatment excludes
the newly detected short arithmetic contacts.

The isolated sparse clone lives at
`C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917` and has independent
Git refs. Source assets are shared read-only by absolute path. Full checkpoints and
run receipts belong to
`C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/rrp015-20260917`.
Native execution status and measured outcomes must be taken from the final results
record, not inferred from this preparation document.

Long-context corpus repair and a fresh independent confirmation instrument remain
necessary for the user's broader goal. The v2 LCR folder currently contains no data.
The large v1 delivery is unsuitable under the measured length and shortcut checks.

See [branch evidence](BRANCH_AUDIT.md), [dataset evidence](DATASET_AUDIT.md), and
the hashed census receipts for the scope and limitations of these conclusions.
