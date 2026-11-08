# Audited research publication, 2026-10-09

This archive preserves the October 5 closeout, quality implementation, partial
native pilot and subsequent offline discoveries. Publication adds no model run,
training, promotion, serving change or permission. **No new intrinsic quality
gain has been verified.** The original negative recurrence result remains intact.

## Evidence and decisions

| Study | Actual evidence | Decision |
|---|---|---|
| SBHM-001 recurrence | Completed four-arm, one-site, seed-17 synthetic DEV matrix: C 172/224 distant, D 171/224; zero D-only correct rows | Justified hold on recurrence spending |
| Balanced update ordering | Implemented and audited 512-slot control/candidate design; three completed native qualifications and one 672-row frozen-anchor measurement | Incomplete; ordering hypothesis untested |
| Selector and answer-shape analysis | Reanalysis of saved outputs, independent record oracles, shortcut controls and engineering fixtures | Hypotheses and measurement requirements; no native gain |

The original C/D synthetic retention total is 29/64 correct: 18 retained
source-correct cases plus 11 repaired source-incorrect cases. Separately, each
damages **4 cases out of 22** source-correct cases. Repairs do not offset damage. The fresh quality anchor
has a separate denominator: **21/64 source-correct retention cases across five
skills**, with no source-correct coverage for string-rule, temporal or unit
conversion tasks. All 14 source-correct old update cases lost by each B/C/D after adaptation
form a third denominator; they are not the 22 retention cases.

The quality pilot stopped at the unchanged 6 GiB runtime free-RAM guard during
the control-training stage. There is a request, but no completed training receipt
or delta, trained DEV measurement or comparison. Optimizer-step count, exact
failure phase and the failing numeric RAM sample are unknown. Admission passing
does not establish training fit. No retry followed this stop.

Primary records, in reading order:

1. [Audited SBHM closeout](sbhm_audit/AUDITED_CLOSEOUT.txt),
   [closeout receipt](sbhm_audit/closeout_receipt.json) and the original
   [completed native receipt inventory](../native_20261004/completed_receipts/manifest.json).
   Final passing audit revisions are
   [provenance v2](sbhm_audit/provenance_receipt_v2.json),
   [pairing final](sbhm_audit/pairing_audit_final.json) and
   [resources v2](sbhm_audit/resources_audit_v2.json).
2. [Quality implementation report](quality_pilot/IMPLEMENTATION_REPORT.txt),
   [reviewed corpus](quality_pilot/corpus-review.json),
   [final corpus manifest](quality_pilot/corpus-v2/manifest.json) and
   [original exact proposal](quality_pilot/proposal/proposal.json).
3. [Partial native closeout](quality_pilot/NATIVE_CLOSEOUT.txt),
   [stop receipt](quality_pilot/native-stop-receipt.json),
   [independent stop audit](quality_pilot/independent-native-stop-audit.json),
   [independent anchor audit](quality_pilot/independent-native-anchor-audit.json)
   and [raw 672-row anchor receipt](quality_pilot/proposal/native_run/anchor-evaluation/receipt.json).
4. [Final intrinsic-quality review](pattern_research/INTRINSIC_QUALITY_REVIEW_FINAL.txt),
   [qualified selector controls](pattern_research/selector_controls_qualified.json),
   [answer-shape caveat](pattern_research/selector_controls_cue_addendum.txt),
   [numeric-lexeme caveat](pattern_research/selector_controls_lexeme_addendum.txt),
   [shortcut opportunities](pattern_research/shortcut-opportunities.json) and
   [claim/falsification ledger v2](pattern_research/claim-falsification-ledger.v2.json).

Earlier drafts remain labelled by their original statuses and are superseded by
the final revisions above. Historical native logs and receipts are unchanged.
Earlier implementation reports describe their own delivery date, not today's
execution state. A draft selector failure was an auditor histogram-key error,
not a failed native model run; its omission hash is recorded.

## Findings at their measured level

Old update supervision rewards the first record in 6/8 TRAIN worlds, versus
3/16 DEV worlds. Adapted B copies it on 31/32 distant updates; C/D on 32/32.
This is concentrated in revision selection: saved non-first lookup, identifier,
number and join controls often succeed. Identifier/number answer-shape cues
also permit target-blind shortcuts, so these controls do not prove robust
entity/field binding or identify an internal algorithm.

The fresh anchor emits integers on all 384 update rows, failing the required
string-valued JSON contract; strict and typed correctness are both zero.
Expected code lexemes occur in 198 outputs. Order stability can include stable
wrong answers. Different old/new renderers and payload distributions prevent a
causal cross-corpus quality comparison. Decimal parsing must preserve printed
lexemes: e.g. `48.00` becoming `48.0` is an analysis artifact, not a model error.

A fixed type-preserving scalar-to-object diagnostic on saved outputs changes
55/672 strict/typed correct to 123/672, repairing only general and present-fact
contracts. It repairs no updates or retention cases. This is a hypothetical
interface transform, not a trained-model improvement. Missing-field queries
still emit non-null values, and the dataset's record-count shortcut is an
opportunity rather than proof that the model uses it. Relational fixture
receipts test engineering/oracle logic; they are never Rineya capability evidence.

One source, one seed, synthetic DEV, repeated exploratory inspection and
dependent world variants constrain all conclusions. Resource results are
instrumented checked-boundary/sample observations, with no continuous headroom,
energy, optimized speed or broad retention claim.

## Portable inspection and integrity

[COPY_PROVENANCE.json](COPY_PROVENANCE.json) maps each byte-identical archived
file to its original local path, SHA256, size and evidence kind. It also records
hashes/reasons for omitted working copies and operational files.
[MANIFEST.json](MANIFEST.json) seals this publication's files and package code.
The package `.gitattributes` disables text conversion to preserve receipt and
corpus hashes across checkouts.

Run the standard-library-only check from any checkout:

```text
python -B experiments/rineya_spikingbrain_hybrid_memory_v1/evidence/research_closeout_20261009/verify_archive.py
```

The check reads local published bytes only, writes nothing, imports no model
libraries and executes no historical script or proposal. It verifies copied
files, publication files, current frozen code inventories, corpus references
and the 50 original native snapshot hashes. A hash-consistent archive supports
artifact integrity, not independent observation of historical execution.

The publication's [fixture receipt](publication_checks/publication-tests-receipt.json)
records **222 passed, zero skipped** in the pinned CPU environment. It preserves
two earlier operator setup failures (wrong scratch root, then missing scratch
parent) alongside the final passing JUnit record. No study code or guard was
changed to obtain that pass. Tiny random models and simulated receipts test
engineering mechanics, not native capabilities or resource fit.

Raw historical files intentionally retain machine-specific absolute paths,
environment identities and original LF/CRLF encodings. Use this relative index
and the copy map for inspection. They are not relocatable native-run configs.
Weights, binary deltas, teacher targets, environments, caches and operational
grants remain local. Their recorded inventories are not their payloads, and
expired grant references convey no authorization. No protected bank content is
included; source metadata bearing confirmation names was hashed as inventory
bytes only during the original audit.

## Next concrete decision

Keep recurrence spending on hold. Balanced ordering remains the smallest
registered untested intrinsic hypothesis; its unchanged budget is five
qualifications, two 512-update adaptations, three 672-row measurements and one
comparison. The [reviewed repeat proposal](pattern_research/proposed-repeat/proposal.json)
and [readiness policy](pattern_research/launch-readiness-policy.json) are plan-only,
unexecuted, and carry no grant. Any later attempt requires a new exclusive output
root, fresh ownership/headroom checks and separate exact native authorization.
The added operator headroom requirement does not weaken existing guards.

The registered advancement gate still requires at least 26 net additional
typed-correct cases on 256 greatest-not-first rows, improved middle/last and
all-six-order groups, no greatest-first/JSON or present/absent regression, and
zero newly damaged source-correct retention cases separately under strict and
typed scoring. Passing permits further review only; failing returns to hold.
Publication neither retries this pilot nor starts another campaign.
