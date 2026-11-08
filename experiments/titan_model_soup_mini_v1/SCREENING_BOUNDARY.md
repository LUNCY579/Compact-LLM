# SOUP-MINI-001: executable offline screening boundary

Date: 2026-09-22. Audited parent: `01bb8b9cc921be5c9dc51b96fb1ecbe394fbf88a`.
This branch was no longer empty when inspected. That parent already added the builder,
source pins, nine-configuration proposal, descriptive scorer and 66-test historical receipt.
This additive follow-up preserves those files and closes only the **offline selection and
report identity boundary**. It does not close native qualification or grant a model run.

## What was missing

`score.py` intentionally accepts externally scored rows and any syntactically valid checkpoint
hash. It compares two runs but does not establish which registered arm produced either one,
apply the proposed selection rule, measure false abstention, or lock a CHECK candidate.
That is adequate for its descriptive purpose, not an executable Phase-A selection gate.
`screen.py` supplies that narrower gate without adding a model loader or changing the scorer.

## Frozen design, unchanged coefficients and budgets

The four endpoints remain exact published Supa, stored Titan, FP32 Titan-Sparse and stored
Opallo. The five mixtures remain `supa_titan_50`, `supa_sparse_50`, `supa_opallo_50`,
`titan_opallo_50` and `sparse_opallo_50`. Every coefficient is exactly 0.5. There is no
coefficient search, additional seed, local-child enrollment, new dataset, optimizer or training.
The stored Titan/Opallo artifacts do not become their unrounded parents when loaded FP32.

CAL imports exactly nine configuration files, at most 256 rows each. Only the two Titan/Opallo
soups are selectable. CHECK imports exactly the four unchanged endpoints plus the **one locked
CAL candidate**, at most 256 rows each. Thus the existing maximum remains 3,584 scored
completions, not an execution allowance. Native collection still requires the separately
reviewed 4,096 total-token / 64-new-token, greedy, batch-one, common FP32 contract and its
own resource budget and grant. This offline importer does not count hidden native retries.

Both mixtures are compared to the exact full endpoints and both appropriate half-strength
component controls on CAL. For sparse soup these are `supa_sparse_50` and `supa_opallo_50`.
For dense soup they are `supa_titan_50` and `supa_opallo_50`.
**The five-configuration CHECK budget has no attenuation controls.** CHECK therefore can
assess held-out retention and capabilities against endpoints, but cannot independently
confirm the attenuation-controlled interaction measured on CAL. Do not expand CHECK to
repair that limitation without a new registration and budget.

## Executable selection rule

The companion `screen-freeze.example.json` is intentionally UNBOUND and must refuse.
Populate and externally freeze it before any CAL result is inspected. It binds the exact
existing protocol and implementation hashes; all nine checkpoint hashes; five BUILD receipt
hashes with input-source pins, sidecars, output dtype and coefficients; actual runtime,
scorer, decoding, qualification-review and ancestry-audit identities; and both bank identities.
The same review hashes are required in every imported run. These are consistency checks,
not a signature authority or certification that a referenced review is truthful or complete.

Each bank's `row_manifest` contains only `id`, explicit `parent_id`, `domain` and boolean
`answerable`. It must not contain prompts, gold answers or predictions. The mini selector
requires one canonical view per globally unique parent, disjoint CAL/CHECK IDs and parents,
and all six registered domains. Repeated-view datasets remain usable by `score.py` for
separately labelled descriptive reporting, not this mini selection path. A review must bind
these manifest identities to the actual dataset bytes and check ALL fine-tuned ancestor
training ledgers; identities alone do not prove fresh, uncontaminated content.

Freeze positive source-correct minimum counts for every domain and donor minimum-correct
counts exceeding the frozen shortcut counts. All are deliberately null in the template;
choose them from the instrument design, not observed model outcomes. They are competence
and adequacy floors, not statistical significance or noninferiority tests. A missing or too
small source-correct denominator blocks the screen rather than passing retention vacuously.

For each candidate, require at least 95% of its relevant Titan endpoint's transition correct
count and at least 95% of Opallo's arithmetic correct count. Integer arithmetic implements
`20*candidate >= 19*donor`. Protect Supa's source-correct items in **all six domains**, including
transition and arithmetic: losses in each must be no more than `source_correct // 20`.
This explicitly resolves the original proposal's undefined phrase "each retention domain"
conservatively without inventing another percentage. New gains never cancel source losses.
Total false abstentions on answerable rows must not rise; report the per-domain breakdown.
The `abstained` boolean must follow the frozen scoring policy, not an unreviewed substring rule.

Among eligible candidates choose fewer total source-correct losses, then higher mean
of the two donor-normalized acquisition scores, then the primary sparse soup. Fractions
avoid floating-point tie ambiguity. If neither is eligible, select **none**. A screen pass
means only that this operational exploratory rule passed. The 95%/5% tolerances remain
screening proposals, not user-approved product tolerances or formal noninferiority margins.

## Commands after independent qualification and result collection

Run the new tests without native dependencies or checkpoint access:

```powershell
cd experiments/titan_model_soup_mini_v1
python -m pytest -q tests/test_screen.py
```

Every run JSON extends the README's scored-run schema. Metadata additionally requires
`arm_id`, `freeze_sha256`, `asset_bundle_sha256`, `qualification_receipt_sha256`,
`ancestry_audit_sha256` and `evaluation_dtype: "float32"`. Every row additionally requires
boolean `answerable` and `abstained`. Tokenizer SHA means the pinned `tokenizer.json` digest;
asset-bundle SHA is `screen.digest(soup.ASSETS)`. Run filenames are exactly `<arm_id>.json`.
The directory must contain only the nine CAL or five CHECK run JSON files.

```powershell
python screen.py CAL --freeze screen-freeze.local.json --runs local-runs/CAL --output CAL_SELECTION.json
# Externally record the printed CAL_SELECTION.json SHA before any CHECK predictions are opened.
# If selected_candidate is null, stop; do not run CHECK and do not promote the least-bad arm.
python screen.py CHECK --freeze screen-freeze.local.json --runs local-runs/CHECK --lock CAL_SELECTION.json --lock-sha256 <EXTERNALLY_RECORDED_SHA256> --output CHECK_REPORT.json
```

Outputs use exclusive creation and cannot overwrite a lock or previous report. Keep them
outside run directories. CAL never reads CHECK prediction files; it sees only the already
frozen identity manifest. CHECK requires the externally supplied CAL-lock hash and never
reselects a candidate, even when the locked candidate fails. Its assessment is descriptive
held-out screening, not a new search. A hash cannot prove nobody inspected CHECK earlier;
external chronological sealing and operator process remain required.

## Interpretation and unresolved gates

A properly collected result could establish a fixed-artifact, fixed-bank preservation /
capability trade-off under matched conditions. Source losses, new gains, all six domains,
full endpoints and CAL dilution controls remain visible. A gain over one endpoint alone is
not proof of transferring a donor-exclusive skill: inspect both donors and both controls.
Small deltas, positive aggregate scores or nonsignificant p-values cannot establish
orthogonal skills, lossless composition, equivalence, broad intelligence or release readiness.
Raw paired p-values remain descriptive; no multiplicity-adjusted discovery claim is made.

No real local source weights, fresh CAL/CHECK banks, native runtime or independently reviewed
qualification/ancestry receipts were available to this follow-up. No native loads, decoding,
checkpoint builds, training, grant changes, lease changes or explicit CI dispatch occurred.
`screen.py` hashes JSON/code/receipts only; it does not rehash checkpoint bytes or verify an
external scorer's correctness. Exact local checkpoint preflight, endpoint decode parity,
full-ancestry contamination review, frozen instrument/verifiers and a separate SOUP-MINI-001
resource/native-read grant remain prerequisites. Historical spent/quarantined banks and
other experiments' authorizations are untouched. All reports keep native authorization,
training authorization, model-quality and release flags false.
