# Acquisition-panel operations review 2

2026-10-02; non-builder reviewer `/root/runtime_audit`, builder `/root/data_audit`.
**Exact source boundary and retained evidence association conditionally accepted;
population and acquisition claims are not accepted by this report.** AP-OPS-1-001
is closed on these bytes. No other consequential operations defect was reproduced
within the scoped second review. This is round2 of maximum3, not approval for a
fourth repair, another validation batch or native/training work.

I read the preserved first candidate, OPS1 and complete adjudication/scope before
freeze; mutable candidate2 was not inspected then. After the exact freeze I read
its full source delta, new tests/contract fields, planned wrapper and retained
raw artifacts. I did not implement acquisition code or run its full suite/core/
scorer/oracle batch. I previously authored the scope/continuation assessment;
shared model family, filesystem, user privileges and supplied semantic assumptions
limit independence. Separate code scrutiny is not independent world truth or OS
secrecy. Only this report is written; accepted optimizer/token bytes and quarantine
records are unchanged. No model/tokenizer/framework, host DB, scheduler, new
panel/holdout or source mutation occurred.

| Frozen artifact | SHA256 |
| --- | --- |
| acquisition_data.py | 90f0cf990e736f4d455ceba07b93ed06957ef68759b3b0e46fe0fef4af6b4943 |
| tests/test_acquisition_data.py | 1329e3ae43e76d6062ab124a1005778aeb4556a08d2d437977db286ca7ac32b0 |
| contracts/ACQUISITION_PANEL_V1.json | 7fd4246c00583dd4e8363022ff04002768d5779af27fe72bf2689693e17c5968 |
| Named root validation REPORT.json | c99ad46b2a01f9625246d7fa3c2d045b03db3fbc40a4094e2cdb2cf3f32a90f6 |

## Boundary repair and independent targeted controls

`validate_rules` and `validate_parameters` now require exact key counts and
primitive types/values before JSON on public/parameter entry paths. `root_identity`,
`construct`, `selection_recipe` and `assess` use those checks. Ancestry and
seen-signature maps have bounded exact fields and strings; the v2 ancestry must
match the versioned recipe. `validate_bundle` now traverses the caller tree before
public consumers. `canonical` itself performs depth24/node200000/container4096/
string131072/aggregate8MiB finite traversal before `json.dumps`. Cycles terminate
at the declared shallow depth bound rather than the interpreter recursion limit.
Finite floats support recorded costs; seeds/slots/counts/rules remain exact typed
integers/booleans in their own schemas, with no boolean/integer coercion.

I executed68 small rejection controls under direct CPython3.11.15 with
`-I -S -B -X utf8`, in0.0099608000000444 seconds; maximum individual check
0.0018161999996664235s. All raised bounded ValueError, no RecursionError.
Malformed JSON encoding was made fatal and recorded0 calls. `build_panel`,
`_registered`, `optimizer_deny`, `cpu_receipt`, optimizer oracle and old `_world`
were made fatal where reached: all remained0. Core/scorer modules were not imported.
No new core/oracle/scorer comparison, panel construction or rename control ran.

| Reviewer controls | API rejections |
| --- | ---: |
| Deep1200/cyclic dict/cyclic list/nested/oversized/extra/nonboolean rules through validate_public, model_input and public_position_controls | 21 |
| Extra/cyclic/nested/oversized/boolean/float parameters through root_identity, construct and assess | 18 |
| Generic deep/cyclic/oversized key/string/list/NaN/infinity canonical input | 8 |
| Bundle traversal before nested public/private consumers or serializer | 5 |
| Nested/boolean/oversized seen signatures/census | 4 |
| Nested/cyclic/extra/boolean ancestry shapes | 4 |
| Oversized ID/UTF8 source and boolean query time | 3 |
| Forged operation/root/prior-root/ancestor chain and external optimizer anchor drift | 5 |

Raw control outcome was `TARGETED_BOUNDARY_CONTROLS_PASS`,68 API rejections,
0 malformed serializer calls and0 new execution/comparison work. This independently
replays the original deep-rules/parameters failure across the exposed paths without
repeating the builder's prior7 targeted tests or the coordinator's15 class tests.
It closes the bounded-input criterion on this frozen source; it is not a general
hostile-Python security guarantee or population qualification.

## Retained validation, source anchors and score mapping

The named directory is `outputs/acquisition_second_freeze_root_validation_001/`.
All7 physical path/size/SHA references independently match REPORT, including the
pre-saved PLAN, wrapper, attempt, private bundle, raw core output, CPU receipt and
class-test log. PLAN/wrapper/source pins and validation ID agree; ATTEMPT is
create-once with no retry. The wrapper guards one receipt call,32 core calls and
32 receipt rename controls before each call, retains the bundle before the batch,
and writes raw predictions/costs afterwards. Its actual retained counters are
32/1/32. It deliberately runs only AcquisitionTests, not the prior targeted class
or a full module suite. The log records15 passing tests,0 failures/errors/skips
in3.586s; REPORT interval is3.5864787999998953s. I inspected that execution and
did not rerun it.

| Retained physical artifact | SHA256 |
| --- | --- |
| PLAN.json | 849dbe4c35db7e24f8ff308e02f9111feba529a040e8edd97d56a71ae1f711c0 |
| run_validation.py | 3ae8cd618be3e505abc0f9d263c8a17a9b92c90c7198f9b0a93de04b44fd1047 |
| ATTEMPT.json | 85923cb1d8cff480cd9557f737635e78706408cd44ee7d59aa6880a276b603c3 |
| PRIVATE_BUNDLE.json | c9acf12fb4d9d1ac7d1f6bb2e7f4c1b07a00da9f41ee248188a9e4bae75e52e3 |
| PRIVATE_CORE_OUTPUTS.json | 5aae8078cd4b058bb3785e91e53d0ed6eb507087a77f6a21cdb6e10abb62aa91 |
| CPU_RECEIPT.json | eaada430d9888ad760e22baefe9ce240da83a257e7bf3f6617cb6f71b45f8331 |
| ROOT_TEST_OUTPUT.txt | 502443a430ef40221183bbb34a025531a3c1f352c78dbd0964cf4947d736a7c3 |

All32 scheduled unique IDs occur exactly once in raw predictions, receipt main
and rename rows. Calls are1..32; each raw input hash matches the retained seven-field
public case. Raw prediction error isNone, semantic status/value projection equals
private gold, optional evidence_ids are bounded unique source-event IDs, and
receipt core_answer/error/cost equals its corresponding raw prediction. This is
the unchanged scorer's declared status/value projection plus optional evidence
format; no scorer function was invoked. Every raw main score and rename invariant
isTrue. Read-only association inspection took0.01767180000024382s, with0 new core/
scorer/oracle/panel batches;32 metadata/model-input/root checks were performed.

The first association harness mistakenly required the raw answer to have only
status/value fields; it stopped on the documented optional evidence_ids. Source
inspection of scoring_v2's projection/format confirmed that field is permitted.
The corrected check retained/verified those references and passed. This was a
review harness assumption, not a data defect; no core/scorer/receipt was rerun.

Five independently recomputed bank digests equal bundle manifest and REPORT:

| Component | SHA256 |
| --- | --- |
| public | f8d60b951de2825b3e103ca5bb6eb7be25d299b304305eb5f63b3f84e80f3b41 |
| gold | d27a8b97d4d746a67645e58e0b5ecf6e467aff2b5a2b3db5a9df6eb8901565fa |
| latent | 9f769b7383488112b67c74535e0435652feb48a7cfc11468e5d26c3b5c58ee91 |
| lineage | 25702f6b392c53bd9c7c0acf291e542774e32e695503b49c6aa9e4bc278ea02e |
| assessments | c6218e9265bb9335d1f2fc44e821227c62a2c37fe68bf1223addf4260b4cd2c2 |

The32 roots/parameters/case IDs and presentation order match the frozen contract
and production registration. Each root is v2, with exact prior-v1 and current-v2
ancestry and `RECIPE_REVISION_OF_UNQUALIFIED_PANEL`; all32 links were verified
through production primitive identity helpers, without constructing another panel.
V2 is related exposed/revised data, not a fresh independent experiment. Source
inspection and retained class controls preserve exact external `_registered`
recipe/bank comparison; coherently resealed caller data cannot redefine the bank.

Accepted optimizer source837bd/contract37f/acceptanceceece, core18d3/scorer4233 and
all reference pins remain exact. Optimizer draw101/order211 and its public03de/
goldee4 bank are not modified. Public `model_input` was checked for all32 rows and
contains exactly source/query/rules. Scorer-only private access remains separate;
no optimizer target API is added. Existing optimizer target validation rejects
the different panel bundle; no OS secrecy claim is made.

Full/cone/kinship algorithm, finite closure and shared4096-state budget remain
unchanged; new signatures require exact bounded code/hash/type association.
The retained maximum is3010 UTF8 source bytes,57 latent records and1025 canonical
states, below8192/256/4096. Bundle544513 and receipt71089 bytes fit8MiB. No root is
pruned/replaced; all32 stay scheduled. Unknown ancestry/canonical exhaustion cannot
certify distinctness. This is declared-source/finite-closure association, not
proof against every undisclosed edit or statistical/pretraining independence.

## Costs, scientific boundary and disposition

Raw rows independently sum192 operations,1060 events,56153 source bytes scanned,
110354 verification bytes and0.047000000000025466s core wall time; peak serialized
state13803 bytes. Core time is nested within the3.586479s validation interval,
not added again. Coordinator-reported whole command4.15628s is separate from
that nested raw timer; it is not reconstructed from per-row costs.

Cumulative history remains161 core calls (including the prior diagnostic),
160 rename controls and321 combined invocations, with160 main comparisons.
This report adds0 to those execution counts. Earlier builder raw rows/hashes and
auxiliary constructor/oracle/canonical aggregate costs remain unavailable; old
budget overrun/terminal-only results are not erased by one current passing batch.
No native budget, owner limit or source-review cap is reset.

The retained receipt reports8 earlier selections,two per family, and single-position
controls terminal24/32,first13/32,penultimate12/32,fixed-position1 11/32. Those
facts are consistent with the frozen recipe, but they do not qualify the
population: Root's separate SCI2 finding shows the combined public terminal/first/
penultimate control gets32/32 while skipping84 candidates. SCI2 independently verified
28/32 v1-to-v2 status changes. This operations review does not rerun that control,
claim depth=N, accept matched-template acquisition, amend H01 or close LA-I1-002.
The raw source feasibility result and scientific population rejection must retain
separate statuses.

This is the complete scoped OPS2 disposition: AP-OPS-1-001 closed, source boundary
and retained association supported, no remaining consequential operations finding
reproduced. Any third candidate requires the coordinator's coherent remaining-round
scope and preservation; this report authorizes no new repair, batch or launch.
Population acceptance remains withheld by the independent scientific disposition.
No acquisition/retention/enrollment, token fit, model/training permission, owner
exception, quarantine reopening, new CONFIRM/CHECK/FINAL or optimizer exposure
follows from this review.
