# Acquisition-panel operational review 1

2026-10-02; fresh non-builder reviewer `/root/acq_ops`. **First freeze is not
accepted: one consequential bounded-input finding remains.** This is round 1 of
the acquisition instrument's maximum three review/repair rounds. It grants no
population, acquisition, enrollment, tokenizer, optimizer, native or owner-exception
authority. Shared model, workspace and supplied semantic assumptions limit
independence. I wrote only this report; I did not change source or registration.

## Frozen evidence and permitted work

Physical SHA256 values matched before and after scrutiny:

| Artifact | SHA256 |
| --- | --- |
| `acquisition_data.py` | `1d20d33cc67ee309c8aef0de466d3be69c544d4ba5cc8510580561a425553d54` |
| `tests/test_acquisition_data.py` | `f7188468df0225878a0df9742c3504c0d2406e316e936f2983f93c3e627a9ac5` |
| `contracts/ACQUISITION_PANEL_V1.json` | `d80358f369e18c2d9ebc6ed8116132a3cbf67b1078957410bef9e6a0fbb6ffc5` |
| Accepted `learning_data.py` | `837bd152ba5aa9d1fc44a5991909cf4475394eaa42af739cca63ac3422ed3795` |
| Accepted `contracts/LEARNING_DATA_V1.json` | `37f316972afc6b1a43f9a1ef8f797b0452fb0de5706724c0fd49f8bf0470015a` |
| Accepted `evidence/LEARNING_DATA_ACCEPTANCE_001.json` | `ceece000901c57917b2a7ad2eab87365e542f4a5c28e6e5c5b8b4155f6874c62` |

I read AGENTS.md, the three acquisition scope reports, all three frozen candidate
files and the accepted optimizer's relevant validation/oracle/anchor code. I used
targeted in-memory constructors, canonicalizers and validators. I did not call
`cpu_receipt`, core or scorer, run the full suite, change any accepted optimizer
bytes/receipts, construct a replacement panel or access a protected bank. Expected
recipe/optimizer replays inside validators are auxiliary construction work, not
another oracle/core comparison batch. No model, tokenizer, framework, native,
host-database, scheduler or paid-service operation occurred.

The coordinator's separately scoped run is
`outputs/acquisition_first_freeze_root_validation_001/REPORT.json`, physical hash
`e0df2615850c2b186e9682d7a56678c7005c62d9b002ef7e5efaec84b2996113`.
I verified all four retained reference hashes, the private bundle and receipt
membership and their digest associations without rerunning comparisons:

| Retained reference | SHA256 |
| --- | --- |
| `PLAN.json` | `79d4e61df9e64da74e8da3457a08ccd97d06209cc8a10820fa519a6692ab301b` |
| `PRIVATE_BUNDLE.json` | `bcac4e876d2df9ee69801474bc05368d1268e20bcd739c7d2a15f93f3c06c6b1` |
| `CPU_RECEIPT.json` | `fdaa45a22db07aefd04e31276a9579dc03a37ae9ca572407d41b88801889bd56` |
| `ROOT_TEST_OUTPUT.txt` | `39511a894af6591bec51683aff06781f8e039f23792b41896d50d2c05d35b383` |

The new run records 15 passing tests, 32 passing main comparisons, 32 passing
rename controls and 2.450573799999802 seconds wall time. Its raw receipt has
exactly the scheduled 32 row IDs in both arrays and all required pass booleans;
this review inspected those retained records, not independently replayed core
outputs. The new bundle's five digests match both its manifest and the report:
public `ec0b87a81dee0c0e62335663b7177c2ebca69df28e3c7198b74dccee796b06cd`,
gold `efddce3092672ff2f65cd95ab9d435f02024d6865fb5083917837985a305aa09`,
latent `ba2bd393142a2fdefadd46607ab783ee17109bc82319279a02725544e6fff79d`,
lineage `b45b4c38bea42fb5963c1c0b253da63ad2b56eb45c7ad49626ccd2f14783f451`,
assessments `16231808536b55215249ef0d5801810f74ed3201ffa6040b2ae96882a54854b4`.
These are this retained run's evidence, not reconstructed historical builder hashes.

Prior builder observations remain summary-only: 96 main and 96 rename invocations
plus one diagnostic, totaling 97 core calls and 193 combined invocations. With the
new coordinator run, recorded cumulative work is 128 main, 128 rename and one
diagnostic: **129 core calls and 257 combined invocations**. Historical raw rows,
historical bank hashes and aggregate row-level costs remain unavailable. Neither
one passing 32+32 batch nor this report erases the earlier work or certifies that
the original one-batch construction budget was observed cumulatively. Root must
retain that accounting/adjudication explicitly.

## Complete consequential finding set

### AP-OPS-1-001 — Reject caller shapes before JSON serialization

Priority P2; blocks source/data acceptance under the explicit finite-field,
depth/receipt bounds. `validate_bundle` calls `validate_public` at line 423 before
its bounded traversal at lines 425–438. `validate_public` serializes arbitrary
caller-supplied `rules` with `canonical` at line 315 without first requiring the
three exact rule fields and their scalar types. A deeply nested rules object
therefore reaches Python's JSON recursion limit before the declared depth-24
guard. `root_identity` at line 243, `construct` at lines 258–259 and `assess` at
lines 355–356 similarly serialize arbitrary nested parameter values before exact
parameter-shape rejection.

Reproduced on the pinned source with a small in-memory object, no core call:

```python
deep = {}
cursor = deep
for _ in range(1200):
    cursor["x"] = {}
    cursor = cursor["x"]

bad = copy.deepcopy(retained_bundle)
bad["cases"][0]["rules"] = deep
panel.validate_bundle(bad)
# RecursionError: maximum recursion depth exceeded while encoding a JSON object

ancestry = copy.deepcopy(retained_bundle["private"]["lineage"][0]["ancestry"])
ancestry["parameters"]["extra"] = deep
world = retained_bundle["private"]["latent"][retained_bundle["cases"][0]["id"]]
panel.assess(world, ancestry)
# The same RecursionError, before registered parameter rejection.
```

Direct `validate_public`, `root_identity` and `construct` controls reproduced the
same failure. Invalid data is not admitted, but the caller-controlled structure
can consume unbounded serialization work or fail outside the intended bounded
rejection path before receipt bounds run. Testing only a large private string
does not exercise this public-rules/parameter path.

Repair criterion: validate exact finite scalar rules and registered parameter
keys/types before any canonical encoding, or apply a bounded traversal first on
every exposed entry path. The fixture above and oversized/cyclic/nested rules or
extra parameter values must reject through a bounded ValueError/Unavailable path
without encoding them first. Keep strict integer seeds/slots and boolean rules,
the fixed recipe replay, all authority flags and optimizer anchors. Retain this
failed freeze. No source repair or extra full-bank comparison is authorized by
this report itself.

## Operational criteria otherwise supported in this freeze

The constructor builds typed lists directly from fixed family/slot factors; it
does not receive or edit optimizer trajectories. Root identity derives from the
registered parameter tuple before evidence, rendering or transformations. Exact
world equality with the direct registered recipe prevents caller-declared new
roots, edited worlds or unknown provenance from becoming admitted parents. This
is cooperative source provenance, not proof against arbitrary undisclosed edits
under shared privileges.

The exact directed typed graph algorithm exhausts unresolved individualization
branches within one 4096-state budget shared across full/cone/kinship. Discrete
leaves retain original node colors and directed edge roles; equality uses complete
canonical strings, not only hashes or first occurrence. Full and cone preserve
typed identity/alias/target relations, binding/fact/query rank order and ties.
Condition declaration dates are ignored by the supplied semantics; alias and
cancel active/future boundaries survive where they affect the query. The cone
retains all potential confirmed/base query bindings and their applicable evidence,
not only the final oracle label. Kinship retains binding topology/order and
referenced predicate-history counts while removing the declared state/date/modality
axes. Targeted rename/retime/list-order control preserved all three codes;
fact-state, fact-cutoff, transfer-modality and cancel-query-boundary changes changed
full codes while preserving kinship. Inspection supports the declared finite
closure overapproximation; it does not establish unrestricted ancestry or arbitrary
logical equivalence. Unknown or exhausted canonicalization cannot certify novelty.

The deny census has 32 accepted optimizer roots, 32 complete signature records and
zero unavailable roots. Regenerated accepted optimizer public/gold/latent/lineage
digests match the immutable acceptance record, including public `03de77bd...` and
gold `ee4aa273...`. Every panel candidate compares full/cone/kinship codes against
optimizer and prior panel candidates. Within the fixed construction no rejected
candidate is replaced or pruned; all 32 remain scheduled and the actual counts are
eight per state and two per state/family. The retained bundle's maxima are 3010
UTF8 source bytes, 57 latent records and 1025 canonical states. Its physical size
is 537119 bytes; the receipt is 13759 bytes, below 8 MiB.

The frozen recipe/version/draw307/order401 and exact 32-member schedule are outside
the caller's copied bundle. Retained registration, scheduled IDs and ordered IDs
match the frozen contract. Recomputing all caller self-hashes after gold, root,
ancestry or order edits did not pass `validate_bundle`. All seven authority-field
flips also rejected: LA-I1 closure, H01 amendment, optimizer exposure, optimizer
targets, native eligibility, training eligibility and new confirmation banks.
An injected public `gold` field rejected at the public boundary. The bounded-input
finding above remains the exception to otherwise effective validation.

`model_input` exposes exactly source/query/rules; coordinator IDs, root/family/split
metadata, private gold, latent state and ancestry remain outside that interface.
The seven-field public Case envelope used by CPU core is separately documented.
The panel provides no training-target API; the accepted optimizer API rejects its
different private/registration shape. Wrong scoring intent rejects, and accepted
scorer intent returns only exact private gold association, ordered membership and
bank hash with optimizer exposure false. This is interface separation rather than
OS secrecy.

## Remaining claim limits and handoff

The proposed 2–9 guarded-candidate/history population explicitly expands the
optimizer's one-primary-transition template and depth/length. Shared primitives
and graph disjointness do not establish matched difficulty or validate the intended
scientific acquisition contrast. Static potential-cone relevance, actual shortcut
scores and the population must receive the separate scientific adjudication.
The retained length-fitted shortcut is 18/32, and shortcut absence is not claimed.
Original CRS-LA-H01 is neither tested nor amended; LA-I1-002 remains open. No
acquisition result, actual token fit, retention enrollment, optimizer exposure,
new CHECK/FINAL/CONFIRM bank or native/training eligibility follows from this
review. Runtime/reader/probe quarantines, the pending owner exception and their
maximum-three-round limits remain unchanged. Resolve AP-OPS-1-001 within this
instrument's remaining review/repair rounds and preserve the source pins, negative
history, raw new-run artifacts and cumulative-accounting limitations.
