# Reader operations review 3: final candidate quarantined

Date: 2026-10-02. Reviewer `/root/data_audit`; builder `/root/runtime_audit`.
**QUARANTINED_AFTER_FINAL_THIRD_REVIEW — NOT APPROVED.** The frozen reader has
a reproduced incompatibility between its actual artifact writer and validator.
Root adjudicated that the three-round cap applies: no fourth repair, no source
integration, reader/model dispatch or inferred qualification follows this review.
The scientific third-round source acceptance was conditional on operations
closure; that condition is unsatisfied. Preserve its original report and the
separate Root adjudication rather than rewriting history.

## Exact subject

| File | SHA256 |
| --- | --- |
| reader.py | `b420e07583bb81ad3ebe7222fdaa254810881f1013a9a336527d9481f9af7a40` |
| reader_execution.py | `0dcd7dc639f3d8b816288b1417750772b6e704261af47d4dc7eba298383f5e9e` |
| tests/test_reader.py | `06c13d9e59c37806e60bcecc854c46e3b837d8d09035d559f1cf7c3244b4eaba` |
| contracts/READER_QUALIFICATION_V1.json | `b50d61609f8bb933045abac1b769dc7979c3b8f6f2163eb256c2fe3e77376bb0` |
| READER_PROTOCOL_V1.md | `cc1636113d6361fe5c7e78057656fa514183aa4fc135ff8088399d955dfa52f8` |

Actual source and the relevant worker/run/job/publication/validator/test paths
were inspected directly. No builder file, native source, operations source,
runtime registry, weight or bank was changed. No reader/model/framework/tokenizer,
GPU API, canonical host database or actual reader worker was executed. Shared
model/workspace ancestry is disclosed. The builder reports 27 CPU tests passed
in40.150 seconds; this reviewer does not claim an independent full27 run.
After the decisive actual-writer control, no broader tests/features were needed.

## Prior operations repairs inspected

The new run identity, registered command/request, canonical child database,
RUNNING reservation check and event context substantially address R2-OPS-001.
All named artifacts now bind one launcher/runtime/run/fence; before-GO members
must be unique and include launcher plus exact runtime, with the suspended
before-resume witness. Source-identical foreign supervisor data can no longer
simply substitute for this run’s recorded job/outcome proof.

Predispatch review/output paths and a read-only recovery adapter, immutable
publication, identical retry/race behavior and no-worker recovery substantially
address R2-OPS-002 in source. Shared slash/backslash/case normalization addresses
the ordinary builder aliases from R2-OPS-003. Existing inherited-supervision
quarantine still denies actual admission before host/model checks. These source
improvements do not establish working publication: the production byte mismatch
below prevents that closure and bars approval of the whole candidate.

## R3-OPS-001 — P1: production supervisor bytes cannot satisfy terminal artifact validation

**Evidence:** reader_execution._supervise_reader_worker writes
SUPERVISOR_RESULT.json via write_new at1099. This is the unchanged native.write_new
alias, which serializes `canonical(value) + b'\n'`. Reconciliation records
`sha(output/'SUPERVISOR_RESULT.json')` at1124: the hash of those actual file bytes.
validate_native_artifacts at875–877 instead reconstructs
`supervisor_receipt_sha256: digest(supervisor)`, the hash of the parsed object’s
canonical serialization **without** the writer’s newline. Thus correct producer
evidence differs from the validator’s expected terminal result.

**Executed independent reproducer:** use the declared synthetic CPU artifact
schema only. Its ordinary fixture baseline returnsTrue from the production
artifact validator. Write the same supervisor object through the **actual**
reader_execution.write_new; compute the resulting file SHA and put that correct
producer hash into the synthetic terminal evidence. The actual production
validator rejects with:

`durable terminal result must bind exact supervisor and immutable outcome receipt`

| Byte representation in the preserved CPU control | SHA256 |
| --- | --- |
| actual production writer bytes, trailing newline | `7a500d3b9deb5855333979fb9d688519083c0381301bea8e78f7527358733c07` |
| parsed-object canonical bytes, no trailing newline | `0c73c4c81d8e04913e2a989f2a009657547a6f468a438773da92cd75959b710c` |

These hashes refer to a clearly synthetic fixture object, not a measured native
supervisor result. The test fixture’s recovery construction writes supervisor
objects directly with `reader.canonical(...)` and no newline (test_reader429),
so it does not exercise the actual producer representation and masks the mismatch.

**Impact:** a valid real successful worker/supervisor can never pass this new
artifact validation/recovery/publication path as frozen. It could consume the
entire qualified model budget and retain a successful component ledger, yet
remain unable to publish full reader qualification. Rewriting a historical
receipt or terminal evidence to fit the validator would destroy the original
byte association and is not an acceptable workaround. Current quarantine
prevents such a model launch; no actual reader/model run was performed here.

**Verification criterion for any separately authorized resolution:** retain
the frozen candidate and raw evidence. Agree one explicit actual-byte protocol
for each artifact; bind/verify the immutable file bytes which the real producer
writes, including newline/encoding, rather than guessing their representation
from a parsed object. Exercise the actual writer, real temporary OperationsStore
reconciliation and read-only recovery consumer end-to-end under pinned APIs,
without models. Reject byte drift, conflicting/missing/partial publication and
wrong-run evidence; identical recovery must preserve row/cost/event/receipt
identity and never acquire/reserve/dispatch. Do not lower hash, numerical,
source, memory, GPU, lease or qualification checks. This report authorizes no
patch or fourth review round.

## Preserved create-only CPU evidence

Under the CRS ignored outputs directory:

`outputs/reader_final3_writer_binding_cpu_v1/WRITER_BINDING_COUNTEREXAMPLE.json`

SHA256:
`0ef349a131757aae5d140274e28088f78c55a517b2455765be19141b1e72b808`

The same directory retains the actual production writer bytes and the contrasting
object serialization as separately named CPU_SYNTHETIC files. They were created
only if absent and existing bytes were verified identical. The counterexample
records exact code/contract pins, positive schema baseline, the actual rejection
and zero actual-reader/model/framework/host-DB calls. Its location is under
outputs, outside the eligible results namespace. It is not native qualification
evidence and must not be consumed as such.

## Adjudication and restart boundary

Root explicitly rejected relabeling another fix as a same-round closure after
this final frozen candidate. Keep the reader candidate quarantined, the original
scientific acceptance/operations rejection and source/history intact. A separate
bounded CPU byte-binding proposal may be reviewed as a proposal; integration or
an extra review round requires an explicit owner exception/authorization. No
new namespace, receipt, copied code or CPU pass silently opens inference.

The separate identity-probe source/commit attestation did not depend on this
reader path. Subsequently Root reported an actual probe failure from the pinned
Windows psutil.disk_usage(Path) API before framework work, with owned-tree exit
and0.062 worker seconds; that probe candidate is separately quarantined at its
own cap. This report has not independently inspected that run’s complete raw
evidence and grants no probe retry. A typed-API proposal likewise cannot bypass
its owner exception. Neither failure retires CRS-001 or discards its discoveries.

Machine-readable review equivalent:

```json
{"review_id":"CRS-READER-OPS-REVIEW-003","reviewer":"/root/data_audit","builder":"/root/runtime_audit","status":"QUARANTINED_AFTER_FINAL_THIRD_REVIEW","reader_source_sha256":"b420e07583bb81ad3ebe7222fdaa254810881f1013a9a336527d9481f9af7a40","execution_source_sha256":"0dcd7dc639f3d8b816288b1417750772b6e704261af47d4dc7eba298383f5e9e","contract_sha256":"b50d61609f8bb933045abac1b769dc7979c3b8f6f2163eb256c2fe3e77376bb0","findings":["R3-OPS-001"],"actual_writer_control_reproduced":true,"independent_full27_suite_run_claimed":false,"reader_or_model_runs":0,"canonical_host_DB_accessed":false,"source_approved":false,"reader_qualified":false,"fourth_round_authorized":false,"resolution_integration_requires_explicit_owner_exception":true}
```
