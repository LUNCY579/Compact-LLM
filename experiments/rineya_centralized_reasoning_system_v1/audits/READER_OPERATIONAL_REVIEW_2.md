# CRS reader operations review 2

Date: 2026-10-02. Reviewer `/root/data_audit`; builder `/root/runtime_audit`.
**Disposition: REPAIR_REQUIRED_NOT_ADMITTED.** This reviews the distinct reader's first repair, not another native D2 repair. Native whole-path quarantine remains closed. No model, framework, tokenizer, GPU API, actual executable launch, canonical host-state database, private gold or protected bank was used. Shared model ancestry/workspace is disclosed; these controls are engineering evidence, not neural results. The reviewer wrote only this report for this review and did not change reader builder files or approve the reviewer's learning/codec implementations.

## Exact frozen subjects

| File | SHA256 |
| --- | --- |
| reader.py | `47c84933f1c539165704b0130efecb77c8b4b1d296f0d4baceb4e693c2175504` |
| reader_execution.py | `0668d35736ab64f7828c453c45c4416fd009dd49b7cb6f40b21318fe3a107ec0` |
| tests/test_reader.py | `7a28d737c843a85501ed4cd68939d6d318d06419a0a51002cad8716c6ef0277e` |
| contracts/READER_QUALIFICATION_V1.json | `6262f27299befe5e6d7da24c6b999565211a836534f31de74a1ed1c8cf0e73d8` |
| READER_PROTOCOL_V1.md | `75d1f24bf713509466dea8eb882ca26956c2a3064ed752e4aecfbc1c67230792` |
| prior operations review | `828cf292520489652460366e70767b94edfec7fd85f9daea28deb1013de4597e` |
| unchanged native.py | `8453128981280019702dff80ebd5083031f7f2dcd8a7c6c9094587d9a2e377a3` |
| unchanged operations.py | `ba27ec8a31437459b956a5642e6ece4838f0584602db2c93ff7bc61437806e30` |

The complete policy, session, worker, terminal validator, artifact validator, supervisor, and publication code was read directly. Source-pinned imported Job/stream-loader/accounting primitives retain their preceding review scope. The contract's generic first_review_sha256 points to the scientific first review (`f46ad9df...`), not this reviewer’s prior operations report; that association was checked rather than assumed.

## Verification and previous finding closure

The registered interpreter ran all20 supplied stdlib tests successfully in13.439 seconds, with no errors/skips or framework imports. Positive completion/artifact fixtures are explicitly synthetic CPU schemas, not native receipts. Independent controls exercised the actual production validators and wrapper with modified fixture inputs. A separate mocked Popen-failure control used real OperationsStore in a temporary CRS outputs SQLite, with the canonical-path/admission boundaries substituted only within that fixture; no canonical database or worker was touched.

R-OPS-001 is substantially repaired: serving requires the complete named artifact bundle, exact expected contract binding, replayed worker/journal validation, loader tensor names, runtime/tokenizer identities and a distinct post-run evidence review. Status-only bundles reject. R-OPS-002 is substantially repaired: ordered measured forwards bind registered stage/prefix/keep/mask/position geometry, attempts/call IDs, blocks, rollouts and a single durable control manifest; coherent bad geometry, missing controls and historical drift reject. These are narrow schema/provenance closures, not proof that CPU fixtures executed a model.

R-OPS-005 is closed as an admission dependency: the actual unchanged contract yields preflight BLOCKED with empty observations and an inherited-supervision QUARANTINED reason. This happens before model-asset hashes or host telemetry. Worker and serving/publication paths independently retain the quarantine check. No resolver outcome alone may change that disposition. Public CLI serve remains separately closed.

R-OPS-003 rejects blank, bare and canonical slash builder identities, but remains incomplete for backslashes (below). R-OPS-004 separates source review from post-run evidence approval, so source approval no longer publishes serving PASS. Its late path check and publication recovery requirements remain unresolved (below).

An independent injected Popen response loss retained OUTCOME_UNKNOWN with actual_seconds=None and600 charged seconds. It attempted only the mocked Popen, launched no process, and never synthesized a no-dispatch/zero-cost proof. Exact Job/query closure, current resource/fencing primitives and conservative unknown-state handling are retained. Synthetic receipts were cleaned with their own temporary paths/stores closed. No reported fixture cost was added to the canonical native budget.

## R2-OPS-001 — P1: source-identical evidence is not coupled to one reserved run

Evidence: reader_execution.validate_native_artifacts (616–704) compares source/model binding and GO-to-containment fields, but its supervisor proof has no required job key, owner/fence, recorded worker identity, request hash or run identifier. The generated SUPERVISOR_RESULT at863–870 also omits those fields. Multiple runs can share exactly the same source binding. The purported complete before-GO tree requires at least2 members but does not require unique PID/time tuples or inclusion of worker_tree_pid/launcher.

Independent executed counterexamples: start with the supplied valid CPU artifact schema and add supervisor `{job_key:'OTHER_RUN', owner:'OTHER_OWNER', fence:999, worker_identity:{pid:999,creation_time:999}}`, unrelated to GO. validate_native_artifacts returnsTrue. Replace before_go_members with two copies of the runtime target PID11, leaving launcher/root PID10 absent: it still returnsTrue. These controls show missing association/completeness checks; they do not authenticate or claim a real native run.

Related child-entry gap from direct source inspection: worker accepts the caller's args.host_db without verifying the canonical host_state_path. The launched worker command and GO omit job_key/epoch/request hash. Worker’s event function verifies only a live owner/fence lease; it never confirms that this exact worker belongs to the admitted RUNNING reader job with the600-second reservation. Ancestry, parent `--qualify` command, independent source review and the exact Job handle are useful, but a lease or source commit alone does not prove this scheduled job. The present quarantine prevents actual dispatch; this must close before future admission is reopened.

Impact: artifacts from different source-identical runs can be associated without proof that the successful supervisor fenced/contained/accounted the measured reader worker. An accidentally mixed GO/lease/database can reach the future child path without coupling imports/model work to the correct one-host-wide reservation. Duplicate target entries can falsely describe a complete owned tree. This is cooperative protocol integrity, not an OS authentication claim against coherent same-user forgery.

Verification: give dispatch, GO, worker, journal, terminal receipt and retained outcome proof one immutable run/job/request/source identity. Require canonical database, correct admitted RUNNING row/reservation/owner/fence/epoch and recorded launcher/runtime PID+creation identities before framework work. Supervisor success must bind the same recorded job/worker and full measured outcome proof. Require unique complete before-resume/before-GO membership with the launcher and exact runtime target, consistent command/transfer/ctime/ancestry. Reject another job's valid lease, noncanonical DB, swapped source-identical supervisor, wrong request/worker/fence, duplicate/missing launcher and direct child entry. Keep unknown costs held and no duplicate launch on every rejection.

## R2-OPS-002 — P2: publication prerequisites and recovery are still late/incomplete

Evidence: supervise invokes _supervise_reader_worker at914, then checks the immutable review path under results at920 and writes READER_EVIDENCE_PENDING_REVIEW at932. The inner path has already reconciled SUCCEEDED/cost and released its model lease at895–899. There is no public operation to reconstruct a missing pending publication from existing verified raw artifacts; rerunning supervise uses the terminal job key. publish_after_evidence_review at969 unconditionally calls create-only write_new without accepting an existing byte-identical published receipt.

Independent executed reproducer: pass a review path outside results and mock only the inner worker adapter’s return as0 plus result reads. The wrapper records `worker_adapter_returned_success` before raising `immutable independent review must be retained under results`. No worker/model was run. The create-only lost-response/retry failure is directly implied by native.write_new: an existing target causes os.link/FileExistsError, with no equality/recovery path in this wrapper.

Impact: invalid artifact placement or interrupted disk/publication can occur after expensive work and ledger success, leaving a missing/unusable pending serving bundle. Source-only approval correctly cannot open serving anymore; nevertheless publication failures must be recoverable from retained evidence without a new model attempt, and repeated identical publication should not be treated as a new failure.

Verification: validate all required path/reference/publication prerequisites before reservation/dispatch. Distinguish successful contained worker evidence from complete pending publication and independent serving approval. Provide bounded idempotent recovery/publication from existing verified immutable artifacts; accept only byte-identical completed receipts on retry and reject conflicting bytes. Inject outside review paths, lost write/response, partial/missing pending manifest, concurrent identical publication and conflicting receipt. No recovery may redispatch a model, reset costs, or use source review as evidence approval.

## R2-OPS-003 — P2: builder identity normalization remains incomplete

Evidence: review_matches (74–79) and validate_native_evidence_review normalize whitespace and forward slashes but not backslashes. They require only that the final forward-slash component differs from runtime_audit.

Independent executed reproducer: reviewer `runtime_audit` and `/root/runtime_audit` reject, while `\\root\\runtime_audit` is accepted by the actual source review predicate with identical status/binding. The same normalization shape exists in post-run evidence review. This is an ordinary recognizable alias of the prohibited builder, not independent reviewer evidence.

Impact: the provenance rule can label a builder identity as independent. A name string does not itself authenticate a reviewer; predictable canonical aliases must nevertheless reject.

Verification: use one explicit normalized reviewer identity policy for source and post-run evidence; reject missing/blank/bare/canonical/backslash/trailing-separator builder variants, and bind identified independent artifacts to their exact source or native evidence role. This remains logical provenance validation, not same-user identity authentication.

## Scope, schedule and remaining launch gates

The registered upper schedule remains278 forwards,146316 input positions and590 logical construct/load/copy/forward calls, with forced65 non-emitting prefixes512..576 and134 maximum owned+stock emissions. Natural cap64 from a512-token prompt predicts at575 and does not generate a65th token at576. Original online likelihood/raw-logit anchors and failed bulk identity are retained under unchanged0.03125/0.01 bounds. The worker/session captures full mask/rotary/hidden evidence and frozen base hashes; actual reference generation and tokenizer behavior remain unexecuted in this review.

The session’s278-call qualification budget is not an admitted future DEV-serving budget. New invocation/worker-time/cumulative token accounting must be separately registered; the CPU codec/draft planned90-span screen is neither tokenizer-fit evidence nor reader qualification. Binding strings do not by themselves attest an arbitrary supplied model object. The inherited host/runtime quarantine, fresh RAM/GPU/process evidence, canonical lease and independent post-run evidence review remain separate conditions.

This report approves no model dispatch, serving, training, registry expansion, app termination or native source edit. Repair the three findings on new frozen reader bytes and close targeted controls; keep original reader/native histories and failures. A clean exact-commit attestation plus independently qualified inherited supervision and fresh unchanged admission are still required for any later bounded reader qualification.

Machine-readable review equivalent:

```json
{"review_id":"CRS-READER-OPS-REVIEW-002","reviewer":"/root/data_audit","builder":"/root/runtime_audit","status":"REPAIR_REQUIRED_NOT_ADMITTED","reader_source_sha256":"47c84933f1c539165704b0130efecb77c8b4b1d296f0d4baceb4e693c2175504","execution_source_sha256":"0668d35736ab64f7828c453c45c4416fd009dd49b7cb6f40b21318fe3a107ec0","contract_sha256":"6262f27299befe5e6d7da24c6b999565211a836534f31de74a1ed1c8cf0e73d8","supplied_tests_passed":20,"findings":["R2-OPS-001","R2-OPS-002","R2-OPS-003"],"quarantine_dependency_resolved":false,"native_model_dispatch_approved":false,"model_calls_by_reviewer":0,"GPU_calls_by_reviewer":0,"canonical_host_DB_accessed":false,"own_codec_or_learning_delta_approved":false}
```
