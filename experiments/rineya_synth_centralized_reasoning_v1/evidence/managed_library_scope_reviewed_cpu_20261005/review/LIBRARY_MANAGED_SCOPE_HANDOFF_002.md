# Frozen V2 managed-scope source milestone

Only staged `prefix_library_policy.py` and new `test_library_managed_scope.py` changed. No root tracked source, runtime/model bytes, native/OS/GPU API, process action or launch permission changed.

Final source SHA256:

- Helper: `bd57c15837493de60e97081b2560110221037d500556eed3ef8ba2d3cb58e401`.
- New tests: `b2dafaa2b12a16d6a9947cb27d963172fac2d929420dfb638977139803d95370`.
- Unchanged legacy51 tests: `eb6bac407f647fc9cf5b529dde55a0d9d5fd01613659935116dcb835ccae669a`.

Final validation: 51 unchanged V1 controls plus 15 new V2 controls pass in 4.359 seconds, zero skips, unchanged helper/test/loader readset, no forbidden imports or actual host/runtime/model/cache byte attempts. Actual output/receipt: `LIBRARY_MANAGED_SCOPE_CPU_1791247209612272400.txt/json`, receipt body `8212fec5841666e1386e431a88795a520c8d52d9f50aa14b63a6751457360855`. The first66 receipt `...1791247171953507300` retains one test regex mismatch: source drift correctly preserved the primary `actual selected library/source bytes changed` error instead of the test's expected cleanup message. Exact first helper/tests remain in `MANAGED_SCOPE_FIRST_VALIDATION_FAILURE_001`; only the expected error and legacy module alias were corrected. Default test discovery now sees just15 new methods, not an imported legacy TestCase.

V1 helper `c88c5c51...eb05` and unchanged legacy tests remain in `PRE_MANAGED_SCOPE_V1_001` and Root's committed reviewed evidence. The source-cited recommendation precedes edits at `LIBRARY_MANAGED_SCOPE_RECOMMENDATION_001.md` (`d3c463cd...a662b`); Root's prospective amendment is `LIBRARY_OBSERVATION_AMENDMENT_002.md` (`72e2d1de...b855b`). New V2 does not reinterpret or qualify old V1 receipts.

The V2 policy uses the same exact8 fields and literal `FINITE_WINDOWS_DLL_BOUNDARY_V2`. Managed setup/cleanup fields are typed; foreign registrations remain literal `UNOBSERVED`. V1 public native guard/direct provider refuse before OS imports. The new provider requires explicit fresh identity, selected provider/preloaded source and module-event provenance; no universal registration trace is inferred. Its construction still loads bootstrap kernel/psapi before event scope start, so those identities require the complete qualified initial baseline. No process-creation event coverage is claimed.

Event/snapshot schemas, append-only complete trace, exact initial/transition identities, prospective hashes, poisoning and final lifetime conservation remain intact. Observer `close()` must stop, losslessly drain/seal and leave a finalized `events()` log readable after close; a missing final drain prevents success. V2 cleanup publishes only managed-cookie accounting. The source terminal result adds policy kind/unknown foreign state while retaining false native qualification/authority.

The current contract's legacy `qualified_complete_registered_DLL_search_rosters` label means exact static selected roster/file walks. It does not mean complete observed foreign AddDllDirectory registrations. V2 observation fields explicitly distinguish that unknown state. The static schema and original source/runtime65, resource/numerical/ownership gates remain Root-owned; no C/W/P edits were made here.

Current qualified native observer implementation, fresh bootstrap and ownership proof, actual runtime/dependency bytes, original/final terminal event/scope provenance export, API/cost dose and trusted owned-entry authority remain pending. No Python DLL-notification callback, OS sandbox or protection against cloned trusted Python namespace/closure state is claimed. This is partial source/CPU mechanics, not native factory readiness, GPU fit, M0 membership, M1/M2 training, scoring or transfer evidence.
