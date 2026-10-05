# One-prefix input construction: reviewed design, native scope absent

The [source design](evidence/prefix_input_diagnostic_design_20261005/INPUT_CONSTRUCTION_DIAGNOSTIC.md)
isolates the reporting branch identified in failed batch05: one CUDA int64 IDs
tensor and one mask of shape[1,422], followed by the exact input/mask metadata
and host-value checks. The input is414 original prompt IDs plus eight preserved
generated IDs. It is exposed diagnostic history and cannot enter M0, training,
grading or confirmation.

The [independent design review](evidence/prefix_input_diagnostic_design_20261005/prefix_input_design_review_1791223620522698300.json)
verifies exact input hashes, integer mask, two3376-byte logical tensors and6752
logical bytes total. CUDA context, allocator and Python overhead still need
measurement under unchanged resource guards. D001 explicitly forbids tensor
allocation and grants no permission to this separate proposed scope.

The [prospective dose](evidence/prefix_input_diagnostic_design_20261005/PROPOSED_DOSE.json)
lists35 success operations including four terminal observations, eight guard
sites, at most14 guard queries and one optional bounded failure-frame capture.
This derives58 phases/116 markers within a proposed128-marker budget. All actual
operation, publication, failure and cleanup branches must be reconstructed from
implemented source before admission; these are design bounds, not native timing
or fit measurements.

Implementation must preserve the original short-circuit expressions exactly:
the mask check compares `tuple(mask.shape)` with `tuple(ids.shape)`, rereading the
IDs shape, and device checks use `str(device)`. Failed predicates stop before
later checks. Original resource/numerical/freshness/deadline/ownership floors
remain. Model construction/loading/forward, decoder, scoring, training, added
synchronization and driver/TDR changes are forbidden.

The design is accepted for bounded CPU implementation and adversarial controls.
No native worker, supervisor, terminal auditor, final manifest or launch is
qualified here. A future positive result would support only this one input
construction branch at its execution time; it would not reproduce the preceding
253 model forwards, identify the initiating kernel/TDR cause, certify model fit,
complete M0 or answer the three-arm transfer question.
