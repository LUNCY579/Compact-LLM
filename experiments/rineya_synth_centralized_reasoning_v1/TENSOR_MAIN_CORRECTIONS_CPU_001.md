# Main-source provenance and required boundary maps: CPU corrections

External source review found two gaps in the preceding candidate: direct main
execution had no explicit-module compile descriptor for preparation to consume,
and the auditor could skip category derivation if both boundary maps were absent.
The exact preceding source graphs remain preserved; their CPU success receipts
do not cover these missing cases.

The auditor now requires matching boundary-counter and source-pin dictionaries
unconditionally, including legitimately empty dictionaries. Resealing a report
and final summary after removing both maps refuses. Root's six affected controls
pass in 1.661 seconds; independent four focused controls pass in 1.145 seconds
with 40 selected pins unchanged. They use CPU fixture metadata and establish no
native event facts.

The worker's counted phase1 source action now records a distinct direct-script
current-frame/source-equivalence observation. It validates the selected bytes,
actual main globals/file/cache/spec shape and bounded Python3.11 code fingerprints
against compiled source without executing that compiled graph. Preparation
rechecks retained function objects/code/globals. The explicit-source-module
compilation route remains separate. This does not prove initial loading history.

Four authored and Root controls pass; independent four pass in 0.217 seconds
with ten pins unchanged. The controls remove exactly the final main dispatch
before initializing an isolated CPU main surface; function/probe-frame bodies
remain unchanged. They exercise the actual helper and preparation record,
reject bad pins/namespaces/capabilities/function rebinding, and retain the imported
module route. No actual direct-script process or native phase1 was executed.

[Preservation manifest](evidence/tensor_main_proposal_corrective_cpu_20261007/ROOT_PRESERVATION_001.json)
binds 44 files and 1,001,492 bytes, including failed review fixtures and exact
predecessors. [Main review](evidence/tensor_main_proposal_corrective_cpu_20261007/independent_review/INDEPENDENT_DIRECT_MAIN001_REVIEW_SUCCESSOR.json)
and [boundary review](evidence/tensor_main_proposal_corrective_cpu_20261007/independent_review/INDEPENDENT_BOUNDARY_POPULATION_FIX001.json)
retain their distinct scopes. Current worker is
`6cd0a9fd598de3fc8bacc494eb67189f7bfa9963c280100d0f396ee315cc1828`;
auditor `985fed33d442b83c69db9bd919865537dd5427f0e690dec934f017ed8e181d4e`.

Public source promotion, current runtime/provider/interface qualification,
actual native main preparation and a separately approved tensor registration
remain pending. Native gates, resource/numerical limits and protected banks
remain closed. No native APIs, Torch, CUDA, model, training or scoring work ran.
The original three-arm transfer and prior-correct question remains unanswered.
