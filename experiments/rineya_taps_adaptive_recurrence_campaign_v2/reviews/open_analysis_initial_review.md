# OPEN analyzer initial independent review

The analyzer remains blocked for outcome-dependent use until the source-only failures below are corrected. This does not change the reviewed OPEN native runner, matrix, admission or resource guards. Six author synthetic tests independently passed in0.72s. Twelve additional independent synthetic probes preserved in `open_analysis_review_probes_initial.json` exposed seven accepted contract violations.

| Required fix | Source evidence | Disposition |
| --- | --- | --- |
| Require exact fixed-alpha grid membership | `open_analysis.py:73` groups by `policy`; inherited `analysis.py:55` renders fixed alpha with six-significant-digit `:g`. The sixth1.1473425965417514 arm therefore admits substituted1.14734. | Blocking analyzer identity gate. |
| Verify extra unit updates remain alpha1 | `open_analysis.py:91` calls shared `validate_gpu_traces`; inherited `analysis.py:195` checks first-pass alpha andA2, without checkingA1. AlteredA1 extra alpha1.1 still passes. | Blocking unit-control gate. |
| Reject recorded worker RSS or host-reserve violations | `open_analysis.py:174` resource screen checks GPU allocation/global-free only; supplied parent RSS7GiB and native host reserve0 still pass. | Blocking safety evidence gate. No native run is implied by fixtures. |
| Validate initial block counts | `open_analysis.py:132` checks call/position/extra-layer counts, omits `initial_block_applications`. A count99 on a one-call row passes. | Blocking computation-accounting gate. |
| Require binary SMOKE task/format endpoints | `open_analysis.py:79` strict regrade runs only on new-bank IDs; aSMOKE score2 passes. | Blocking full-matrix schema gate; original SMOKE scorer should also be bound/regraded if feasible. |
| Enforce registered per-row generation caps | `open_analysis.py:121` derives counts from raw tokens but does not enforce8 OPEN/24 inherited SMOKE caps. A consistently journaled nine-token OPEN row passes. | Blocking intervention-identity gate. |

The new-bank scorer regrades strict canonical task/format/UNKNOWN contracts. Complete608-cell coverage,48 nonphysical oracle copies and16 frozen comparator aliases, repeat averaging, DEV-only pool and validation firewall checks are implemented. Original costs copied into aliases/oracles are excluded. Retrospective validation fixed envelopes remain diagnostic; the pre-frozen DEV comparator is the primary validation comparator. Capture has no clipping and is null when finite-grid headroom is nonpositive. Tiny exposed source-visible generator families remain exploratory; semantic natural-language abstention and reasoning beyond the scalar answer contract remain unassessed. None of these source checks establishes native qualification or efficacy.

No OPEN outcomes, model weights, protected/nativeTEST bank, actual GPU inventory or native worker were accessed by this analyzer review. Corrections belong to the engineer's analyzer/test ownership; reviewer artifacts remain independent and preserved.
