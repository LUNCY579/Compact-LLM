# Track B Stage 2: prospective frozen generation dose screen

This separate Track B stage follows the complete independently audited Stage 1
mechanism GO. The selected transformation is B3, inserted after ordinary Supa
layer 19 using Qwen3-4B donor layer 24 (zero-based). It was selected from all 16
depth pairs and all six objectives using calibration alignment only. All Stage 1
metrics, gates, data, candidate ranking and failed receipts remain unchanged.
Stage 1 GO establishes transformation fidelity under that test, not capability.

The code lives exclusively under `cross_scale/generation_stage/v1`. It reuses the
existing model loader, deterministic generation, strict exact/JSON scorer,
output validity guards, paired statistics, quality/cost reporting, byte hashing
and machine lease. Its sources, artifact bindings, runtime qualification,
configuration, registration, evidence and decisions are independent of Native
ARMR and of the completed alignment stage. Neither original stage is rewritten.

## Frozen inputs and the 17 experimental cells

Use exactly the already frozen Track B `architecture_dev` bank: 128 independent
original scenarios, 16 each from eight families. The complete manifest contract
is `6857158ee7a360bdf2f72a21c5ee78041ab0a5ecda264c0387e1e5facf638a46`;
the unchanged scorer contract is
`62937db0530db2f8b43672ff1ea3e9ffa96dce5e8b7c9cf5bac6996b3d29c11e`.
All cells receive the identical complete set. There is no prompt truncation,
resampling, template substitution or post-output answer edit. Track A efficacy
rows and answers never enter fitting, qualification or selection. Calibration
is a distinct existing Track B 96-fit/32-holdout role. The 1,024 independent
confirmation originals and all confirmation diagnostics remain closed.

| Cell family | Fixed operation |
|---|---|
| X0 | Intact Supa, once across the 128 cases. |
| XG | Exact selected S19/D24 B3 bridge and frozen donor. |
| XS | Exact BS shuffled bridge from the same original S19/D24 artifact. |
| XR | Exact BR random bridge from that same original artifact. |
| XM | Independently initialized, frozen Supa-width residual transformation below. |

XG, XS, XR and XM each run at gates **0.03, 0.10, 0.20, 0.40**, unchanged from
Stage 1's prospective future grid. This gives 17 cells, identified by X0 and
`XG_g0.03`, `XS_g0.03`, etc. Neither zero gate nor disabled/restored diagnostic
forwards are additional experimental cells. XN is explicitly unavailable: at
this freeze no independently selected best Native recurrence and completed
Native confirmation exists. A plausible Native candidate cannot be substituted
or selected using Track B outputs. Native efficacy evidence stays separate.

Export admission requires the complete committed Stage 1 matrix, selection,
attempt inventory and independent PASS audit, including failed attempts. The
export contains byte-identical XG=B3, XS=BS and XR=BR tensor subsets with all four
affine tensor keys and exact original tensor hashes. There is no new fit,
precision conversion or normalization of exported coefficients. The graft is
exactly `h + g * ((L(project(h)) - project(h)) @ down)`, using the inherited
RMS-normalized Supa upward input, full causal context, original positions and
donor rotary geometry. Downward affine bias is retained for artifact identity
and cancels from the residual. Donor and backbone remain frozen.

XM uses one independent frozen SwiGLU at Supa layer 19, hidden width 1,024,
intermediate width 34,560, seed 2702 and Gaussian standard deviation 0.02. It
uses unit-RMS input normalization with epsilon 1e-6 and fixed SiLU. Its
106,168,320 active matrix coefficients add 212,336,640 dominant matrix FLOPs per
token, matching the selected donor-plus-bridge estimate. There is no training,
dummy work, seed search, architecture-answer fit or intentional weakening.
Report this dominant matrix estimate separately from full measured generation
cost; attention, normalization, instrumentation and teacher-forced answer NLL
are excluded from this estimator.

The separate stage-owned XM calibration registration defines one label-free
positive scalar from the 96 Track B fit scenarios: match scenario-equal ordinary
token residual energy of the unscaled selected graft. Holdout is report-only;
no scalar search or clipping is permitted. Exact alpha and control-state digest
are derived only from the complete committed verified calibration artifact.
Its report must bind the same export manifest. This is a new B calibration,
never an imported Native scalar or efficacy decision. Final configuration is
absent until these artifacts exist. `build_config` derives it with no alpha
override; the caller writes it exclusively and commits its registration before
any real qualification or generation.

## Reference runtime, safety and independent GPU admission

Reference execution is CPU FP32, eager attention, four threads, no cache,
deterministic algorithms and seed 2702. Generation uses at most 48 new tokens,
256 input tokens, a 900-second per-example deadline including prefill and gold
NLL, an 8-GiB RSS ceiling and hidden output/input RMS bounds [0.25,4.0]. CPU
admission requires 3.5 GiB available after imports. GPU uses only CUDA:0 FP32,
TF32 off, highest matrix precision, deterministic cuBLAS, no offload, 0.75 GiB
available RAM, 4 GiB available VRAM and a 6.5-GiB CUDA reserved-memory ceiling.
These are separate explicit profiles; the CPU phase of qualification uses its
3.5-GiB admission floor. Model jobs remain sequential under the shared lease.
Deadline and resource checks occur at forward boundaries; an overrun is a
failure, never a truncated successful sample.

GPU requires this stage's own committed per-cell qualification. Use eight
deterministic calibration fixtures, the minimum example ID in each family.
Compare CPU/GPU first logits with maximum absolute error 0.002, relative L2
0.0002 and minimum cosine 0.999999; answer NLL absolute error is at most 0.002.
Capture Supa hidden states at layers 2, 19 and 27 with relative L2 at most
0.0002 and cosine at least 0.999999. Require exact greedy outputs, repeated GPU
outputs/diagnostics, and disabled/restored identity. Store all raw comparisons;
verify complete Supa, donor, bridge, control and checkpoint hashes and no
gradients before/after. Qualification does not select on scores. Neither a
Native GPU qualification nor the failed Stage 1 GPU qualification admits this
stage. Baseline qualification must pass before candidate admission.

A cell's hard qualification failure excludes that cell from GPU generation
and rejects its entire dose for screen selection. All other fixed cells,
including that dose's qualified siblings, still run for descriptive evidence.
Actual generation failures likewise terminate only their own cell. Preserve
every raw partial output and immutable FAILED receipt; never compute partial
accuracy or synthesize sibling failures. An interrupted attempt needs verified
process termination and explicit recovery evidence; a live or uncertain worker
is never presumed dead. Resource admission failure before any example is
distinct from numerical/qualification/generation rejection. Root lifecycle
controls resumptions without overwriting failed evidence or mixing profiles.

## Fixed screen decisions and reporting

X0 must complete all 128 cases. Its failure stops the campaign. Final selection
waits until every one of the 17 fixed cell IDs is complete on the exact 128 IDs
or has its own terminal qualification/generation failure. Unknown, unfinished,
duplicate or mismatched evidence cannot yield a selection.

A dose is eligible only when all four same-dose cells complete all 128 cases,
have zero safety stops and no additional invalid outputs versus X0. XG must:

1. Have at least two paired wins and at least one net correct answer versus X0.
2. Add at most one error over X0 in the combined 48-case preservation stratum:
   ordinary QA preservation, instruction/evidence and retrieval/multihop.
3. Reach accuracy at least as high as each same-dose XS, XR and XM. Ties are
   allowed in this small selection screen; stronger specificity needs confirmation.

The preservation condition concerns the difference in total errors in this
predefined combined stratum; paired wins/losses there remain fully reported.
No output-identity, KL or NLL threshold is chosen after results. The required
positive paired decision changes already reject an inert graft.

Rank eligible doses first by largest XG accuracy minus the best same-dose
control, then largest XG accuracy minus X0, then lower gate. Freeze at most one
XG dose. Report **all four doses**, all mandatory controls and every failure,
including why each dose failed. Never remove a control that performs well.
No eligible dose yields SCREEN_NO_GO; it is useful evidence about this tested
transfer mechanism, not proof against all possible cross-scale architectures.

Use unchanged shared paired analyses: raw accuracy, wins/losses, exact McNemar,
95% scenario-bootstrap intervals (10,000 replicates, seed 2703), absolute and
relative error reduction, family effects, answer NLL and output identity.
Report first-token cosine/KL, captured hidden RMS, trajectory RMS range,
invalidity, token-cap terminations, measured latency, peak RAM/VRAM, active and
resident parameters and estimated work. Describe the measured quality/cost
frontier without fitting a scalar utility. Independent audit must reconcile
raw rescoring, source/config/data/tensor hashes, fixtures, every receipt and
ranking before any next-stage claim. A failed cell has no accuracy estimate.

This is architecture selection on 128 examples. P-values and confidence
intervals are descriptive; neither SCREEN_GO nor alignment GO establishes a
generalizable capability improvement. SCREEN_GO licenses preparation of a new
prospective committed confirmation protocol only. It does not directly open
the 1,024-case bank. No routing, 8B, MLA/MHA/MTP, PEFT, backbone training,
distillation or sealed evaluation is licensed here.
