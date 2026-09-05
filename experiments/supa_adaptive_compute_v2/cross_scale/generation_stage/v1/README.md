# Prepared interfaces for a possible Track B generation stage

Status: engineering preparation only. No Stage 2 protocol, dose selection,
runtime qualification, model caller or generation license exists here. The
current Stage 1 matrix is incomplete. Its scientific source map and the Native
calibration source map must remain unchanged while these isolated interfaces
are developed and tested with tiny fixtures.

`src/upstream.py` prepares verification/export of the one complete Stage 1
selection and the same pair's shuffled/random controls. Real export is closed
until every one of the 16 pairs and six objectives, the aggregate, and its
independent audit are complete, committed, hash-verified and yield mechanism GO.
There is no partial-matrix selection, refitting, replacement seed or efficacy
input. The three exported tensor groups are XG, XS and XR, with exactly `up`,
`up_bias`, `down` and `down_bias`. Verification reuses the existing Stage 1
helpers with metadata-only activation handling; their imports may load torch,
but verification does not deserialize activation caches or load models.

`src/operators.py` supplies low-level X0 and frozen XG/XS/XR wrappers around the
preserved historical `Bridges`, `ResidualExpert` and `Graft` implementations.
The graft is exactly `h + g * ((L(project(h)) - project(h)) @ down)` after the
ordinary Supa insertion layer. The affine downward bias remains in the artifact
but cancels from the residual. Original causal masks/positions and donor-specific
rotary geometry are preserved. The existing four-dose grid is accepted for
mechanical tests; zero gate and disabled wrappers are identity diagnostics.
Constructed treatment fields cannot be changed between or within contexts.

The wrappers restore the original Supa hook even when execution or integrity
checks fail. They check donor/bridge tensors, all buffers including nonpersistent
rotary frequencies, gates, identity mode, enabled state, insertion/donor indices,
RMS bounds, trainable flags and gradients across each context. They expose
separate donor parameters, bridge coefficients, active coefficients and the
historical dominant donor-plus-bridge matmul estimate. Downward bias contributes
to stored coefficients but not active work. Attention, normalization and
instrumentation remain excluded from the matrix FLOP estimate. A future caller
must additionally enforce whole-Supa/checkpoint integrity and all stage gates.

The tests use tiny actual HF decoders and synthetic receipt fixtures. They
establish mechanics and rejection behavior, not generation safety or capability
on the canonical model. The new subtree is outside both existing track source
inventories; future stage admission must bind its own complete transitive map.

Still required after a complete mechanism GO: independently choose and freeze
the mandatory XM design and its own calibration rule, determine whether a frozen
XN is available, preregister the exact generation cells and dose-selection,
inertness/destructiveness, preservation and validity rules, implement the stage
lifecycle/analysis, freeze runtime and resource limits, commit the stage protocol
and registration, and qualify any chosen GPU profile. The existing Stage 1 GPU
failure is preserved. The 128-example Track B architecture bank and 1,024-example
confirmation bank remain unexecuted; confirmation requires further gates.

See `cross_scale/planning/generation_stage_interface_plan.md` for the remaining
interfaces. No command in this directory runs a real model or opens confirmation.
