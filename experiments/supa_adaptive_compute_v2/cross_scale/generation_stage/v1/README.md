# Track B: prospectively gated generation screen

The complete original CPU alignment matrix and independent audit passed.
Its frozen selection is B3 at Supa post-L19 / donor L24. This subtree implements
the separately registered 128-case architecture-development generation screen.
Implementation/testing is not execution admission: source commitment, exact
bridge export, separate B-only XM calibration, final configuration/registration,
and per-cell runtime qualification still precede real screen generation.
Confirmation and downstream mechanisms remain closed.

`src/upstream.py` verifies/exports the one complete Stage 1 selection and the
same pair's shuffled/random controls. Export requires all 16 pairs and six
objectives, the aggregate and independent audit to remain complete, committed,
hash-verified and mechanism GO.
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
rotary geometry are preserved. The four-dose grid is .03/.10/.20/.40; zero gate
and disabled wrappers are identity diagnostics, not extra experimental cells.
Constructed treatment fields cannot be changed between or within contexts.

The wrappers restore the original Supa hook even when execution or integrity
checks fail. They check donor/bridge tensors, all buffers including nonpersistent
rotary frequencies, gates, identity mode, enabled state, insertion/donor indices,
RMS bounds, trainable flags and gradients across each context. They expose
separate donor parameters, bridge coefficients, active coefficients and the
historical dominant donor-plus-bridge matmul estimate. Downward bias contributes
to stored coefficients but not active work. Attention, normalization and
instrumentation remain excluded from the matrix FLOP estimate. The stage-owned
execution caller additionally checks whole-Supa/checkpoint integrity and all
stage gates.

XM is an independently initialized frozen SwiGLU with width 34,560, CPU-local
seed 2702, normal standard deviation .02, and unit RMS normalization. It has
106,168,320 matrix parameters and 212,336,640 dominant FLOPs/token, matching the
selected donor-plus-bridge matrix work. Its single alpha is calibrated against
the unscaled selected mapped donor residual using only B's 96 fit scenarios;
32 held-out calibration scenarios are report-only. The formula gives equal
scenario weight after averaging ordinary-token relative energy. No labels,
directional distillation, optimization, Native calibration or efficacy results
enter this fit. See `protocol/control_calibration_v1.md`.

The screen has X0 plus XG/XS/XR/XM at all four doses: 17 cells on the same 128
original B architecture scenarios. XN is unavailable because no best native
operator has been independently selected from completed Native confirmation.
Hard failures terminate only their cell, retain partial artifacts and contribute
no partial accuracy denominator. Other fixed cells continue; a dose with a
failed required cell is ineligible for selection. Complete cells are rescored
from raw responses and checked against saved first-token tensors.

The preregistered screening rule requires at least two paired XG wins and one
net correct gain over X0, at most one additional preservation error, no added
invalidity/safety stops, and XG accuracy at least each same-dose control. Rank
eligible doses by margin over the best control, then gain over X0, then lower
dose. This is candidate selection, not an efficacy claim. The exact contract
is in `protocol/generation_screen_v1.md`; all controls remain in the record.

Qualification saves independent CPU and GPU phases for every cell. CPU reference
fixtures can survive an interrupted GPU process. Eight own-track calibration
fixtures per cell cover all families. All first/repeated/disabled/zero/restored
raw outputs are hash-bound; admission recomputes numeric and identity checks.
Candidate GPU phases require a completed committed X0 CPU/GPU comparison.
Operational/resource failures remain pending review, while hard cell rejections
remain in the dose record. No qualification accuracy selects a dose.

The tests use tiny actual HF decoders and synthetic receipt fixtures. They
establish mechanics and rejection behavior, not generation safety or capability
on the canonical model. The new subtree is outside both existing track source
inventories; its own source map includes the new implementation, protocols,
scripts, tests and actual shared/legacy dependencies. Configs and artifacts are
bound separately to avoid circular fingerprints.

Execution order from the repository root, with the canonical Python environment:

1. Commit all stage source/protocol/test files after validation.
2. Run `scripts/export_selected.py` with the committed Stage 1 run/audit and a new
   export directory; verify and commit its manifest.
3. Run `scripts/calibrate_control.py freeze` with that export; commit the separate
   calibration registration before `preflight` and `run`. Verify/commit the report.
4. Run `scripts/freeze_stage.py config` to derive a new CPU/GPU config from those
   artifacts, then `register`; commit both before qualification.
5. Run `scripts/qualify_runtime.py` phases. Commit complete CPU references before
   GPU phases, and the complete X0 comparison before candidate GPU phases.
   Commit finalized cell comparisons before screen execution.
6. Run `scripts/run_campaign.py`; then `scripts/audit_screen.py` on complete or
   explicitly rejected cells. Only a screen GO permits a separate confirmation
   protocol/freeze. This caller cannot execute confirmation.

Existing evidence is never overwritten. New attempts are explicit; interrupted
workers require PID-verified abandonment. Missing local bulk artifacts are not
silently refitted. The original alignment GPU failure and all earlier Native
control failures remain unchanged.
