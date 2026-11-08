# Track A v2.3: prospective label-free control strength calibration

The original AM alpha=1 architecture attempt
`evidence/01_architecture_dev/20260905T220307_858493Z_7cc95fc0` failed its
malformed-generation gate. Its failure, weights, source, protocol and all earlier
evidence remain unchanged. The four native architecture arms completed 128
examples; no confirmatory outputs exist at this amendment. This new engineering
stage was designed after observing an architecture failure. It is not independent
confirmation and may not rehabilitate the failed v2.2 matrix. The old control
must remain in the historical record. This amendment changes only one scalar
strength for a separately registered v2.3 control.

Before any real-model calibration, freeze and commit this protocol, executable
config, caller, transitive sources, tests, original Track A dataset manifest and
calibration JSONL. `calibrate_control.py --freeze` creates an exclusive registration;
commit that registration before `--preflight` or execution. The caller verifies
the committed source/config/data contract before model loading. There is no CLI
override for a different dataset, partition, scalar rule, dtype or control seed.

Only original Track A calibration prompts are used: all 128 scenarios, with
the existing scenario-index modulo-four rule giving 96 fit and 32 holdout. The
label-free projection exposes prompt, prompt hash, example/scenario identity,
family/template and frozen partition. It does not consult answers, scorers,
programmatic answer oracles, generated responses, accuracy, NLL, architecture
outputs or confirmatory data. The frozen JSONL container contains labels; its
bytes are integrity-hashed and the parser immediately discards all nonallowlisted
fields. No answer validation or scoring function is called. Hashing the manifest
does not open any other dataset file. Track B data and outputs are excluded.

Use CPU float32/eager, four threads, seed 17, deterministic full-prefix inference
without cache or gradients. The same chat-template rendering as native inference
is used, with thinking disabled. Full prompts must have at most 256 tokens;
truncation and token subsampling are forbidden. Ordinary tokens exclude the union
of tokenizer special IDs and every added-vocabulary ID. A prompt without ordinary
tokens fails the run; no scenario may be dropped.

Stream embedding rows and actual installed HF decoder layers L0 through L13
using the preserved loader exposed by `shared.hidden_capture.load_layer` and
its original causal `positions_mask`. Do not load a whole causal LM or output
head. Keep raw post-L13 states for the complete prompts. On each state h13,
compute an actual extra L12 followed by L13 at the original position IDs, causal
mask and rotary embeddings, exactly as native A3 does. Define
`delta_A3 = L13(L12(h13)) - h13`. Separately compute the original frozen M(h13)
using `FrozenSwiGLU`, with unchanged local CPU seed 7301, normal std .02, width
10240, canonical RMS epsilon and gate/up/down draw order. No weights are fitted,
redrawn or changed; there are 31,457,280 frozen matrix coefficients.

For scenario s and each ordinary token t define
`e_native(s,t) = mean_hidden(delta_A3(s,t)^2) / max(mean_hidden(h13(s,t)^2), 1e-8)`
and the same `e_control(s,t)` with M(h13) in the numerator. First average over
ordinary tokens within each scenario, then equally over the 96 fit scenarios.
Choose exactly once:

`alpha = sqrt(E_fit_native / E_fit_control)`.

Diagnostic reductions use CPU float64 on float32 inference tensors. Require
finite positive energies and alpha. There is no alpha grid, optimization,
clipping, fallback, reward or quality-based choice. Freeze alpha to an exclusive
`fit_scale.json` before evaluating held-out transformations. Report the 32 held-out
scenario energies and `alpha^2 * E_holdout_control`; they cannot select, reject,
refit or clip the scalar. Held-out nonfinite tensors still fail numerical safety.
All-token, ordinary, special/control and final-prompt norms and energies are
reported separately. Empty special strata are null, never invented measurements.

Bind exact canonical checkpoint/tokenizer/config sidecars and installed HF source
hash to the preserved qualified CPU reference. Record every streamed prefix and
replayed layer's state hash before/after, rotary state, original control state,
prompt/token/mask/position/RoPE/embedding-row/hidden/delta hashes, full and retained
lengths, excluded token IDs, per-scenario metrics and separate fit/holdout IDs.
Frozen module inventories must show zero trainable parameters and zero gradients.
Recheck checkpoint bytes, source/config/data identities and module states after
the run. No full-model state hash is claimed for this streamed prefix operation.

Use the programme machine lease; heavy jobs remain sequential. Admission requires
1 GiB available physical RAM; RSS must remain below 6 GiB. Each individual prefix
layer/prompt forward and each complete two-layer/control measurement has a
900-second boundary-checked budget. The full stage has a 172800-second budget.
Forwards cannot be preempted safely. Resource, runtime, identity or numerical
failures preserve an exclusive failed manifest and all partial cases; retry only
into a new attempt. No elapsed failure licenses changing scientific settings.

Successful output is `evidence/00_control_calibration/<run>/report.json`, accompanied
by immutable started/final manifests, cases and fit-scale receipt. Before
downstream inference, commit the complete evidence and bind the exact report path,
SHA256, alpha and original control-state hash in a separate GPU v2.3 config.
`verify_calibration_artifact` independently reconstructs scenario aggregation and
checks exact formula, partitions, committed provenance, zero gradients and state
identity. A new CPU/GPU qualification and complete five-arm v2.3 architecture
matrix are still required before any v2.3 confirmation. Old CPU/GPU science
receipts cannot be mixed into the new paired matrix. A calibrated random control
is still untrained; comparison with it cannot establish superiority to a trained
same-scale alternative. All earlier efficacy, preservation and validity criteria
remain the responsibility of the separate v2.3 prospective campaign registration.
