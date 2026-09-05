# Track A v2.3: prospective calibration of the independent control's strength

This amendment follows the original alpha=1 AM architecture-development safety
failure in `evidence/01_architecture_dev/20260905T220307_858493Z_7cc95fc0`.
The original v2.1/v2.2 protocols, registrations, random control and all completed,
partial and failed evidence remain intact. The failure is evidence about that
original control. This amendment does not convert its failed gate into a pass.
Confirmation remains untouched. Architecture accuracy, answer NLL, generated
answers and confirmatory outputs do not determine the new alpha.

The only scientific change is a prospective, label-free scale calibration for
the existing AM transform. All A0/A1/A3/A16 operators, the AM initialization and
matrix work, frozen questions and data roles, scoring, statistical hierarchy,
primary comparisons, sample sizes, GO thresholds, preservation and validity
criteria, resource guards and closed downstream gates remain unchanged. AM keeps
its arm name; protocol/config/registration identity distinguishes its calibrated
v2.3 version. It remains an independently initialized, untrained generic
transformation. Matching residual strength does not confer native semantics or
make it a learned control.

## Frozen calibration rule

Use only the existing Track A calibration role and its frozen scenario partition:
96 fit scenarios and 32 held-out scenarios, with one original prompt per scenario.
Expose prompt and identity fields to
the calibration computation; do not expose answer or scoring fields. No Track B
data or efficacy roles may enter the fit. The held-out partition is report-only:
it does not select alpha, normalization, a seed, a dose grid, an arm or a cutoff.
All variants stay with their scenario. Preserve full prompt tokenization and
ordinary position/mask/rotary inputs; do not truncate or extract independent
context-free token forwards.

For ordinary post-L13 state `h`, define the native residual as the output of one
additional L12 then L13 block minus `h`. Define the unscaled control residual as
the existing `M(h)`. For each, measure token-relative energy as
`mean_hidden_dimension(delta^2) / max(mean_hidden_dimension(h^2), 1e-8)`.
Diagnostics are computed in FP64 from the FP32 inference tensors. Ordinary tokens
exclude the union of tokenizer special IDs and added-vocabulary IDs. First average
ordinary tokens within each original prompt, then scenarios with equal weight.
The single fitted scalar is exactly:

`alpha = sqrt(fit_native_relative_energy / fit_unscaled_control_relative_energy)`

Write the immutable `fit_scale.json` immediately after the 96-scenario fit and
before evaluating held-out residuals. There is no clipping, grid search, repeated
seed, efficacy objective, training,
held-out early stopping or post-fit discretionary adjustment. Nonfinite or
nonpositive fit energies or alpha fail the calibration. Report fitted and
held-out energies, scaled-control energies, counts and token-category diagnostics
without treating them as capability evidence. Preserve unsuccessful calibration
attempts in their original exclusive directories.

The transform remains the same CPU-float32 normal draws in gate/up/down order,
seed 7301, standard deviation .02, width 10240 and Supa width 1024. Unit-scale
RMS normalization keeps the canonical epsilon. It retains 31,457,280 frozen
matrix parameters and 62,914,560 dominant multiply-plus-add FLOPs per token,
matching A3's two added native layers under the original estimator. The v2.3
operator computes `h + alpha M(h)` after ordinary L13; no additional native
mechanism changes. Timed latency and memory must still be measured independently.

## Freeze and provenance

The estimator configuration is `configs/control_calibration_v2_3.json`; use
`native_armr/scripts/calibrate_control.py --freeze` to create its separate
calibration registration and `--preflight` for a model-free admission check.
Commit this amendment, the calibration source/config, dataset manifest and
calibration preregistration before calibration model execution. Calibration uses
the registered frozen CPU FP32/eager/no-cache reference with four threads and
the existing machine lease; no model or GPU jobs overlap another heavy worker.
Stream the actual canonical embedding and ordinary L0-L13 decoder tensors, then
replay L12/L13 on the captured post-L13 states. The calibration does not load the
causal-LM head, score answers or generate tokens. Its separate resource contract
requires 1 GiB available RAM, at most 6 GiB process RSS, 900 seconds per prompt
operation and a 172800-second outer budget, with admission checked at execution.
Bind the complete report to its immutable calibration run, committed registration,
source hashes, manifest, fit/held-out IDs, checkpoint identity, model/control state
digests and zero trainable/gradient inventories. A report is eligible only if
its formula and provenance independently verify and the report and required
provenance files are committed.

Only after such a report exists, run
`native_armr/scripts/freeze_calibrated_control.py --calibration-artifact <report.json>`.
It creates `configs/protocol_gpu_v2_3.json` exclusively, deriving alpha directly
from the verified report. There is no command-line alpha override. The config
binds the report's canonical repository-relative path and exact SHA256, and
records protocol version `native_armr_v2.3_gpu_fp32_calibrated_control`.
An existing final config is never replaced in place. A new scientific adjustment
would require a later amendment and version.

Freeze and commit a new native registration with this final config and the
unchanged native dataset manifest. Every v2.3 protocol load verifies the committed
artifact, exact alpha, original checkpoint/control identity and full unchanged
scientific configuration before the caller can load a model. AM construction also
requires its actual initialized tensor digest to equal the calibration digest
before forward execution. The CPU reference copy used by GPU qualification keeps
the same calibrated alpha and transform bytes. Arbitrarily editing alpha, changing
the report, substituting another seed/checkpoint, stale source provenance, or a
missing commit fails admission; no execution switch bypasses these checks.

## Independent qualification and complete architecture matrix

GPU runtime settings and all numeric/identity tolerances remain exactly those
of v2.2: CUDA:0 FP32, eager, no cache, TF32 disabled, deterministic computation,
and the same CPU and GPU memory limits. The new runtime qualification ID is
`native_gpu_fp32_v2_3`. Run the existing `scripts/qualify_native_gpu.py` with
`--config native_armr/configs/protocol_gpu_v2_3.json` and the unchanged manifest.
It must independently pass all five arms on the eight calibration fixtures,
with all original per-cell logit/NLL/hidden and full greedy-identity thresholds.
The old qualification does not qualify this changed AM.

After qualification passes and its receipt is committed, complete a fresh
128-example architecture matrix for all five arms under the v2.3 registration.
The operational architecture order is A0, AM, A1, A3, A16, using the existing
campaign caller's explicit `--arms` option. Checking the changed control after
its complete baseline can reveal another mechanical failure before spending
compute on unchanged candidate arms. This changes neither case allocation nor
the required complete five-arm matrix; all arms still use their frozen seeds,
operators and same 32-shard allocation. Confirmation uses the original arm order.
The original A0/A1/A3/A16 evidence remains valid historical development evidence
but cannot be imported into the v2.3 paired matrix: source/config/registration
and qualification identities must agree. Complete the matching A0 shard before
each candidate shard, keep the same 32 deterministic shards, and reject malformed
generation, additional invalid outputs, safety stops or any original mechanical
gate failure. No architecture result licenses changing fitted alpha or deleting AM.

Only a complete passing v2.3 architecture matrix licenses its unchanged primary
and diagnostic campaign. The 1,024 primary scenarios and 128 separate diagnostic
variants retain their original definitions. A recurrence-specific GO still
requires the original hierarchical A3-vs-A0 then A3-vs-AM tests and every original
effect-size, preservation, validity and cost-reporting requirement. Neither
calibration nor runtime qualification supports a capability claim. Keep CPU,
v2.2 GPU and v2.3 GPU scientific evidence separate. Routing, MLA/MHA, MTP, PEFT,
backbone training and sealed evaluation remain closed.
