# Phase 0 audit

Status: PASS (source/repository audit; model execution pending).

Question: can the historical FP32 Supa ablation be extended without canonical
source/cache mutation or sealed-data access?

Setup: current workspace ChatBubble-cross-scale, starting branch
research/rineya-cross-scale-qwen4b-8b, clean HEAD fde553c. New dedicated branch
research/supa-a1.1-anchor-routed-middle-recurrence-v1. No submodules in this
checkout; published core source is an ordinary tracked tree. Actual 1.19GB
weights resolve to the sibling Chat-Bubble published-core vault, as in the
historical manifest. Expected SHA256 a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026.

Inspected layer_ablation.py, run_supa_layer_ablation_screen.py, validation.py,
resource_trace.py, leakage.py, evaluation/runner.py and scoring.py, original
ranking/report/manifest, core config and custom forward. Supa is Qwen3 with
28 layers, hidden size 1024, tied embedding/head and UNKNOWN offset 5.156249985
on token 51378. Preserve that custom class and offset.

The sibling SupaExperiment repository has recurrent-adapter and normalized
gate-LR branches, plus a remote recurrence-atlas-review-v2 design branch.
Its current branch is research/supa-a1.1-recurrent-normalized-gate-lr-v1,
HEAD 1f1284d, with untracked research/ user work. Read-only audit only.
Its model submodule is pinned to 199eed212ab3a3db487270dc6f97b16756f7bff0;
MODEL_BASELINE.md records the same checkpoint weight hash. Its separate FP16
GRU/controller protocols are not the present FP32 layer-reapplication experiment.
The atlas branch's recurrence_design_lab_v1 explicitly reports CPU toy/reference
work, no real Supa load. Neither supplies evidence that middle decoder layers
are repeatable. No prior branch was merged or edited.

Runtime: default PATH Python is a different Hermes environment (torch 2.13 CPU,
transformers 5.15). Use sibling Chat-Bubble/.venv Python with torch 2.12.1+cu126
and transformers 5.14.1, matching the checkpoint/historical era. CPU 4 threads.
No GPU job is interrupted. Source inspected for both installed Qwen versions;
actual loaded implementation will be hashed in the manifest.

Cache findings: Qwen3DecoderLayer returns a Tensor; Qwen3Attention applies RoPE
then unconditionally updates any non-None past_key_values at self.layer_idx.
The model constructs DynamicCache when use_cache=True and derives positions from
cache length. Direct inputs_embeds and hidden-state wrappers are supported.
ARMR must reject both cached mode and supplied cache objects at model entry;
repeated passes reuse exact position/mask/RoPE arguments. Model-backed intact
prefill/decode cache equivalence will be recorded in baseline qualification.

Sealed policy: existing Invisible-Limit TEST2/stage6 and sealed private paths
remain closed. Inspected only firewall code/protocol and allowlisted smoke data.
Later fixed qualification may read MERGEDEV-fast. No sealed contents were read.

Interpretation: wrappers can isolate recurrence from cache and weight confounds.
Non-claims: no repeatability or efficacy conclusion yet. Next gate: freeze this
protocol, test the harness, and reproduce historical anchor/soft/null ablations.
