# Source and implementation map

Checked 2026-10-09. These are primary sources; the implementation is original code,
not a vendored upstream implementation. No upstream benchmark is reported as a
result of this experiment.

## Paper-specific behavior

Peng Jin et al., **ID Balancing: Stable Training of Extremely Sparse MoE via
PID-Based Load Control**, arXiv:2609.39137v1, 30 September 2026.
https://arxiv.org/html/2609.39137v1

Sections 2.2 and 3, Algorithm 1, Eqs. 2–3 and 10–12 specify: select using original
scores plus expert bias; normalize original selected scores for combination;
accumulate normalized load error and a derivative correction gated by
`previous_error * (error - previous_error) > 0`; then center the bias. The first
window has no derivative correction. Reference mode disables clipping and minimum
expected-load deferral. The exact-quantile baseline follows Eq. 7 on a small
uniform pool; it is not the distributed histogram implementation.

Mapping: `idr/control.py`, `idr/moe.py`, and `tests/test_control.py`.

## Architectural and API precedents

Dengchun Li et al., **MixLoRA**, arXiv:2404.15159v3.
https://arxiv.org/abs/2404.15159

Precedent for LoRA experts on a frozen dense backbone. IDR-001 uses an isolated
projection wrapper, not a reproduction of MixLoRA's full architecture/framework.

Lean Wang et al., **Auxiliary-Loss-Free Load Balancing Strategy for
Mixture-of-Experts**, arXiv:2408.15664.
https://arxiv.org/abs/2408.15664

Precedent for non-gradient load-bias updates; represented by the sign controller.

Official PEFT API: `from_pretrained`, `load_adapter`, `set_adapter`, `disable_adapter`.
https://huggingface.co/docs/peft/package_reference/peft_model

Official Transformers model loading API: local-only model loading and dtype.
https://huggingface.co/docs/transformers/main_classes/model

Official PyTorch safe tensor-state loading API:
https://docs.pytorch.org/docs/stable/generated/torch.load.html

## Rineya proposals, NOT established paper results

Eligibility/quality masks, per-operation pools, nonuniform targets, bounded
zero-mean projection, rare-target deferral, checkpoint custody, and the stated
promotion thresholds are IDR-001 experimental choices. Their correctness as
mechanisms is tested; their effectiveness on Rineya is unmeasured.

The small score-stream experiment is not a reproduction of 768-expert pretraining.
Quantile is currently a controller reference, not a native joint-training arm.
No speed, accuracy, stability, or frontier-intelligence gain is inferred from the
source papers for this specific implementation.

## Repository context

Read-only remote inspection established current ChatBubble main and its experiments
subtree before implementation. Root and experiments trees had no applicable
AGENTS.md. The historical CSS branch name did not resolve in this inspection;
therefore this branch neither imports its absent modules nor claims a complete
remote-branch duplicate audit. No registered RCWM/BWM data is copied here.

The preceding research review identified generalization failures as a separate
competence problem. This experiment retains that distinction through an
analysis-only oracle diagnostic and held-out challenge strata. Existing results
are motivation, not new verification or replication by IDR-001.
