# JumpLoRA × Rineya skill isolation protocol

Status: preregistered implementation and qualification protocol. No efficacy
run is authorized by this package.

## Primary mathematical source

The source of mathematical truth is the v3 paper, not a secondary summary:
[JumpLoRA: Sparse Adapters for Continual Learning in Large Language Models](https://arxiv.org/abs/2604.16171v3), arXiv:2604.16171v3, revised 2026-04-28.

For a base weight `W_base`, the paper uses a LoRA update

`ΔW = A · B`, with `A ∈ R^(d_in × r)` and `B ∈ R^(r × d_out)` in the
paper's matrix convention, and merges `W_base ← W_base + ΔW` after each task.
The implementation stores the equivalent PyTorch `A[r, d_in]` and
`B[d_out, r]`, so its materialized update is `B @ A`.

The paper defines the one-sided gate (Equation 1):

`JumpReLU_τ(x) = x · H(x − τ)`

where `H(x)=0` for `x≤0` and `H(x)=1` for `x>0`. Because `ΔW` has both
signs, the symmetric sparse update is (Equation 5):

`ΔW_jump = JumpReLU_τ(ΔW) − JumpReLU_τ(−ΔW)`

which is equivalent to `ΔW ⊙ H(|ΔW|−τ)` under the strict gate.

The Heaviside derivative is replaced by the paper's straight-through
pseudo-derivative (Equation 2):

`∂ JumpReLU_τ(x) / ∂τ = −(τ/ε) [H((x−τ)/ε + 1/2) − H((x−τ)/ε − 1/2)]`.

The implementation applies that band-limited threshold derivative to both
signed terms of the symmetric gate. The input derivative is the hard-gate
identity on active coordinates and zero on inactive coordinates.

To avoid blocking learning at the paper's zero-initialized `B`, the effective
update is interpolated (Equation 3):

`ΔW_interp = (1−γ) · ΔW + γ · ΔW_jump`.

The schedule (Equation 4) is:

`γ = clip((s−S_start)/(S_final−S_start), 0, 1)`.

Algorithm 1 initializes `A` with Kaiming uniform, `B` with zeros, and uses
`β=α/r`. For the first `S_start` steps, training is dense. At `S_start`,
`τ` is initialized so the number of entries with `|ΔW|>τ` equals the number
of trainable parameters in `A` and `B`. `γ` then increases linearly until
`S_final`; after that the sparse path is used exclusively. At completion the
paper forms `ΔW_final=(A·B)⊙H(|A·B|−τ)`, merges `β·ΔW_final`, and removes the
task adapter.

The paper reports `ε=0.001`, one epoch per task, AdamW, total batch size 32,
constant learning rate `1e−3`, `S_start=20%` of an epoch, `S_final=80%`,
query/value targets, rank 8, and `α=32`. This experiment preserves the
compatible adapter geometry and q/v placement. It does not copy the paper's
dataset, task order, or H200 budget into Rineya.

## Rineya binding and declared differences

The repository's actual checked-in reference configuration is
`Rineya_Published_Cores/Rineya-Pulse-1.6.0-devfreeze/config.json`. It records a
Qwen3-compatible 28-layer, hidden-size-1024 family configuration. The exact
runtime checkpoint and tokenizer must be selected and hashed on the
development run; this package does not silently substitute a checkpoint.

The four arms are:

* **J0**: ordinary dense LoRA.
* **J1**: the same dense training path, followed by deterministic global
  magnitude top-k pruning. Its final active count is bound to the frozen J3
  development result; it is not tuned on held-out or sealed data.
* **J2**: dense warmup, then a magnitude top-k support chosen at the registered
  warmup step and frozen for the remainder of training.
* **J3**: the paper's learnable threshold, symmetric JumpReLU, and gradual
  dense-to-sparse schedule.

J1 is a matched-density control, not a claim that post-training pruning is a
literal JumpLoRA implementation. J2 is a timing control for support freezing.
The code keeps the arm names and support rules explicit so a receipt cannot
silently collapse the comparison into a cosine-only analysis.

## Measurement contract

For every A/B pair and both acquisition orders, the efficacy runner must bind
the same ordered item IDs and report individual skill performance, base
retention, A+B and B+A composition, per-item repair/damage, delta cosine,
support sizes, intersection/union/Jaccard, layerwise overlap, parameter
counts, peak GPU memory, host-RAM low-water mark where measurable, wall time,
throughput, and exact train/evaluation split hashes.

The qualification runner exercises the metric and provenance contracts on a
synthetic CPU micro-run only. It loads no Rineya checkpoint, tokenizer, or
dataset and rejects sealed/ANY_READ paths.

## Freeze and success rules

No threshold, rank, learning rate, sparsity level, seed, pair, layer placement,
or other hyperparameter may be selected from a sealed/private confirmation
set. An efficacy arm is frozen on development/calibration data before any
held-out confirmation evaluation is authorized.

The preregistered positive-result rule is unchanged in `config.json`: J3 must
be within one percentage point of J0 on individual skill aggregate, reduce
merged damage by at least 25% relative and 2 percentage points absolute, lose
no more than one percentage point of base retention, agree in direction across
three seeds, examine both orders, and have lower support overlap than J1 when
claiming learned coordinate isolation.
