# Centralized Rineya 0.6B research protocol

Frozen initial design: 2026-09-28. Source foundation:
`03db218ceebc61a2bc04f2482cf7e094045e01c1` on `origin/post-training-Mimo`.
Execution branch: `research/post-training-mimo-centralized-v1`.

## Authority and scope

The current user brief authorizes autonomous, bounded qualification, original data
construction, experiments after their scientific gates, and incremental commits
and pushes. The report's proposal-only authorization fields and smaller 344M
planning envelope are superseded by this brief. This does not waive scientific,
resource, provenance or held-out access gates. No paid cloud or teacher expenditure
is registered in this cycle. No changes or merges to the parent branch or main.

The complete supplied 66-page report was read. Its SHA256 is
`a1c5b5c66e1cef0edd04631ab7c487c93731b1a207bb67ee87fddd40cf43c2d7`.
The report's literature claims remain RECORDED FROM PRIOR WORK unless independently
verified. No paper implementation is inferred from a method name.

## Primary student

Select `Rineya-Titan-Experimental-1.0.0-devfreeze`, the original dense BF16 Titan,
with weight SHA256
`537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7`.
Measured tensor inventory: 596,049,920 unique serialized parameters, 310 tensors.
All primary descendants must bind `receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json`.
Runtime qualification is separate from static identity. No 344M GPU training.

Selection is based on the user's Titan preference, local availability, exact
published provenance, and a simple dense BF16 representation. No new capability
ranking among Titan variants was measured. Titan-Sparse and Titan-Transition are
FP32 composed endpoints; introducing their composition/dose and rounding effects
would add variables to the initial qualification. They remain historical evidence.

## Experiment 0

Question: can this exact Titan runtime perform unchanged inference and strictly
confined rank-8 q/v LoRA updates with reproducible save/reload/resume?
Use only non-evaluation mechanical fixtures. Register tolerances, seeds, scope and
ceilings before running. Stop the affected run on any failed check. Never widen a
tolerance after a failure. Keep failure receipts and all incurred compute.

The stock local Transformers Qwen3 architecture must load every checkpoint tensor;
the only absent state key permitted is tied `lm_head.weight`. Compare tensor content
digests against the source and verify the loaded config, shape and tying contracts.
One GPU worker at a time. No teacher, RL, remote code or network model downloads.

## Experiment 1

Question: does completion-only SFT yield transferable acquisition while preserving
existing behavior? Parent is the exact unchanged Titan. Independent variable is
one epoch of registered rank-8 q/v LoRA training versus the unchanged base.

Acquisition: 4096 original parents, 1024 each arithmetic, register updates, finite
sets and constrained text. Replay: 1024 separately qualified training-safe rows.
Seed 101 first, with 211 and 307 conditional on the DEV screen. Selection seed
20260927. Completion mean CE, then sequence mean; 16 acquisition +4 replay per
20-sequence accumulation; microbatch 1. AdamW LR 1e-4, betas .9/.999, epsilon 1e-8,
weight decay 0, clip 1.0; 5% warmup then linear decay. Rank8/alpha16/dropout0;
base frozen. BF16 base and FP32 adapter/optimizer states if E0 qualifies.
Context ceiling1024, target128; never truncate a verified target. One epoch,
256 steps, checkpoints/DEV at 0/64/128/192/256. Final checkpoint only.
Per-seed ceilings: 5,242,880 nonpadding training tokens, 24 GPU/wall hours;
generation ceiling 262,144 tokens including all failures, teacher calls0.
Actual data counts and token sums must be frozen before E1; ceilings are not
measurements or license to fill missing replay with acquisition examples.

Independent 256-parent TRAIN_SELECTION and 256-parent DEV banks are added under
AMENDMENT_001. CHECK acquisition512 and preservation512 remain closed until final
checkpoints and the seed decision are frozen. FINAL remains unopened and no final
sample size is asserted before a confirmation/power design. Existing exposed banks
cannot become fresh by re-splitting. Acquisition success does not certify language,
retrieval, abstention or long-context preservation.

## Decisions and evidence

DEV seed screen: gain >=3 percentage points, no aggregate preservation loss >2pp,
no represented family loss >5pp, malformed/EOS failure increase <=1pp each.
Passing earns the two confirmation seeds; failure earns diagnosis, not more seeds.
Three-seed pilot: positive acquisition >=2/3 seeds, descriptive mean >=3pp,
paired parent-bootstrap 95% acquisition interval excludes0, same preservation
screens. This is exploratory screening, not a noninferiority certificate.

Bootstrap entire parents with all variants/seeds together (10000 resamples,
seed20260927). Report per-seed and per-family effects, source-correct losses and
source-incorrect repairs; zero denominators are undefined. Report all attempted
tokens, calls, failures, resources and runtime. Do not pool row-by-seed as independent.
No CHECK-driven hyperparameter selection. FINAL requires a frozen candidate and
separate prospective confirmation/multiplicity policy.

Use MEASURED / REPRODUCED, RECORDED FROM PRIOR WORK, VERIFIED SOURCE, INFERRED,
PROPOSED, UNMEASURED, BLOCKED, NEGATIVE RESULT literally. Only passing registered
behavioral promotion gates may update CURRENT_BEST_RECIPE.

## Later order

Qualification -> verified SFT -> order-only curriculum or isolated preservation
or scope diagnosis -> WSD/data selection -> matched RSFT/GRPO/MaxRL -> sampler
comparison -> competent per-skill donors -> composition -> actual MiMo transfer.
GFlowRL needs reference/behavior likelihood and canonical reward-maximizing control;
DPO needs ranked same-prompt pairs. F-WANDA is post-hoc. Architecture and inference
tracks are independent, require matched controls, and do not block simple SFT.
