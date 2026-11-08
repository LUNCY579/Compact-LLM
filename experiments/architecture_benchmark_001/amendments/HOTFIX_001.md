# HOTFIX-001 — numerical integrity and bounded recovery engineering

Date: September 10, 2026 (America/Chicago). This is an explicit implementation/protocol amendment to RAB-001, not a retrospective reinterpretation of a successful experiment.

## Preserved scientific contract

The official Supa source lock, four parameter-matched primary shapes, 128-dimensional heads, causal shared encoder-final memory, donor selection and FFN growth policy are unchanged. Decoder-local attention, sparse attention, MoE, recurrent sharing, quantization changes and tokenizer substitutions remain forbidden. Calibration, pilot and training groups remain disjoint. Original evidence files are retained byte-for-byte.

G3 recovery, G4 registered DEV and G5 sealed evaluation remain unimplemented and closed. Filling a readiness form does not authorize them. No acceptance criterion is selected using a student's quality outcomes. There is no new quality, pretrained parity, 8-GB fit, throughput or deployment result in this delivery.

## Explicit gate changes

The uploaded CED acceptance rule used exact argmax plus a tensor-global scale bound after strict elementwise failures were observed. This amendment rejects that substitution. The existing numerical constants are retained, not loosened: for canonical expected logits a and tested logits b, every scored finite element must satisfy `abs(b-a) <= 1e-4 + 1e-4*abs(a)`. Argmax agreement is additionally required. The global-scale bound, TV, KL, margins and violation counts are diagnostic only.

Qualification now uses the first eight already-open pilot rows, chunk sizes 1, 7 and 16 plus a halves schedule, tail-only prefill, prefix-length comparison and same-shape suffix perturbation to test causal isolation. This is a **prospectively expanded diagnostic grid**, not evidence that the original one-prompt failures disappear. The prior strict failures remain failures under their originally reported metric. FP64 is an observation-only comparison lane, not permission to deploy or train FP32 after a failed FP32 gate.

Schema-2 receipts require named checks, stage-specific artifacts, current implementation/runtime/data/source identity and recursively validated actual parent receipts. Schema-1 records cannot be relabelled or upgraded. POSIX path keys make the implementation digest platform-independent; file bytes and runtime remain separately binding. Old historical Windows digests are still meaningful in their original format.

## Bounded optimization extension, separate from recovery

A new opt-in G2_CACHED_<arm> engineering lane reconstructs full-vocabulary teacher logits from saved final-normalized FP32 teacher states and the frozen official tied head. It uses the same CE/forward-KL objective, temperature 1, alpha 0.5, SGD without momentum at 1e-5, clipping 1, exactly 25 steps and seed 17. Backbone and head recomputation reduce retained activations; no quantization or low-rank adapter is introduced. These settings are fixed in protocol.json and hashed.

B28 requires fresh G0; D40 requires fresh G1_WIDTH. C28 additionally requires strict G1_CED_C28 and matched G2_CACHED_B28. C40 requires strict G1_CED_C40 and matched G2_CACHED_D40. The control and CED arm must use identical data, cached teacher targets, recipe and seed. A CED strict-parity failure blocks its optimizer lane. These engineering smokes do **not** fulfill the recovered-ordinary-donor requirement of a later C40 scientific campaign.

Teacher preparation is not training authorization. No teacher backbone runs inside the cached student loop; the CPU vocabulary projection still does real work. Projection work is instrumented including backward recomputation; cache preparation and shared cost attribution are separately recorded. Do not claim this is free distillation or that it fits an 8-GB GPU without measuring it.

## Required local decision

Read REVIEW_REPORT.md and RUNBOOK.md before executing. First qualify the pinned native environment with no native skips, then perform fresh official G0/G1 execution. A future G3 implementation requires reviewed and frozen datasets, budgets, controls, endpoints, effect/regression margins and measured resource limits. Do not add `--force`, widen tolerances, relabel a failed receipt or tune on sealed TEST.
