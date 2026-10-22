# Sources, provenance and decisions — 2026-09-15

## Repository of record

All additions are scoped to `experiments/cross_scale_stitch_graft_v1/` in ChatBubble, on a new branch directly parented by `fde553c908db39e87a15a52ed239d1ba0a68bfd3`. The inherited root tree is `5aa393b03ed518b642883cd1b368f9a77675538e`.

The original cross-scale branch is preserved as a dependency. No old protocol, source module, model asset, prediction, report, workflow or other branch is rewritten. The package is an additive research framework, not a merge of CED and cross-scale experiments. It imports none of the inherited evaluation-bank generators or sealed-data readers.

## Inspected source locations

Paths below are under `Rineya_Cross_Scale_Capability_Transfer_Lab_v1/` at the immutable parent unless stated otherwise.

| Source | How it informs this work |
|---|---|
| `README.md` | Old study scope and distinction between unsuccessful operators and the general transfer question |
| `src/rineya_cross_scale/direction_transfer.py` | Existing linear-map and weight-projection conventions; new affine fitting is independent and does not alter old numerics |
| `src/rineya_cross_scale/functional_translator.py` | Prior low-rank diagnostics; avoid presenting in-sample proxy values as new held-out results |
| `src/rineya_cross_scale/aligned_projection.py` | Existing embedding-fitted maps and matrix orientation; new package uses site-specific activation maps |
| `cross_scale_evidence/02_architecture/compatibility_report.md` | Exact shape inventories, tokenizer comparison, FP16 custom A1.1 identity, tied versus untied output tables |
| `cross_scale_evidence/13_v3_transport_diagnosis/report/v3_1_donor_side_report.md` | Per-depth functional correspondence is motivation, not transferred-capability evidence |
| `cross_scale_evidence/13_v3_transport_diagnosis/report/correction_0001_e7_metric_withdrawn.md` | Preserve the E7 withdrawal and correct A1.1 versus 1.7 distinction |
| `cross_scale_evidence/14_v4_aligned_transport/protocol/amendment_v4_aligned_transport.md` | Existing proposed weight transport remains separate; matching K/V dimensions is not proof of matching coordinates |
| `cross_scale_evidence/15_supa_layer_ablation_screen/report.md` | Eight-prompt identity-skip sensitivity does not establish graft locations or trainability |

The old `direction_transfer.ridge_linear_map` is not reused at runtime because it is an uncentered FP32 helper coupled to the older checkpoint module. This package needs an affine intercept, FP64 centered streaming moments, capture identity, split isolation and independent held-out reporting. That separation avoids changing past results or importing old bank/loader behavior.

## Primary external references consulted

- Alan Chen, Jack Merullo, Alessandro Stolfo and Ellie Pavlick, **Transferring Linear Features Across Language Models With Model Stitching**, arXiv:2506.06609v3 (2025). Source: https://arxiv.org/abs/2506.06609v3 . Supports studying affine residual-space connections and feature transfer; does not establish improved Rineya task performance.
- Hugging Face Transformers **v4.51.3** Qwen3 implementation: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py . Reviewed native block ordering, post-attention normalization, explicit head dimension, Q/K normalization, rotary buffers and no-cache block signatures. The adapter uses supplied native modules; it does not copy or vendor that implementation.
- PyTorch linear-algebra documentation: https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html . The implementation is tested against the locally available Torch build, with a direct independent FP64 solve comparison. The live documentation redirect is not a runtime pin.

These references inform a new experimental design, not a claimed reproduction of a published result.

## Decisions and corrections to tempting shortcuts

**D1 — Preserve subject identity.** Keep historical Supa-A1.1 separate from Supa Core 1.7. Reject unsupported custom logits instead of quietly taking a dense stock path.

**D2 — Prefer intact computations.** Start with a donor FFN and fitted boundaries rather than assuming coordinate-wise weight projection preserves its function.

**D3 — Fit translations for states, not deltas.** An affine input map can include an intercept. A transported FFN displacement uses only the linear return map.

**D4 — Raw sites are explicit.** The final hidden-state tuple of a library model may be final-normalized. Do not treat it as a raw block output by index alone. Stitch and graft maps are separately fitted.

**D5 — Shape compatibility is not semantic alignment.** Matching K/V head counts and dimensions does not eliminate head/basis correspondence. Learned diagonal normalization gains and nonlinear FFNs do not generally commute with arbitrary rotations.

**D6 — Parameter matching is not compute matching.** The ordinary adapter matches the bridge trainables, not the frozen donor parameters, FLOPs or capture cost. The latter must be measured separately.

**D7 — Zero is only a control.** Exact host preservation with bypass/zero gain is a mechanics result. A useful candidate needs a nonzero path and task-level evidence.

**D8 — Keep candidate identity complete.** Retained sources, sites/cuts, bridge states and gain define a candidate. Its suffix or host identity alone is not sufficient.

**D9 — Separate implementation from execution.** This approval preserves ideas and implements complementary code. It does not reopen native model calls, training authority, old holdouts, workflow dispatch or deployment.

**D10 — Keep the first comparison interpretable.** Do not mix WSD, new corpus construction, backbone recovery and a new cross-scale operator in the first experiment.

## What the delivery evidence means

The source-bound CPU suite exercises the new package only. The parent repository's full test suite is not claimed as rerun. Additive Git-tree preservation, not a fabricated inherited-suite count, establishes that existing files were not modified.

The test suite includes real random-fixture causal forwards, bidirectional stitches, nonzero grafts, same-source/bypass parity, gradient flow through frozen components, source-tensor immutability, artifact round trips/tampering, leakage refusal, paired-statistics checks, resource refusal and an end-to-end demo. The optional native helper's missing-runtime refusal is tested; no real Transformers/Qwen adapter parity is claimed.

An initial independent-reference test used a default FP32 identity matrix in an otherwise FP64 ridge reference. That fixture was corrected to explicitly FP64 rather than relaxing its tolerance. The final result is recorded in `evidence/LOCAL_VERIFICATION.json`; development failures are not relabeled as native observations.
