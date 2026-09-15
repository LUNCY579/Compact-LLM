# RRP-001 results so far — goal remains active

**No pretrained model has been evaluated or updated in this iteration. There is no new model-quality result.** The deliverable required by [GOAL.md](GOAL.md) remains incomplete.

## Completed evidence

| Check | Actual result |
|---|---|
| Source archives / extracted tree |142 files byte-verified against the two supplied archive hashes|
| Supa source inventory |596,049,920 serialized BF16 parameters;314,618,880 in220 tail tensors; native instantiated recount pending|
| Opallo historical selection |41 exact ordered rows recovered from matching10,000-row pool|
| Fresh gold/trace checks |512 v1 TRAIN targets;256 v2 TRAIN worlds and structured traces|
| Actual tokenizer survey |768 unique TRAIN worlds, direct v1 and direct/reasoning v2 contracts; no Torch/model load|
| Full v2 microfit preparation |Failed before selection: zero answerable evidence-chain cases fit2K+128 reserve|
| Scope-only v2 preparation |Same failure; full canonical target records preserved; failed artifacts retained|
| Primitive microfit input |32 TRAIN worlds,8 families,496–851 full tokens;20,907 processed/148 supervised tokens per exposure|
| Listed overlap keys |Zero selected-world collisions against the declared open banks; semantic/near-duplicate qualification still pending|
| New code tests |36 passed,0 skipped; includes fresh-instance full-state resume equality, budget refusal and create-only evidence|
| Supplied v2 tests |222 passed,0 skipped in6.693s; synthetic verifier/format/environment mechanics|

The new engine reuses MQ chunked CE, CPU AdamW, WSD, RNG and full-state checkpoint components. Its final fixture suite performs six tiny CPU optimizer updates across the full/resumed test trajectories; those are not Supa updates or a native resume qualification. Two engine-suite invocations occurred during development; their raw XML is retained. Source/mask/loss algorithms in the inherited MQ package are unchanged.

## Proposed first model experiment — unrun

[MICROFIT_PLAN.json](MICROFIT_PLAN.json) fixes exact Supa, tail20, response+EOS, FP32, LR1e-5, betas(.9,.95), WSD20/60/20, decay0 and clip1. Eight repeated exposures mean **32 unique worlds,256 presentations,167,256 processed tokens,1,184 supervised tokens and105 update groups**. This is an explicit successor to the old25k single-pass lane, not an unannounced extension of an old permit.

The prospective engineering criterion is at least31/32 canonical training answers correct, finite gradients and intact frozen tensors. It does not measure new-world generalization or ordinary-response preservation. Preserve update2 and final full-state checkpoints; a fresh-process resumed trajectory must reproduce final parameters and predictions. The native supervised controller and this actual source/shape/resume path remain unqualified.

The pending envelope caps total owned work at3,600s, model-worker/GPU time at2,400s, all-attempt training tokens at400,000, evaluation/qualification requests at160, generated tokens at20,480, new disk at16GiB, one model-selection round and one retry. No RL rollout, teacher/cloud spend or release is included. An evaluation request means one prompt evaluation (including its fixed-probe CE and greedy generation); actual prefill/forward/decode counts must also be recorded. This does **not** mean160 individual transformer forwards.

## Still required for the full goal

- An approved finite envelope, enforced accounting/controller, current resource admission, zero-update native parity/shape/gradient checks and actual microfit/resume evidence.
- Reviewed preservation data, exact TRAIN lineage contracts and complete shortcut/near-duplicate audits for the proposed40%preservation/20%v1/40%v2 token mixture.
- A/B/C sequential SFT comparisons and qualified4K/longer contexts or genuinely checked shorter traces; no arbitrary trace/evidence truncation.
- On-policy RL only after verified SFT competence, adversarial verifier tests and nonzero rollout signal; current222 tests do not qualify a native GRPO trainer.
- Fresh independent confirmation, multiple training draws for reproducibility, preservation/abstention/format/cost cards, exported-dtype qualification and separately budgeted distinct-lineage replication.

No plateau, capacity limit, superiority, release candidate or improved checkpoint is claimed. The32-world microfit is one prerequisite within the original goal, not a replacement goal.
