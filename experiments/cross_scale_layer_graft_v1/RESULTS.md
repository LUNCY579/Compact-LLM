# Mechanically viable; no observed development gain

**Completed 2026-09-05.** One frozen Qwen3-4B L16 was inserted after ordinary Supa L12 using fitted residual bridges and a fixed gate of 0.1. All layer indices are zero-based.

The native Supa pathway, early stem, custom abstention head and canonical checkpoint were preserved. One donor application was tested; the experiment trained only affine bridge coefficients by ridge regression.

## Main result

The aligned graft scored **12/16**, versus **12/16** for intact Supa: **0 paired wins and 0 losses**. 16/16 greedy outputs were identical. Mean first-token cosine was 0.997334, and KL was 0.005037 nats.

Measured development wall time increased by **16.3%**; estimated dominant linear-matmul work increased by **24.1%**. The added operator has **100,930,816 frozen parameters**, with **5,246,464 bridge coefficients**. Latency includes instrumentation and optional gold-answer scoring; serial order and host load confound timing. The FLOP estimate excludes attention, normalization, instrumentation and gold-answer scoring. This is not a compute-matched efficacy result.

Mean reference-answer NLL on 9 cases shifted from **1.4952** to **1.4790** (lower is better). This descriptive probability shift did not change the generated answers.

## Development controls

| Arm | Correct | Wins / losses vs A0 | Identical outputs | Mean cosine | Mean KL | Gold NLL | Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 intact | 12/16 | 0 / 0 | 16/16 | 1.000001 | 0.000000 | 1.4952 | 367.0s |
| G aligned | 12/16 | 0 / 0 | 16/16 | 0.997334 | 0.005037 | 1.4790 | 426.7s |
| R random bridges | 12/16 | 0 / 0 | 16/16 | 0.998402 | 0.002521 | 1.4384 | 438.4s |
| S shuffled pairs | 12/16 | 0 / 0 | 15/16 | 0.998850 | 0.002767 | 1.5405 | 435.0s |
| A1 native L12 repeat | 12/16 | 0 / 0 | 15/16 | 0.989133 | 0.037465 | 1.4186 | 394.6s |

The random-bridge control improved reference-answer NLL more than the aligned graft. Therefore the aligned arm's small NLL improvement does not demonstrate a benefit specific to learned correspondence.

The earlier A3 L12–L13 block scored 13/16 in the separate ARMR replication. A3 was not rerun here. The single large-layer graft did not reproduce that block's counterfactual correction.

## Mechanical controls

| Arm | Correct | Mean cosine | Mean KL | RMS range | Pass |
|---|---:|---:|---:|---|---|
| A0 | 6/8 | 1.000002 | 0.000000 | 1.000000–1.000000 | True |
| Z | 6/8 | 1.000002 | 0.000000 | 1.000000–1.000000 | True |
| I | 6/8 | 1.000002 | 0.000000 | 1.000000–1.000000 | True |
| N | 6/8 | 1.000002 | 0.000000 | 1.000000–1.000005 | True |
| R | 6/8 | 0.998636 | 0.005956 | 1.000000–1.000024 | True |
| S | 6/8 | 0.999075 | 0.006209 | 0.999613–1.000000 | True |
| G | 6/8 | 0.999612 | 0.003998 | 1.000000–1.001518 | True |
| A1 | 6/8 | 0.988599 | 0.073012 | 0.999848–1.000029 | True |

Z is exact gate zero; I is an identity donor; N is the aligned donor at sigmoid(-8). Z/I were bit-identical in logits and greedy IDs. R/S/G use gate 0.1. All smoke arms passed the frozen mechanical gate, including the negative controls. A small tolerable perturbation alone does not demonstrate useful transfer.

## Bridge diagnostics and data boundaries

Affine maps were fitted on **2544 tokens**, with **526 tokens held out by scenario**. The 59 calibration prompts split into 47 training and 12 held-out prompts; all paraphrases of a held-out scenario stayed together. Three calibration prompts were truncated to 256 tokens. Full development prompts were used. No answer label entered bridge fitting.

| Map | Held-out up cosine | Held-out down cosine |
|---|---:|---:|
| aligned | 0.6120 | 0.8574 |
| shuffled | 0.1582 | 0.5919 |
| random | -0.0018 | -0.0024 |

On held-out prompts, the donor delta from projected Supa states versus native donor states had cosine **0.4432** and relative MSE **3.6576**. Absolute-state alignment did not yield equally strong transformation alignment.

Post-hoc diagnostics (no map/gate changes):

| Held-out subset | Up cosine | Up mean token-relative MSE | Down cosine | Down mean token-relative MSE |
|---|---:|---:|---:|---:|
| all_heldout_tokens | 0.6120 | 6.6693 | 0.8574 | 0.2695 |
| text_token_ids_below_151643 | 0.5852 | 7.4157 | 0.8444 | 0.2914 |
| last_prompt_token | 0.7111 | 1.8540 | 0.9458 | 0.1193 |

Global relative MSE masks substantial text-token error: the upward map's global relative MSE over all held-out tokens is 0.002393, but it is 6.7726 over ordinary text tokens alone. High-norm chat tokens dominate the first metric. This is a concrete limitation of the present absolute-state fitting objective, not a proven explanation of the quality result. No objective or gate was changed after inspecting these diagnostics.

All 8/8 smoke prompts are in the calibration bank, and 6/8 were used for bridge fitting. They are mechanical checks, not held-out efficacy. Development has 0/16 exact prompt overlap with calibration, but it is a small, previously inspected procedural set with related templates. No sealed data was accessed.

## Integrity and validation

**34/34 independent audit checks passed.** The run recorded 144 generations: 144 EOS completions, 0 safety stops, and 0 invalid outputs. Zero/identity controls and post-run restoration were exact. A0/A1 outputs and first-token metrics reproduced historical receipts exactly.

Supa and donor loaded-state hashes matched before/after, checkpoint-file hashes were unchanged, and there were zero trainable backbone parameters and zero gradients. The real streamed Supa fixture was bit-identical to the canonical model. Source and dataset hashes matched their pre-run receipts.

CPU FP32/eager, four threads, no KV cache, no GPU inference. Total model-run time: 39.8 minutes; peak process working set: 5.01 GiB. Twenty meaningful tests passed across preflight and auditor suites; see [VALIDATION.md](VALIDATION.md).

## Interpretation

This prespecified graft can be executed stably, but this run supplies no capability gain that justifies its additional parameters or compute. It does not reject other donor layers, stronger alignment, or delta-oriented bridge objectives. It provides no support yet for combining this operator with routing, MLA, MTP, or repeated refinement.

A follow-up should first address held-out transformation alignment and retain the shuffled-pair control. Any new map objective or gate sweep needs a separate prospective experiment; this run's settings and historical evidence remain fixed.

## Artifacts

- [Frozen protocol](README.md) and [configuration](protocol.json)
- [Run report](evidence/20260905T141942_899240Z/report.md), [manifest](evidence/20260905T141942_899240Z/run_manifest.json), and [raw cases](evidence/20260905T141942_899240Z/raw_results.jsonl)
- [Bridge fit](evidence/20260905T141942_899240Z/bridge_fit.json) and [post-hoc token diagnostics](evidence/20260905T141942_899240Z/posthoc_alignment_diagnostics.json)
- [Independent audit](evidence/audit_20260905T150001_276474Z/audit.json)
- `graft.py`, `run.py`, `audit.py`, `diagnose_alignment.py`, `summarize.py`, and `tests/` reproduce the implementation and checks.

Local activation and bridge tensors are ignored by Git and referenced by SHA256. No Rineya checkpoint was trained, exported, committed, or published.
