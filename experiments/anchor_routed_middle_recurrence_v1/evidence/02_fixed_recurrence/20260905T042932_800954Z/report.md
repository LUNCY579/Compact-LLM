# Fixed middle recurrence development qualification

Status: FAIL for development preservation; mechanical qualification PASS.

Question: does the frozen selected recurrence preserve the existing unsealed
MERGEDEV-fast suite, and does it license an adaptive-router scaffold?
Setup: CPU FP32/eager Supa-A1.1, seed 17, four threads, full-prefix no-KV;
16 development items. A1: L12, two total applications. A2 licensed: False.
A3 block: (12, 13); independent smoke gate passed: True.
Selection and gates were frozen before development outputs. Source provenance,
dataset hash, parameter/checkpoint hashes and resources are in run_manifest.json.

Mechanical qualification: A1 reproduced the original completed smoke outputs
and logits; block execution/order has separate unit tests and an eight-case
real-model smoke gate; all development baseline logits restore bit-identically.

Intact development baseline: 10/16.

| Arm | Correct | Paired wins | Paired losses | Logit cosine | Preservation | Router licensed |
|---|---:|---:|---:|---:|---|---|
| A1_dev | 10/16 | 0 | 0 | 0.9891 | False | False |
| A3_dev | 11/16 | 1 | 0 | 0.9838 | False | False |

Interpretation: each arm must preserve overall accuracy, lose no family by more
than 0.10, add no invalid outputs and correct at least one previously missed
answer to license routing. Family deltas and paired case IDs are recorded in
the comparison JSON files. A frozen operator can be mechanically stable yet
fail this development gate. The original L27 campaign failure remains FAIL;
only completed, audited middle operators were eligible here.

Non-claims: no training, router, PEFT, long-context reasoning or sealed efficacy
result. Sixteen development items and mostly one item per family are a small
screen. Serial CPU timing includes instrumentation and is not deployment speed.
No compute-matched efficacy claim is made.

Next gate: STOP adaptive escalation; retain the fixed-operator evidence and preregister one further causal development test.
