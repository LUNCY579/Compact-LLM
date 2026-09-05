# A0/A1/A3 CPU-budget replication

**Status: PASS. The full development matrix completed without censoring, and
the audit confirms that the time budget was the only experimental change.**

The preregistration was committed and pushed at
`af7c49d432a652c55cbafc5155c378d87497cf24` before any new Supa execution.
The deadline changed from **90 to 900 seconds per prompt**. Checkpoint,
tokenizer, selected layers, data/order, generation token caps, CPU FP32/eager,
four threads, seed 17, cache-free execution, scoring and existing gates stayed
unchanged. The original 90-second failed/inconclusive evidence is preserved.

## Results

| Arm | Architecture | Correct | EOS completions | Timeouts | Paired wins / losses vs A0 |
|---|---|---:|---:|---:|---:|
| A0 | Intact Supa-A1.1 | 12/16 (75%) | 16/16 | 0 | — |
| A1 | L12, two total applications | 12/16 (75%) | 16/16 | 0 | 0 / 0 |
| A3 | L12–L13 block, two total block applications | 13/16 (81.25%) | 16/16 | 0 | 1 / 0 |

Both previously timed-out retrieval cases were answered correctly by all three
arms. No development generation reached its token cap. Maximum complete-case
latencies were 195.9 s (A0), 196.4 s (A1) and 224.3 s (A3), comfortably below
the registered budget. The unchanged deadline implementation checks between
complete Torch forwards; reported case latency also includes optional gold NLL.

The block's sole paired correction is the previously observed counterfactual
transitivity question: every yellow object is coral, every coral object is
ochre, and N is yellow. A0/A1 answered **NO**; A3 answered the required **YES**.
It had no paired losses and no family-preservation or validity regression.

This resolves the earlier censoring: A3's existing one-case improvement survives
the now-complete development preservation check. It is not new independent
evidence for that already inspected example, a general capability claim, or
evidence that adaptive routing adds value.

## Fidelity and mechanical qualification

All **19/19 audit checks passed**:

- Exact same 72 case-runs and order: 24 smoke plus 48 development generations.
- All 66 previously non-censored generations have identical token IDs.
- All six previous timeout generations are exact prefixes of the completed outputs.
- All 72 first-token logit-metric comparisons have maximum difference **0**.
- Configuration differs only in `max_prompt_seconds`; 13 preserved execution,
  configuration, data and scorer files match the registered hashes.
- Runtime/library versions, Qwen implementation, selected arms, checkpoint and
  loaded state match the prior run. All restored development logits are bit-identical.
- Canonical weights/files are unchanged; zero trainable parameters and zero gradients.

The original eight-case A1 smoke result reproduced, A3 passed its independent
smoke gate again, and A2 remained excluded. No screen retry, additional layer
selection, router, PEFT, training or sealed evaluation occurred.

## Diagnostics and resources

| Arm | Mean logit cosine vs A0 | Mean KL | Observed RMS ratio range | Mean canonical-answer NLL (9 cases) | Development wall time |
|---|---:|---:|---|---:|---:|
| A0 | 1.0000 | 0 | 1.0000–1.0000 | 1.4952 | 409.0 s |
| A1 | 0.9891 | 0.03746 | 0.99984–1.00495 | 1.4186 | 420.8 s |
| A3 | 0.9838 | 0.05422 | 0.99967–1.01029 | 1.3064 | 458.9 s |

One Supa load, CPU FP32/eager, PyTorch 2.12.1+cu126, Transformers 5.14.1,
Python 3.11.15, four threads, seed 17. Total model-run wall time: **1406.5 s
(23.4 min)**. Peak RSS: **4.34 GiB**. KV memory: zero; no GPU inference.
Timing includes instrumentation and varies with CPU load. The arms add different
amounts of computation, so these measurements are not a compute-matched efficacy
comparison. Dense-matmul estimates and full hidden/logit metrics are archived.

## Gates and scope

- **Fixed preservation:** A1 and A3 PASS on the complete registered development set.
- **Existing router-scaffold development gate:** A3 now meets it; A1 alone does
  not, because it supplies no paired correction. This is eligibility for a future
  scaffold experiment, not a claim that a router was implemented or is useful.
- **PEFT recovery and sealed evaluation:** remain unlicensed.
- **This approved experiment:** complete at the identical matrix and its audit.

The sample remains only 16 unsealed, previously inspected development cases.
The one-case difference is exploratory. No general intelligence, independent
held-out efficacy, long-context reasoning or learned-routing result is claimed.

## Evidence and reproduction

- [Prospective protocol](preregistrations/fixed_cpu_budget_v2.md)
- [Frozen configuration and hashes](preregistrations/fixed_cpu_budget_v2.json)
- [Model-run report](evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/report.md)
- [Complete raw results](evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/raw_results.jsonl)
- [Run manifest](evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json)
- [Independent audit report](evidence/02d_cpu_budget_v2_audit/20260905T135230_002601Z/report.md)
- [Audit decision and all checks](evidence/02d_cpu_budget_v2_audit/20260905T135230_002601Z/decision.json)
- [Prior-output identity comparisons](evidence/02d_cpu_budget_v2_audit/20260905T135230_002601Z/prior_generation_identity.csv)
- [Dynamics and efficiency](evidence/02d_cpu_budget_v2_audit/20260905T135230_002601Z/dynamics_efficiency.csv)

Run from the repository root using the original Python environment at
`C:/Users/clclc/BusinessProject/Chat-Bubble/.venv/Scripts/python.exe`:

```text
python -m pytest -q experiments/anchor_routed_middle_recurrence_v1/tests
# 65 passed, 0 failed

python -m compileall -q experiments/anchor_routed_middle_recurrence_v1/src experiments/anchor_routed_middle_recurrence_v1/scripts experiments/anchor_routed_middle_recurrence_v1/tests
# PASS

python experiments/anchor_routed_middle_recurrence_v1/scripts/run_fixed_recurrence.py --atlas experiments/anchor_routed_middle_recurrence_v1/evidence/01c_completed_candidate_atlas/20260905T042420_466033Z --preregistration experiments/anchor_routed_middle_recurrence_v1/preregistrations/fixed_cpu_budget_v2.json

python experiments/anchor_routed_middle_recurrence_v1/scripts/audit_budget_replication.py <new-run-directory>
```

New tests reject changes to threads, cache, token cap, seed, candidate layers,
dtype and unregistered deadlines. Identity tests reject altered complete outputs,
changed timeout prefixes and logit drift. Existing model mechanics, state/cache
guards and development-gate tests remain passing. No tests for unimplemented
training or routing are claimed.
