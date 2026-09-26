# Fixed CPU-budget replication, v2

Prospective amendment: frozen and committed before any new Supa forward.
This is an unsealed development replication, not a confirmatory/sealed study.

The sole experimental change is **max_prompt_seconds: 90 -> 900**. The prior
run's two long prompts used approximately 15–22 seconds for prefill and reached
the old deadline after only five generated tokens. Their unchanged token caps
are recorded in the dataset. A 900-second per-case generation deadline allows
the bounded decode to finish with substantial allowance for CPU variability.
It is a cooperative deadline checked between complete Torch forwards, using
the existing implementation. Gold NLL and integrity checks retain their original
semantics. No adaptive extension of this deadline is allowed during the run.

Keep exactly the original matrix and order:

1. A0/A1/A3 smoke qualification, eight original cases each.
2. A0 intact Supa, all 16 MERGEDEV-fast cases in original order.
3. A1 L12, two total applications, the same 16 cases.
4. A3 contiguous L12–L13 block, two total block applications, the same 16 cases.
5. Restore and compare all intact development logits; hash state/files afterward.

Keep checkpoint/tokenizer, custom abstention head, CPU FP32/eager execution,
four threads, seed 17, deterministic greedy decoding, no KV cache, positions,
RoPE, masks, prompt rendering, per-case token caps, NLL computation, output
scorers, validity checks, layer choices, gates, source datasets and arm selection
unchanged. A2 stays excluded. No router, PEFT, training or sealed access is part
of this replication, regardless of the resulting development gate.

The JSON registration pins the prior receipt, selected atlas, complete effective
configuration, unchanged execution/data files and existing comparator function.
The comparator already includes the censored-baseline correction recorded before
this amendment; no new scoring or selection rule is introduced. Runtime and
checkpoint/state hashes must match the prior real run before execution proceeds.

Report every planned case. Preserve timeout, token-cap, EOS and safety-stop
categories separately. A timeout or other invalid baseline means the quality
comparison remains inconclusive. Numerical/cache/mutation failures stop the run;
other case-level stops retain the original bounded-case semantics. Never retry
a stopped case or change parameters to obtain a positive result.

Audit the unchanged choices mechanically after the run: exact matrix/case IDs,
configuration difference limited to the time budget, file hashes and runtime,
unchanged generation IDs for all previously non-censored cases, and identical
old generated prefixes for previously timed-out cases. First-token logit metrics
must also reproduce. Report any mismatch rather than assuming time was the only
effective difference.

Primary reporting: completed versus censored cases, accuracy and paired
corrections/regressions for A1–A0 and A3–A0, family preservation, invalid rate,
EOS/truncation, logits/hidden dynamics, latency/FLOPs/RSS, unchanged-state checks
and the existing development gate. One corrected development example is not
proof of general capability improvement or adaptive routing value.

Reproduce with the original Python environment, after the JSON registration has
been frozen in Git:

```text
python experiments/anchor_routed_middle_recurrence_v1/scripts/run_fixed_recurrence.py --atlas experiments/anchor_routed_middle_recurrence_v1/evidence/01c_completed_candidate_atlas/20260905T042420_466033Z --preregistration experiments/anchor_routed_middle_recurrence_v1/preregistrations/fixed_cpu_budget_v2.json
```

Use a fresh immutable `evidence/02c_fixed_cpu_budget_v2/<timestamp>/` directory.
The original failed and inconclusive evidence remains unchanged.
