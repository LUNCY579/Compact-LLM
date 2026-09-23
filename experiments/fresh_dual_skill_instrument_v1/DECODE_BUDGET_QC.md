# Decode-budget QC — is max_new_tokens=48 censoring any correct answer?

A decode budget that truncates correct answers biases systematically toward arms that answer
tersely and against arms that reason out loud. The audit panel found exactly that defect in the
peer probe (32 tokens, three cells truncated mid-answer). This checks whether this bank has it.

## Measurement

From the v1 calibration predictions (192 rows x 3 arms, `max_new_tokens=48`):

| arm | correct | median generation | longest generation | truncated? |
|---|---|---|---|---|
| BASE (Supa) | 2 | 35 chars | 129 | yes, at the cap |
| SPARSE (Titan-Sparse) | 63 | 22 chars | **35** | no |
| OPALLO | 1 | 34 chars | 86 | yes, at the cap |

Longest gold answer in the bank: **35 chars**. Longest *correct* generation: 34-35 chars.

## Verdict: no censoring bias

The donor never comes within the budget. SPARSE's longest output of any kind is 35 characters,
exactly the longest gold, and it terminates on EOS rather than the cap.

The two arms that do hit the cap are not being cut off mid-derivation — they are **echoing the
prompt back**:

```
BASE:   {"before":{"X":93,"Y":92,"Z":19},"event":{"kind":"swap",...},"after":{"
OPALLO: {"before": {"X":93, "Y":92, "Z":19}, "after": {"X":93, "Y":92, "Z":19}, "after": {"X":
```

Under exact-match against a bare state object, `{"before":...,"after":{...}}` is wrong at **any**
budget, so raising the cap cannot convert these into correct answers. The truncation lands only
on outputs that are already format-non-compliant. This matches the recorded instrument, where
Supa also scores near zero (6/384) under the same rule.

## Cross-lane replication of the wall-time signature

The wall-time asymmetry this produces was independently observed in a different lane, on a
different bank, with no shared code (DOSELORA-001, 2026-09-22):

| lane | source/base model | trained arm | ratio |
|---|---|---|---|
| this instrument | 298.5s (BASE) | 134.3s (SPARSE) | 2.2x |
| DOSELORA-001 DPACQ | 548.8-642.2s | 249.5-308.6s | ~2.2x |

Same cause in both: the base model does not emit EOS and runs to the token cap, while a trained
arm stops early. Worth recording because that peer's first pass at a contention analysis pooled
source runs with trained arms, saw a 105% spread, and briefly read it as a contention signature.
It is not — it is the source-versus-arm split. **A wall-time spread that separates cleanly by
arm class is a termination-behaviour signature, not evidence of resource contention.**
