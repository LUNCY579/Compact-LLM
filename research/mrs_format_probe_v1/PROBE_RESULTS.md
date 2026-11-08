# MRS-FP-1 — output-contract probe results

**OPEN DEVELOPMENT / MECHANISM SCOUT. Not confirmation. Not a release claim.**

Executed 2026-09-17 under the preregistration committed at `65f9093` **before** any probe
generation. Branch `research/mrs-format-probe-v1`, parent `06eb0ae`. MRS-1 freeze
`278ce0962bd10864fba7d8ce2e7248579f18b22495d176c4f0f40db903a68837` bound by hash and verified
unchanged; bank bytes unmodified; MRS-1 frozen source commit confirmed an ancestor.

| Run | Evidence | Calls | Wall | Peak GPU | Peak host RSS | State |
|---|---|---|---|---|---|---|
| `supa-1.7-frozen` | `evidence/probe-supa-002` | 120 | 46.6 s | 2.37 GiB | 2.32 GiB | unchanged |
| `d1-seed17-bf16-export` | `evidence/probe-d1-001` | 120 | 41.6 s | 2.37 GiB | 2.42 GiB | unchanged |

Zero optimizer steps. Zero weight modification: the post-inference parameter digest equalled
the pre-inference digest for both models.

## Primary result — format compliance (the registered metric)

Strict compliance = parsed AND terminated, n = 8 per cell.

| Contract | Supa R0 | Supa R2 | Supa R3 | D1 R0 | D1 R2 | D1 R3 | Eligible |
|---|---|---|---|---|---|---|---|
| `C0_JSON_MRS1` (control) | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | no |
| `C1_BARE` | **8/8** | **8/8** | **8/8** | **8/8** | **8/8** | **8/8** | **yes** |
| `C2_LINE` | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | no |
| `C3_JSON_PREFILL` | 8/8 | 8/8 | 5/8 | 7/8 | 0/8 | 0/8 | no |
| `C4_LINE_PREFILL` | 8/8 | 0/8 | 0/8 | 8/8 | 0/8 | 0/8 | no |

The control reproduced the MRS-1 native failure exactly (0/8 everywhere) under this probe's
own harness, confirming the probe measures the same phenomenon.

**Applying the registered rule, `C1_BARE` is the only eligible contract** — the sole contract
reaching the 0.75 threshold on every gated role for both models, and it does so perfectly.
Selection used format compliance only; correctness was not an input.

## The mechanism: this family emits exactly one field

The failure modes are consistent and diagnostic:

* Asked in prose for a container (`C0`, `C2`), the model **ignores the instruction entirely**
  and emits a bare answer. Instruction-following does not reach output formatting here.
* Given a prefilled opening (`C3`, `C4`), the model **completes that container correctly** —
  `{"answer": "` becomes `{"answer": "2"}`; `ANSWER:` becomes `ANSWER: 3`. Format can be
  imposed by prefill even though it cannot be imposed by instruction.
* But every checker cell fails with `checker field absent` or `two-line form not matched`:
  after emitting the answer the model **stops**. It never produces a second field.

So the binding limitation is not JSON, not syntax, and not the prompt. It is **arity**: the
family can emit one field, and one field only. A prefill chooses the container; nothing
observed makes it emit two.

`C1_BARE` was also the only contract needing no prefill, so it is the cheapest and the least
coupled to tokenizer details.

This is survivable for a multi-reasoning protocol. The checker booleans MRS-1 wanted are
recoverable from the answers themselves: R2's frozen instruction already says to answer
`UNKNOWN` when evidence is insufficient, so its abstention *is* its sufficiency verdict, and
R3 recomputes the answer, so disagreement with the candidate *is* its validity verdict.

## Descriptive correctness — recorded, and excluded from selection

Registered as descriptive and forbidden as a contract-selection input. **8 open items; far too
small for an efficacy claim.** Under `C1_BARE`:

| Model | R0 | R2 | R3 | Oracle ceiling over 3 roles | Role disagreement |
|---|---|---|---|---|---|
| `supa-1.7-frozen` | 3/8 | 3/8 | 3/8 | 3/8 | 0/8 |
| `d1-seed17-bf16-export` | 1/8 | 2/8 | 1/8 | 2/8 | 1/8 |

Gold on these 8 items is `{1:3, 0:3, UNKNOWN:1, 2:1}`, so a **constant "0" scores 3/8**.

* **Supa emitted `0` for all 8 items under all 3 roles.** Its 3/8 is the constant baseline, not
  reasoning. Zero input sensitivity, zero role disagreement, oracle ceiling equal to any single
  role.
* **D1 is input-sensitive** — it produced `3,0,3,1,1,1,3,3`, so the transition training did buy
  responsiveness to the input — but at 1–2/8 it scores **at or below the constant baseline**.

The bank is not at fault. Two SMOKE items were solved independently by hand and both matched
gold (`MRS1-38ba0e29…` → 1; `MRS1-70650c65…` → 0), and the bank carries its own
`forward == independent exhaustive` verification.

## What this licenses, and what it does not

Licensed: **building and freezing a successor instrument on `C1_BARE`.** MRS-1 is unrunnable
and this is the measured contract that replaces it. That instrument is required for Phase II
regardless of today's accuracy, because cells C and D need a harness that can execute.

Not licensed: any claim about multi-reasoning efficacy. Eight items cannot support one, and
format compliance is not competence.

Flagged for the successor's interpretation, not concluded here: on this evidence the plausible
bottleneck is **baseline task competence**, not reasoning diversity or selection. If a
checkpoint cannot beat a constant answer, the contrast A3 vs A2 has little room to mean
anything. The successor should therefore measure and report a **constant/majority baseline
alongside every arm**, so that "above chance" is always visible. That is a reporting
requirement, not a new pass/fail threshold.

SOFTWARE PASS: true. NATIVE EXECUTION PASS: true (both models, both probes).
OPEN DEV RESULT: format compliance only. MECHANISM SCOUT RESULT: not yet produced.
CONFIRMATORY CLAIM: **false.** RELEASE CLAIM: **false.**
