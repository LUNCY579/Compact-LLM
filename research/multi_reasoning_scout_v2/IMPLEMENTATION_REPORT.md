# MRS-2 — implementation report

## Lineage

```
MRS-1  codex/multi-reasoning-frozen-scout-v1-completion   freeze 278ce096…
  |    instrument built, frozen, software-passed; native runs blocked on host RAM
  |    2026-09-17: RAM recovered -> both models loaded, both failed SCHEMA_SMOKE 0/8
  |    CLOSED. Cells A and B NOT PRODUCED. Both MODEL_LAUNCH tokens spent.
  v
MRS-FP-1  research/mrs-format-probe-v1
  |    measured 5 output contracts x 3 roles x 8 open SMOKE items x 2 models
  |    C1_BARE is the only eligible contract; the family emits exactly one field
  v
MRS-2  research/multi-reasoning-scout-v2                  freeze 297aa886…
       successor instrument on C1_BARE; bank reused by hash
       both models completed 120 items -> cells A and B PRODUCED
```

No historical branch was modified. MRS-1's freeze, bank, prompts, preregistration and
evidence are byte-unchanged and re-verify.

## What was built

| Component | Reused | New |
|---|---|---|
| Frozen bank (120 items, 120 worlds) | MRS-1, bound by hash | — |
| Integrity, create-only writes, supervisor, GPU lease | `model_quality_gpu_v1` | — |
| Native backend, resource guards, budget | MRS-1 `native.py`, `budget.py` | — |
| Statistics: `summarize`, `complementarity`, `factorial`, `paired_counts`, `cluster_ci` | MRS-1 `analysis.py` | — |
| Output contract + parser | — | `contracts.py`, `parsing.py` |
| Role prompts | MRS-1 wording | single-field projection |
| Runner, freeze, analysis entry | MRS-1 structure | `runner.py`, `freeze.py`, `analyze.py` |
| Constant baseline + role degeneracy | — | `analyze.py` |

MRS-2 deliberately preserves MRS-1's per-item record shape so that MRS-1's statistics apply
unchanged. That is why the 2 × 2 machinery needed no reimplementation.

## Tests

120 tests, all passing: 80 in MRS-FP-1 (`test_contracts.py`), 40 in MRS-2 (`test_mrs2.py`).
Synthetic fixtures only — these validate software, never efficacy.

Covering the Part 17 requirements: no gold enters any role prompt (asserted structurally —
the answer vocabulary legitimately appears in the system prompt, so a lexical scan would be
both false-positive and meaningless); selectors receive exactly `role`, `answer`,
`format_failure`, `terminated` and provably not raw text, extra parsed keys or gold;
independent roles cannot see earlier outputs; a candidate from another item is rejected;
malformed output stays visible and is never repaired; nontermination fails closed; the bank
is the MRS-1 bank by hash with no quarantined ID; evidence writes are create-only; resource
ceilings fail closed including on `NaN` and `None`; the smoke gate is proven to cover every
consumed role; and the future C/D 2 × 2 accepts dummy fixtures and computes every delta,
event mask and all 16 correctness patterns correctly.

Two of those tests were wrong on first write and were fixed, not worked around: the
label-blindness assertion tripped on its own answer vocabulary, and the 2 × 2 fixture was
missing hash bindings that `factorial` correctly demands.

## Execution receipts

| Run | Evidence | Qualification | Smoke | Generation | Wall | Peak GPU |
|---|---|---|---|---|---|---|
| MRS-FP-1 supa | `probe-supa-002` | — | — | 120 calls | 46.6 s | 2.37 GiB |
| MRS-FP-1 d1 | `probe-d1-001` | — | — | 120 calls | 41.6 s | 2.37 GiB |
| MRS-2 d1 | `mrs2-d1-001` | PASS | PASS 8/8 × 9 | 1322 calls | 444.5 s | 2.90 GiB |
| MRS-2 supa | `mrs2-supa-001` | PASS | PASS 8/8 × 9 | 1322 calls | 450.6 s | 2.90 GiB |

Zero optimizer steps, zero weight writes, zero retries, zero guards weakened, state digest
unchanged on every run. Every raw generation and per-item role output is preserved.

## Honest limitations

* **The result is at the floor.** Both candidates sit at or below the constant-answer
  baseline, so no arm contrast is interpretable as reasoning. This was registered as a
  reading *before* the run, not adopted after seeing it.
* **96 cases.** Small by design. Two-case differences carry intervals spanning zero.
* **No compute matching.** A1 is token-budget matched in name only; under a single-field
  contract the model stops after the answer regardless of allowance, and the run confirmed
  this at exactly 0 repairs and 0 damage.
* **S1 vs S2 is weaker than MRS-1 intended.** With one field there are no checker booleans
  to project, so the two selectors differ only in candidate source, not in projected detail.
* **Checker verdicts are indirect.** R2's sufficiency lives in its abstention and R3's
  validity in its recomputation. The MRS-1 `checker_errors` counters are structurally zero
  here, which is an absence of measurement and not evidence of reliable checking.
* **Two candidates, one seed each.** No training-seed variance is estimated.
* **Semantic overlap with protected populations is unverified.** No protected payload was
  opened.
* **Supa and D1 are not interchangeable** and are never pooled.

## Status

SOFTWARE PASS **true** · NATIVE EXECUTION PASS **true** · OPEN DEV RESULT **true** ·
MECHANISM SCOUT RESULT **true, negative** · CONFIRMATORY CLAIM **false** ·
RELEASE CLAIM **false**.
