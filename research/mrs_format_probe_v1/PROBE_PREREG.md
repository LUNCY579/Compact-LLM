# MRS-FP-1 — output-contract probe preregistration

**Status: prospective. Written and committed before any probe generation.**
**Classification: OPEN / DEVELOPMENT / SELECTION-EXPOSED FOREVER. Never confirmation.**

## Why this probe exists

MRS-1 executed natively for the first time on 2026-09-17. Both registered candidates
(`d1-seed17-bf16-export`, `supa-1.7-frozen`) passed `NATIVE_QUALIFICATION` and then failed
the prospectively frozen `SCHEMA_SMOKE` gate at **0/8 valid on every role**. The cause is
measured, not inferred: the models emit a bare in-vocabulary answer token (`3`, `0`,
`UNKNOWN`) and terminate after two tokens, instead of the required JSON object. The
rendered chat template was verified to deliver the MRS-1 system prompt verbatim with
thinking disabled, so this is a family instruction-following limit, not an instrument
defect.

MRS-1 forbids prompt repair (`schema smoke failed; no prompt repair or CHECK evaluation`)
and its `MULTI_REASONING_POSTTRAIN_CONTRACT.md` requires that any necessary incompatibility
create a **new experiment, not a silent amendment**. MRS-1 is therefore closed at cells
A and B NOT PRODUCED, and this separate probe determines whether a successor instrument is
viable at all.

This probe answers exactly one question:

> Which output contract, if any, can this model family satisfy reliably enough to carry a
> multi-reasoning protocol?

It does **not** measure reasoning quality, and no result of it may be reported as an
answer to the multi-reasoning research question.

## Population

The **8 permanently-open MRS-1 SMOKE items only**, read from the frozen bank
`research/multi_reasoning_scout_v1/bank/public.jsonl`, selected by `split == "SMOKE"`.

* CAL and CHECK are never read by this probe; the runner raises if any non-SMOKE item enters.
* These 8 items were already exposed by the MRS-1 smoke runs and are already selection-exposed.
  Using them spends no unexposed material.
* The MRS-1 freeze (`278ce0962bd10864fba7d8ce2e7248579f18b22495d176c4f0f40db903a68837`) and the
  bank bytes are **not modified**. The scout bank is not regenerated.

## Candidate contracts

Five contracts, all label-blind, all reusing the MRS-1 role instructions verbatim so that the
probe varies the **output contract** and nothing else:

| ID | Form | Prefill | Carries checker boolean |
|---|---|---|---|
| `C0_JSON_MRS1` | MRS-1 JSON object (control) | none | yes |
| `C1_BARE` | bare answer token | none | no |
| `C2_LINE` | `ANSWER: <x>` (+ `SUFFICIENT:`/`VALID:` line for checkers) | none | yes |
| `C3_JSON_PREFILL` | MRS-1 JSON object | `{"answer": "` | yes |
| `C4_LINE_PREFILL` | line form | `ANSWER:` | yes |

A prefilled assistant opening is a fixed, deployable, label-blind decoding procedure: the same
bytes are prepended for every item, role and model. It carries no item-specific information.

`C0_JSON_MRS1` is carried verbatim as the control and is expected to fail; it establishes that
the probe reproduces the MRS-1 result under the probe's own harness.

Roles probed: `R0` (direct solver), `R2` (evidence/premise checker), `R3` (consistency checker).
These are the three gated roles whose structured fields the protocol actually depends on.

## Primary metric and selection rule — registered before execution

**Primary metric: strict format compliance** for a (contract, role) cell =
fraction of the 8 items where the output both `terminated` and parsed under that contract's
strict parser, with `answer` in `("0","1","2","3","UNKNOWN")` and any required checker boolean
present and genuinely boolean. No substring extraction, no fence stripping, no coercion,
no label-based repair.

**Eligibility:** a contract is eligible only if compliance is `>= 0.75` on **every** gated role
for **both** models. 0.75 is the MRS-1 smoke-gate threshold, carried deliberately rather than
newly chosen.

**Selection:** among eligible contracts, select the one with the highest mean compliance across
roles and models. Ties break toward (1) carrying the checker boolean, then (2) no prefill, then
(3) contract ID order as listed above.

**Answer correctness is recorded but is FORBIDDEN as a selection input.** Selecting an output
format on accuracy would tune the instrument on outcomes. Correctness is computed only after
generation, reported separately, and labelled descriptive.

If no contract is eligible, the registered outcome is that **no successor instrument is
viable for this family at this scale**, and that negative result is reported as-is. It is not
grounds for widening the battery until something passes.

## Bounds

Inference only. Zero optimizer steps. Zero weight modification; the model state digest is
recomputed after generation and must equal the pre-inference digest.

5 contracts x 3 roles x 8 items = **120 generation calls per model**, output cap 64 tokens,
so at most 7,680 generated tokens per model, batch size 1, one model load per model, serial.
Reuses the MRS-1 preload guards unchanged (6 GiB host, 6 GiB GPU free, 8 GiB disk; 1.5 GiB
host and 0.5 GiB GPU while running; 1800 s hard wall-time per model). Guards are not weakened.
One create-only `PROBE_LAUNCH_<model>.json` token per model; no silent retry.

Evidence is create-only under `research/mrs_format_probe_v1/evidence/`. Raw generations,
malformed outputs and parse errors are preserved verbatim; malformed output is never
silently repaired.

## What this probe cannot establish

It cannot establish that multi-reasoning helps, that any role is complementary, or that any
selector works. It measures format compliance on 8 open items. A passing contract licenses
**building and freezing a successor instrument**, nothing more. The successor must then be
frozen prospectively and run on CAL/CHECK before any efficacy statement exists.

SOFTWARE PASS, NATIVE EXECUTION PASS, OPEN DEV RESULT, MECHANISM SCOUT RESULT, CONFIRMATORY
CLAIM and RELEASE CLAIM remain separate. The last two remain **false**.
