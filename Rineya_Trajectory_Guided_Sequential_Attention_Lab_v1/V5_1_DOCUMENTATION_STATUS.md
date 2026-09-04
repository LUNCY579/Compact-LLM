# TGSA v5.1 Documentation Status

This index prevents historical documents from being mistaken for the current
operator contract. V5.1 intentionally preserves prior preregistrations and
results verbatim where changing them would rewrite history.

## Current canonical documents

| Document | Status | Authority |
|---|---|---|
| `README.md` | Current | Programme status and claim ceiling |
| `V5_1_UNIFIED_PROGRAMME.md` | Current | Comparative audit and final v5.1 scientific design |
| `TGSA_MTP_CANONICAL_PATH.md` | Current | Stage identities, lineage, and promotion graph |
| `PHASE2_TGSA_MTP_RUNBOOK.md` | Current | Operator commands, stop rules, and evidence preservation |
| `configs/v5_1_programme_contract.json` | Frozen/current | Machine-readable registration and evidence boundary |

When a historical or inherited document conflicts with one of these on what may
be run or claimed after September 2, 2026, the v5.1 contract governs.

## Inherited technical documents

| Document | Status under v5.1 |
|---|---|
| `PHASE1_V5_PREREGISTRATION.md` | Preserved design rationale, plus dated amendments. Its former “draft/not registered” state is superseded prospectively by the exact-byte v5.1 programme lock. The original text is not rewritten, but §7 now carries an amendment dated 2026-09-04 registering `every_seed_beats_its_family_null`, added before any nondevelopment run. A check enforced in code and registered nowhere is an unregistered gate, which `test_the_prereg_gate_table_lists_every_check` refuses, so a code-side strengthening must be recorded here. |
| `PHASE2_TGSA_MTP_DESIGN.md` | Preserved Phase 2A mechanism-bridge design. Gradient-mode order and stage promotion are governed by v5.1. |
| `PHASE2_V4_REGIME_INTEGRATION.md` | Historical explanation of the rejected v4 boundary; cannot authorize new runs. |
| `ARCHITECTURE.md` | Current architectural reference, subject to the v5.1 evidence boundary. |
| `CLAIMS_AND_FALSIFIERS.md` | Current general claim ladder; v5.1 adds stage-specific ceilings. |
| `EVIDENCE_LEDGER.md` | Historical ledger until a new immutable v5.1 result is committed. |

## Frozen historical evidence

The following are records of what was preregistered or observed at the time.
They must not be edited merely to match current plans:

- `PHASE1_V3_PREREGISTRATION.md`
- `PHASE1_V3_RESULT.md`
- `PHASE1_V3_COMPETENCE_POSTMORTEM.md`
- `PHASE1_V4_PREREGISTRATION.md`
- `PHASE1_V4_PILOT_REPORT.md`
- `PHASE1_V4_COMPETENCE_RESULT.md`
- `PHASE1_V4_RUNBOOK.md`

V3 was inconclusive because the arms were not competent. V4 rejected its
registered regime family. Neither result identifies the TGSA mechanism in
either direction.

## Externally pinned programme sources

Two important implementations remain on separate source branches and are pinned
by commit in the v5.1 contract:

- Phase 2B exact-decoding system factorial:
  `research/tgsa-mtp-rineya-complementarity-v1` at `49aa961...`;
- pretrained Qwen3/Rineya retrofit safeguards:
  `research/tgsa-rineya-pretrained-retrofit` at `ef5194b...`.

`PHASE2_IMPLEMENTATION_SUMMARY.md` belongs to that first group and does **not** exist on this branch. It lives only on `research/tgsa-mtp-rineya-complementarity-v1` at `49aa961`, and the programme ledger imports a verbatim copy under `TGSA_PROGRAMME/imported/phase2b/`. It is a historical implementation summary, not a result. It was previously listed above among this branch's inherited documents, where every other entry does resolve.

Their source documentation remains authoritative for low-level implementation,
while v5.1 governs cross-stage promotion and claim interpretation.

## Update policy

A document is “up to date” in one of two ways:

1. it is current and edited with the v5.1 branch; or
2. it is historical and explicitly classified here so its original record is
   preserved without being treated as current authorization.

Any later scientific amendment must update this index, the programme contract,
the relevant runbook, schemas, tests, and README in the same commit.
