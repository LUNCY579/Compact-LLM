# FUNCTIONAL-ADDITIVITY-001 — Amendment 1 (POST-HOC) and audit corrections

Date: 2026-09-23. **Post-hoc.** The plan (`PLAN.md`) was written before the re-score, but after
the registered run and its audit. Nothing here replaces the registered primary, which stands:
κ = −0.06395, CI [−0.0740, −0.0545], **ADDITIVE** under the registered ±0.10 rule.

Evidence:
- `TF_EXACT.json` (sha `4d1dcca5…`): five arms re-scored under the lease, 60 GPU-seconds. The
  canary against `TF_RAW.json` is **bitwise**: max |Δ clipped log-odds| = 0.0 and 0 strict-flag
  mismatches on every arm.
- `AMENDMENT_1.json` (sha `ff6ba84d…`), produced by `amend_analysis.py`.
- `AUDIT_2026-09-23.json` (sha `ef2a6b44…`): the five-audit review with its claims and verdicts.

## 1. The ADDITIVE label is specific to the registered clip

PREREG claimed the ±10 clip only trims irrelevant tails. **That premise was false.** The clip
caps 21.2% of SPARSE's content positions but only 7.0% of ADD-100's, which pulls I toward 0.
With exact log-odds:

| clip | κ | 95% CI | label the registered rule would give |
|---|---:|---|---|
| 5 | +0.0602 | [+0.0393, +0.0793] | ADDITIVE |
| 8 | −0.0197 | [−0.0329, −0.0075] | ADDITIVE |
| **10 (registered)** | **−0.0640** | [−0.0743, −0.0547] | **ADDITIVE** |
| 12 | −0.0986 | [−0.1078, −0.0904] | UNRESOLVED |
| 15 / 20 / none | **−0.1038** | [−0.1130, −0.0957] | SUB-ADDITIVE |

κ moves monotonically with the clip, and the registered value sits on the steep part of that
curve. **Unclipped, the content interaction is about −10% of Titan-Sparse's own content effect.**
I < 0 on 167/192 rows at clip 10.

## 2. Secondary (b) was implemented wrongly, and as registered it is SUB-ADDITIVE

PREREG line 71 defines (b) as `Z_content` restricted to decision **positions**. `analyse.py`
applied the rule per **row**. That keeps 190/192 rows, so it just recomputed the primary, and
RESULTS.md then reported it as "also ADDITIVE". Implemented as registered (1,227 of 2,274
content positions):

- clip 10: κ = **−0.1185**, CI [−0.1295, −0.1089], **SUB-ADDITIVE**
- unclipped: κ = −0.1235, CI [−0.1343, −0.1141], SUB-ADDITIVE

Three independent computations agree on these values: two audits and this script.

## 3. Per family

| family | n | κ at clip 10 | κ unclipped | mean g_S (unclipped) |
|---|---:|---:|---:|---:|
| ledger_replay | 64 | −0.036 | −0.081 [−0.092, −0.070] | 76.7 |
| register_machine | 64 | −0.052 | −0.093 [−0.102, −0.084] | 71.3 |
| **prerequisite_plan** | 64 | **−0.469** | **−0.469** [−0.519, −0.430] | 7.0 |

prerequisite_plan's content is only 0–2 leading-digit tokens. Its interaction is −2.1 nats of
log-odds per content token, 9–16× the other two families. Its small denominator is part of the
story, not all of it.

## 4. Registered secondary 7 (period propensity), now computable

The site is the final gold position on the 128 `}` rows (merged `}.`), and the EOS position on
the 64 integer rows (`.`).

| arm | mean log p of the period variant | rows where it beats the gold continuation |
|---|---:|---:|
| BASE | −10.10 | 1/192 |
| SPARSE | −10.04 | 0/192 |
| OPALLO | −1.65 | 68/192 |
| ADD-O75 | −1.33 | 68/192 |
| **ADD-100** | **−0.34** | **182/192** |

This matches the recorded generation, where ADD-100 produced 178/192 trailing periods.

## 5. Other audit corrections to RESULTS.md

- The registered closing consistency expectations hold but were not reported. Z_closing(ADD-100)
  < Z_closing(SPARSE) on 192/192 rows (≥150 was required), and the mean of
  Z_closing(OPALLO) − Z_closing(BASE) is −8.77 (required < 0). Secondary 8 matches exactly except
  OPALLO, 9 vs 8 (ref `12`, gen `128.`).
- "Closing: SUB-ADDITIVE" is an outcome label for a readout that had no registered outcome rule.
  It also depends on the scale: unclipped log-prob I_closing is +0.029, CI [−0.22, +0.30].
- "The strong negative is confined to the closing channel, not the content skill channel"
  overstates the result. Content carries a real ~6–10% interference, and −47% in one family.

## Corrected reading

Adding dOpallo to Titan-Sparse at full strength is **approximately additive on transition
content**. It carries a **small but real sub-additive interaction**: about 6% at the registered
clip, about 10% on exact log-odds, and concentrated in prerequisite_plan. It is **strongly
non-additive at the closing position**, where Opallo's period habit takes over. The registered
label, ADDITIVE, is correctly computed. The analysis in §1 shows it is not robust to the
registered readout's clip choice.
