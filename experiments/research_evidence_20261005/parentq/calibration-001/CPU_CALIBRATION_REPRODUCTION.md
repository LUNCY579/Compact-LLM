# Offline synthetic PARENTQ calibration reproduction

No model, native outcome or capability evidence. Legacy draft remains interpretable.

| Legacy scenario | Predictive P | Status |
|---|---:|---|
| perfect_parent_fixture | 0.874843129 | PARENTS_QUALIFIED |
| legacy_middling_shuffled_index_mod3 | 0.667757598 | PARENTS_MARGINAL |
| legacy_half_shuffled_index_mod2 | 0.365668832 | PARENTS_NOT_QUALIFIED:H |
| H_never_abstains | 0.075695851 | PARENTS_NOT_QUALIFIED:H |
| H_always_abstains_BASE_correct | 0.013088516 | PARENTS_NOT_QUALIFIED:H |
| T_BASE_both_wrong | 0.049578262 | PARENTS_NOT_QUALIFIED:T |
| BASE_ceiling_both_parents_correct | 0.024270037 | PARENTS_NOT_QUALIFIED:H |

| Equal BASE/H toy accuracy | H gate pass |
|---:|---:|
| 0.15 | 0.180396064 |
| 0.25 | 0.244903564 |
| 0.50 | 0.335937500 |
| 0.60 | 0.347750400 |
| 0.75 | 0.335723877 |
| 0.90 | 0.241033500 |

Declared toy true combined pass: 0.817152000.
Across 20000 synthetic banks: qualified 0.2750, marginal 0.7218, not qualified 0.0032.

Conditional bank signs: 6 positive, 0 negative; screen value -6.
T14/T2 operative collision: ('ledger_replay', 'scale', 19, 3, 57) == ('ledger_replay', 'scale', 19, 3, 57).

Legacy blanket-abstention label: NATIVE_PROMPT_ELICITS_H_ABSTENTION; H_OFF answerable abstained 12/12.

All choices remain prospective and unfrozen; see PROSPECTIVE_DESIGN_OPTIONS.md.
