# Preregistration Template

## Hypothesis

State the exact compact-state sufficiency claim and the model family to which it applies.

## Frozen artifacts

- base model path/revision/hash:
- tokenizer hash:
- dataset archive SHA-256:
- training config hashes:
- evaluation config hashes:
- code commit/release hash:

## Primary arms

- FULL adapter:
- SPLIT adapter:
- seeds:
- matched-update budget:
- adapter architecture:

## Primary outcome

- baseline protocol: `direct_solve`
- candidate protocol: `generated_rollout`
- metric: `score.correct_strict`
- non-inferiority margin:
- bootstrap confidence:

## Secondary outcomes

List oracle transitions, ablations, direct SPLIT performance, error-propagation gap, and subgroup analyses.

## Exclusions

Declare only technical invalidity exclusions. Model failures and parser failures remain in the denominator.

## TEST access

Record the date and reason TEST was unlocked, and affirm that no further tuning occurred afterward.
