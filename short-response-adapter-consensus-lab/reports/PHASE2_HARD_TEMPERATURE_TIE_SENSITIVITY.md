# Hard-vote temperature invariance at the tie boundary

Status: **DEV_HARD_TEMPERATURE_TIE_SENSITIVITY_COMPLETE**

Positive temperature scaling preserves each expert's top-1 proposal and all plurality vote counts. The deployed selected token is therefore invariant whenever the plurality winner is unique. Its confidence-sum tie-break is a narrower exception.

- Plurality-tied positions: **1,097/18,760 (5.85%)**.
- Every shared-temperature intervention leaves every hard decision unchanged.
- The fitted per-expert temperature vector changes **259** local decisions, with token/example accuracy 78.18%/24.54% versus raw 78.33%/25.26%.
- Largest single-expert grid effect: **grounding at T=0.25**, changing 513 token decisions (46.76% of tied positions).

This qualifies the invariance claim: votes and unique-plurality winners are exactly scale-invariant, while confidence-based resolution among tied vote leaders is not invariant to expert-specific scaling. The counterfactual is DEV-local and does not establish a free-running accuracy effect.

Frozen cache SHA-256: `4b192a736451e81fd9e3dcd212f3f3aa97cda0a67c440536802453187377ee13`.
