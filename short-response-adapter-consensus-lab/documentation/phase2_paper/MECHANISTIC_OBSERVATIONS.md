# Mechanistic observations

## 1. Policy composition drives the individual-adapter gain

Hard voting is worse than Track A on answerable v2.2 TEST cases but much better
on abstention cases. Its aggregate advantage is therefore a composed policy
tradeoff rather than uniform reasoning improvement.

## 2. Hard consensus limits confidence-scale dominance

Every expert receives one top-1 vote. A high-confidence dissenter cannot defeat
a unique plurality merely by being sharper. This property matters because the
seven adapters have substantially different calibration and entropy profiles.
The confidence tie-break is the measured boundary: unequal rescaling can alter
decisions only when vote leaders tie.

## 3. Calibration mismatch contributes but is insufficient

Per-expert calibration improves soft fusion and beats hard on same-split DEV.
It does not close the hard gap on v2.2 TEST. Calibration is therefore part of
the mechanism boundary, not a sufficient explanation for the cornerstone.

## 4. Early divergence is not trajectory correction

Hard and soft sequences often diverge at the first token, but the DEV
teacher-forced method gaps are almost identical to the free-running gaps. Early
choice matters descriptively; the predicted hard-specific corrective advantage
does not appear on the adjudicating split.

## 5. Committee composition dominates size

Both cached all-subset analysis and selected free-running systems are
nonmonotonic in K. The selected K=4 committee outperforms K=7. Grounding is
harmful in the deployed K=7 context, Procedure is neutral, and Social changes
sign between cached Shapley and free-running removal analyses.

## 6. Abstention is valuable but changes utility

The abstention specialist has positive overall marginal value and a very large
positive abstention-class contribution, while reducing answerable accuracy.
Adaptive consultation exposes the same tension: similar aggregate accuracy can
hide a materially different refusal/answer policy.

## 7. The effect is benchmark-specific

Hard K=7 wins the v2.2 cornerstone but loses sharply to raw probability on the
fixed-method v2.1.2 transfer. The paper should claim a demonstrated composition
mechanism under stated conditions, not universal ensemble superiority.
