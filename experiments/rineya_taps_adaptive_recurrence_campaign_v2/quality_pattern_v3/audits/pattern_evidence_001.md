# Existing-evidence patterns: selector headroom is unmeasured policy potential

No model, tests, data/source/state change, fitted candidate or new alpha rollout. All results here are descriptive joins of the exposed288-case generation run and192-case CPU canonical-prefix study.

| Split | Intact K0 | Unit K1 | K1 gains / losses vs K0 | Retained K0 | Perfect retrospective K0/K1 choice | Above unit |
|---|---:|---:|---:|---:|---:|---:|
| DEV192/96parents |63|69|12cases/11parents gained;6/6 lost|57/63|75|6cases/6parents,3.125pp|
| Exposed VAL96/48parents |33|34|6cases/6parents gained;5/5 lost|28/33|39|5cases/5parents,5.208pp|

The six-constant oracle70/35 is a different action set and does not bound K0/K1 selection. A perfect selector would keep unit-only gains and recover intact-only losses; its gold-graded ceiling is not an implemented/validated policy. Both arms still fail117DEV and57VAL cases.

DEV intact-only successes: deduction3, program state2, document-CONFLICT1. VAL intact-only: deduction3, document-CONFLICT1 and MISSING1. Unit gains include source-grounded numeric answers replacing UNKNOWN; a blanket abstention-preservation rule risks discarding them. A0 answers all48/24 unanswerable cases correctly, while unit loses1DEV conflict and2VAL conflict/missing cases. These logged labels are forbidden online features.

Six DEV intact-only cases favor local scale decrease;11/12 unit-only gains favor increase. This promising descriptive association uses canonical targets/teacher-prefix losses. No online rule can read it. The frozen input-only balance is positive in all192 cases and its ranges overlap between both preference groups; progress/fluctuation also overlap. Thirty-four/96 parents have opposite reliable variant directions. Sign inversion agrees97/188 local directions versus91/188 increase, but is not an accuracy intervention and still harms some variant in63parents. Existing fixed.8/.9 score below unit on both splits.

First-token changes occur on4/6 DEV intact-only and11/12 unit-only cases; on exposedVAL, all5 intact-only and all6 unit-only change. They also occur on10DEV/7VAL cases where both arms are wrong. Nongold KL, margin-change and logit ranges overlap; no threshold was selected. Two DEV intact-only losses have identical first-token top1 despite different final correctness. These diagnostics do not certify which response is right.

A deterministic K0/1 guard is a proposal requiring new prospective generation measurement. Allowed inputs are actual prompt/tokens and target-independent model quantities available before scores; never gold, logged source_status/answerability/category/family/split/IDs, canonical teacher tokens, losses/gradients or oracle decisions. An explicit approved-evidence consistency checker derived from prompt text could target abstention regressions, but on this bank it is a symbolic/stateless comparator susceptible to template shortcuts and must be compared separately. No such checker or guard was fitted, implemented or graded.

Next measurement: freeze one rule and qualify it; run actual own-trajectory generations with intact/unit/stateless controls and charged guard overhead; retain strict gains/losses and preservation against both baselines; keep format/EOS/UNKNOWN separate. Use genuinely unexposed source/author/generator holdouts, parent-group separation, renamed/reworded/reordered evidence and query roles, positive controls and strong query-blind floors. ExistingVAL cannot become unexposed after looking at these outcomes. CPU local derivative and GPU generation remain separate measurements.

The zero-training/protected scope is preserved. The evidence supports exploring deterministic selection, not claiming quality gain, guaranteed preservation, source-general reasoning or speed. All29 discordant case records across28 parents, parent/family/status counts and exact input bindings are in JSON.

The actual K0/K1 gain and loss cases occupy disjoint parent sets (17DEV,11VAL discordant parents); no present parent has one variant favoring K0 and another favoring K1 by strict correctness. This differs from34/96 opposite local-scale CPU directions and cannot justify using parent IDs or memorized case lookups online.
