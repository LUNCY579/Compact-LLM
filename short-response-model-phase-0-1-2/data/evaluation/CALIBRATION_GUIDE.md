# Quality Evaluator v3 calibration packet

This packet separates policy regression examples from future human calibration.
`policy_expectation` records the approved behavioral contract; `human_rank` is
left `PENDING` so the repository does not fabricate human judgments.

Reviewers should rank replies within and across groups according to functional
adequacy, comprehensibility, logic/context, and safety. Minor grammar, slang,
ellipsis, and natural follow-up questions must not be downgraded merely for not
matching textbook English or one reference answer.

After human rankings are collected, adjust only the versioned proportions in
`configs/quality_evaluation.json` and
`configs/relative_opportunity_evaluation.json`, rerun the sensitivity report,
and increment the policy version when rankings materially change.
