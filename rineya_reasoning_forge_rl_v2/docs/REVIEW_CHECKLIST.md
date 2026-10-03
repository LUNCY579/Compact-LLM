# Reviewer checklist and decision contract

**Release state: human review has not been performed.** Machine checks do not establish fluent, diverse, useful model-training data. The following protocol is proposed for the user before a substantial optimization run.

Review a TRAIN/validation sample stratified by eight families, four word tiers, four layouts, and applicable ANSWER/UNKNOWN/IMPOSSIBLE classes. Keep task families with class imbalance visible. Suggested initial scope: eight examples per family plus eight targeted long or edge cases (72 total). This number is a proposed review budget, not a completed receipt.

For each reviewed row, record its ID, source split, native token count, correctness decision, rules clarity, evidence sufficiency, trace-checkpoint accuracy, public-answer completeness, and any unresolved issue. The report must distinguish format problems, unsupported labels, native preparation problems, and usefulness concerns. Do not use labels such as "reviewed" or "approved" for rows only machine-checked.

Inspect whether task instructions unambiguously define time, status authority, preconditions, tie-breaking, and missing-evidence behavior. Recompute representative solutions manually or with another implementation. Check that unknown and impossible examples are not trivial formatting cues. Check whether long examples require meaningful context use rather than only filtering a scope.

Inspect the reasoning target as a task solution: correct initial state, every relevant transition, consistent carries, legal actions, valid optimization witness, and concise final object. The trace is not expected to include a full exhaustive search transcript. A prose claim about exhaustive impossibility is supported by the oracle execution, not a separately supplied formal proof tree.

Inspect actual native SFT batches: exact template, prompt-prefix parity, response mask, EOS/turn terminator, no padding loss, and no silent truncation. Inspect actual RL batches: no gold target or metadata in the model input, correct ID joins, proper grouped sample order, generation reserve, and a verified truncation policy.

Freeze a review receipt with model/tokenizer identifiers and hashes, selected row IDs, excluded row IDs/reasons, unresolved defects, and the chosen experimental split contract. Keep changes on a new research branch when integrating with the user's repositories; this standalone artifact does not modify any repository.

Suggested machine-readable fields appear in `configs/review_receipt.template.json`. Its null values intentionally leave native execution and human approval closed.
