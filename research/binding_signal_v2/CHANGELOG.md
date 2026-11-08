# Binding signal candidate v2

V2 is an explicit successor to the preserved candidate-v1 exclusion failure. It
starts from committed v1's structured oracle, grouped counterfactual generation,
independent policy-input semantic decoder and adversarial tests. BUILD.json binds
the parent code commit/blob and the unchanged failed candidate-v1 BUILD receipt.
No v1 files, rows, labels, findings or proposed numerical bands are modified.

TRAIN now uses a community sports club. DEV uses community craft/sewing circles.
The person, role and action pools and all four template families are rewritten;
the independent parsing grammars are separately rewritten to cover them. Person
questions retain natural **who/whose** wording. Role questions explicitly request
the role. Root seed is now23 and world IDs use the distinct `BINDV2` namespace.

The design still has64 TRAIN and32 DEV worlds, with all12 related views kept
together:768 TRAIN and384 DEV rows, four balanced answerability/control categories.
Every world still contains four occupied roles, one vacant role and one role
without an assignment, plus two listed people without assignments. Duty-target
counterfactuals and all independent semantic BUILD checks are retained.

The exact approved LC exclusion manifest is hash-pinned. A mandatory BUILD gate
checks **every complete generated user turn** against every listed proper-noun,
title, domain and task-type phrase. It uses the already-declared literal policy:
NFKC, casefold, punctuation/underscore to spaces, whitespace collapse and complete
phrase token boundaries. Generic entries such as Which, What, Name, Give, Work,
Case, State, According and Harbor receive no exception. A contact blocks output;
rows are never silently dropped, relabeled or regenerated to hide contacts.

Literal exclusion success is not complete external coverage. Exact LC8-gram
normalization/stopwords and approved reserved row/parent/entity/prompt indices
remain unavailable. No raw sealed prompts or answer keys are read. Output status
therefore remains `BUILD_ONLY_BLOCKED_EXTERNAL_COVERAGE`, even with zero literal
contacts. Open-bank exclusions must be repeated for these new bytes as well.

No tokenizer compilation, model call, training grant, inference budget, scientific
improvement or release is produced here. Licensing remains
`UNSPECIFIED_INTERNAL_RESEARCH`; human review is `NOT_PERFORMED`.

Build command, from the new code worktree:

```powershell
python research/binding_signal_v2/generate.py --out candidate-v2
```

Output is create-only under the designated binding-signal records root. The
default corpus is fixed; smaller scopes exist only for mechanical fixtures.
