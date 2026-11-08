# Development verification notes

Session date: 2026-09-12. No official-source model execution occurred here.

An initial development run had 67 passing tests and two failed assertions.
One test demanded bitwise equality across differently shaped full/tail output
matrix multiplies; the protocol already specified elementwise atol=rtol=1e-4 for
such paths. That assertion was corrected to use the existing rule, without
changing the gate, and tail-path comparisons were added to the active check grid.
Same-path source/augmentation initialization identity remains bitwise exact.

The other test incorrectly assumed a seeded random permutation must differ from
the identity permutation. Seed 17 legitimately leaves a three-element list in
identity order. The test now uses eight examples and checks the exact expected
seeded order. It does not manufacture a different order or change the seed.

Additional checks cover invalid objective shapes/weights and the fixed length
bound. The final scoped suite result is recorded in LOCAL_VERIFICATION.json and
pytest.txt. These are tiny/random mechanical tests, not native, official-weight,
capability, performance, or all-repository test results. No independent external
adversarial-review claim is made.

During the final parent-commit review, the newer recoverability-frontier result
was found alongside older continuation narratives. RESEARCH_PLAN.md records that
partial sharing can recover under the tested budget, and does not treat repeated
60-prompt evaluations as 180 independent questions or CI overlap as equivalence.

Only new files under full_width_transition_v1 are included in the publication.
The legacy dependency snapshots used locally match their pinned Git blob hashes.
