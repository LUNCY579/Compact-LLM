# First quality cycle, 30 September 2026

The end-to-end plan, long-context curriculum, retention design and every-row
supervision workflow are implemented in a separate dataset workspace. The
centralized experiment and sibling datasets received no writes; no model/GPU,
paid-teacher, training or protected-evaluation work ran.

The pilot has72 worlds/48 TRAIN candidates across six objectively gradable task
families. Independent verification passed every row and432 applicable mutation
checks. Two agents reviewed each public problem with target/specification hidden;
their72 answers per role agree with the independent verifier. Adversarial review
checked2,075 records and240 applicable controls. Exact native-tokenizer measurements
span278-4429 total tokens;24 rows fit512 by length. Dataset readiness and model
eligibility remain distinct; training_ready is false.

The review process improved the pipeline rather than merely voting PASS. Before
generation it removed duplicate active worlds and length/outcome confounds, and
corrected an archive-payload verifier overconstraint. Controller fault injection
then exposed self-review aliases, invalid token ledger values, forged
tokenizer/readiness metadata and unsafe role-derived path access. A second review
found numeric reviewer identities and Boolean/integer template aliases. Corrections
created new producer versions and retained failures.39 unit tests now pass.
The final controller review passed15 cases: the valid baseline passed and14
invalid fixtures were rejected.16 genuine binding/export checks also passed,
including byte-exact reproduction of the TRAIN export, lineage and release manifest.

The current genuine release is pilot-v3. All three pilot versions contain exactly
the same72 canonical problems; canonical SHA256:
`227ebb2fcc5c9f549643441ebfdf4682786c5b535e76780e284bd32ff6271102`.
Candidate SFT SHA256:
`b206bc4ebe5e6837613af95b89cf7118981f36869cca9ea511207fca5a9b17ba`.
Original review reuse requires exact row/packet/assignment byte equality and is
documented. Producer/source snapshots preserve reproducibility across edits.

## Next quality priorities

1. Establish a durable source/semantic-parent/structural registry across future
   batches, and safely audit authorized legacy TRAIN provenance/exclusion inventories.
2. Build a fresh, small mechanism pilot requiring many relevant distributed facts,
   deepening the reasoning horizon independently of irrelevant context length.
3. Add realistic documents with verified rights and source-grounded reference
   review, plus genuinely unfamiliar structural templates and wording.
4. Calibrate reviewer-control banks with known defects before broader prose release.
5. Coordinate dataset admission and preservation enrollment only after the active
   experiment owner provides qualified native capture/training context and budgets.

There is no measured student gain or forgetting result. The scheduled quality
supervisor is active every six hours in this chat and prioritizes quality over
volume. It works in bounded locked cycles, keeps failures, uses separated
subagents and stays quiet while no meaningful change occurs. No independent
confirmation or model promotion follows from this research release.
