# RQSFT-001 — balanced quality in Rineya 0.6B

New consumer experiment for the approved small verified-bank/SFT plan. Read
PROTOCOL.md and AGENTS.md. Existing MQ/DSD/RRP evidence remains untouched.

Implemented: owner-export admission reader, all200-row/full-history tokenizer
compiler, two masked objectives, supervised native stages, bounded CPU-moment
AdamW, checkpoint recovery, raw predictions, fixed blinded rubric, paired quality
analysis and immutable attempt accounting. Dataset owner is active; latest
observed state has zero admitted rows. No real bank, fresh model baseline, native
qualification, model improvement or training is asserted by this implementation.

Read [RUNBOOK.md](RUNBOOK.md) for exact commands and [STATUS.md](STATUS.md) for
current receipts and unresolved gates. The owner request is metadata only;
TEST_ONLY/EXCLUDED fixtures cannot become training data.

The first native screen has two trained cells at1e-5, final-checkpoint evaluation
only, all200 TRAIN rows once per arm,32updates/25k processed tokens maximum,
512 training/2048 evaluation context,64 generatedtokens percase. Same-source,
complete512 case coverage, raw termination and actual costs are mandatory.
No grid, paid service, source edit, automatic retry or release path exists.
