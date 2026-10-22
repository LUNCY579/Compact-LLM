# Prospective storage amendment: sequential exports

No DOSELORA model has yet run and no new scientific endpoint exists. Transparent
compression verified all664 completed DOSEINT model/checkpoint files against
their existing hashes. Every logical byte and path was preserved. Physical free
space reached about18.2GiB, below the initial19.5GiB two-export reserve.

Use sequential exports with a physical-space gate at each step. Require17GiB
free before training, which covers the256MiB native checkpoint ceiling, one full
2.4GB HF export, and the unchanged14GiB floor. After the LoRA HF export is fully
verified, compress that new immutable weight file transparently and verify its
receipt hash again. Require at least16.3GiB free BEFORE building the second full
HF norm control; its existing builder independently requires outputweight bytes
plus14GiB. Compress the verified norm-control file afterward, again hash-verifying.

If the first export does not compress enough to pass the second-export gate,
record BLOCKED_ON_STORAGE and stop before that export. Do not remove evidence,
change model dtype, lower the14GiB floor, or score an unmatched LoRA arm. This
amendment changes storage scheduling, not model/data/optimizer/bands or budgets.
The new-records logical disk ceiling remains7GiB; compression never discounts
logical-byte accounting. SESSION_ENVELOPE_V2.json supersedes only the prospective
pretraining reserve, preserving the original start/deadline and all other limits.
