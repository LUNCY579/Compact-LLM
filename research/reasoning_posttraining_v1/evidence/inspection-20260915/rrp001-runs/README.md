# RRP-001 executed run receipts

Three supervised launches, each with its own terminal supervisor receipt:
`qualify-001` (zero-update qualification), `train-001` (the 32-world microfit) and
`resume-001` (fresh-process resume verification).

`budget-journal/` is the hash-chained, create-only all-attempt budget journal:
346 sealed events plus the grant they are bound to.

Checkpoint tensors are deliberately NOT copied here. The three checkpoints total about
11.3 GiB and live only under
`rineya-reasoning-posttraining-records/rrp001-journal/runs/*/microfit/checkpoint-*/`.

`verify_resume.py` is the resume comparator. It lives outside the package on purpose:
the binding that `validate_checkpoint` enforces includes the RRP package code hash, so
adding a file to the package between the candidate and resumed attempts would have
invalidated the resume it verifies (see `CLAIM_LEDGER.json` CL-013). It should be
promoted into the package for the next experiment.
