# RRP-015 reserved retry: evaluation monitoring dependency

Both planned training launches completed (control 28 updates, rehearsal 55), with
native qualification PASS and no budget breach. The first Supa evaluation attempt
stopped in the wrapper's resource preflight before loading any model: the legacy
RAB Python environment lacked `psutil`.

The failed attempt is retained at
`rrp015-20260917/eval-rrp015_supa-process/`, including the traceback, claimed worker,
failed terminal and supervisor receipt. Installed only `psutil==7.2.2` into that
environment; Torch 2.10.0+cu126 and Transformers 4.51.3 remain unchanged. No scoring,
prompt, generation or resource threshold was modified.

The single reserved retry is assigned to
`eval-rrp015_supa-process-retry001`. There are no training retries. The aggregate
ceiling remains 11 attempts (2 training + 8 planned evaluations + 1 reserved retry)
and 8,400 seconds; this explicitly resolves the original registration's ambiguous
placement of the reserved retry beside the training launches. No second retry is
allowed. Count the failed pre-model attempt in the session ledger rather than
erasing it or reporting only successful launches. All eight completed scientific
evaluation outputs remain required before calling this pilot fully measured.
