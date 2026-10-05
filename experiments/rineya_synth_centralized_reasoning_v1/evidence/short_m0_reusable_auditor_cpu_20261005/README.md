# Fixed raw M0 terminal auditor

`auditor.py` adapts the actual successful batch01 audit code preserved verbatim in
`batch01_executed_audit_code.py.txt`. It accepts only the current first collector
contract, one independently specified preregistered64-row batch, successful owned
terminal state, exact config-file and preservation-body SHA256 pins, and the root's
explicit `ROOT_TERMINAL_AND_PRESERVED` signal. Without that signal it performs no
heavy reads or evidence write. An output is optional and, when supplied, must be
a new owned evidence JSON named for its raw leaf. It never launches a worker,
imports Torch/Transformers, restores a model, grades correctness, enrolls rows,
opens protected targets, or edits native sources/gates.

The CPU audit hashes original/preserved raw files,83 original/snapshot sources,
64 runtime files,17 decoder files,5 public data files and7 model asset files;
recomputes every token prefix, exact finish and cumulative cost; decodes only
generated IDs with the pinned CPU tokenizer; and checks base/zero-B/resource/
deadline/owned cleanup closure. API and resource counts are derived from actual
complete receipts, with fixed registered caps retained. Internal ATen operations
remain outside the registered API-boundary ledger. Prior numerical qualification
does not become a per-row numerical replay or correctness claim.

API: `audit_terminal(config_path, raw_directory, preserved_directory, *,
expected_selector=("atomic_RS_DEV", 2), expected_config_file_sha256=...,
expected_preservation_body_sha256=..., terminal_preservation_signal=SIGNAL,
output_path=None)`. The CLI exposes the same arguments and explicit signal flag.
Run only after the root's terminal/preservation signal and outside an isolated
host measurement. `test_auditor_controls.py` uses light mock header controls and
isolated CLI denial only; these tests do not execute a heavy audit.
