The two root V2 test-portability changes pass independent review. Both selected
methods pass in the targeted CPU run, with no skips, unchanged production/test
bytes before and after, and no Torch/Transformers/Tokenizers imports.

`Controls.test_native_entry_remains_denied_before_IO` now explicitly exercises
`STAGED=True` regardless of where the test module lives. Its `_terminal_files`
trap proves native dispatch rejects before terminal I/O. The restored production
implementation-availability assertions remain valid for both root and stage.

`FilesystemControls.test_identity_probe_error_is_not_dead_and_fixture_cannot_native_via_alias`
retains its injected identity-permission-error rejection, then independently
checks the physical staging filename with `STAGED=False`. This preserves the
guard against rebranding staged code through a patched module flag; its terminal
I/O trap also remains active.

The independent runner additionally replaced the real identity, asset/model/
runtime and checkout probes with assertion traps. Each recorded zero calls.
No native model, GPU, encoder, decoder, binary-asset probe or live OS identity
probe was executed, and no native PASS or qualification was issued.

Production validator SHA256 remains
`afca22e2a6f99dd5cb4485375392d8d3181951921c278ce7c5053866a718566e`;
production driver SHA256 remains
`164ea401617ef459da9d5b346403b9ed09d1c2c378d3ba503a3eb9fff0d0f1db`.
Receipt: `root_v2_test_portability_review_1791213772338376800.json`.
No root source file was edited by this reviewer. No broader or native-study
conclusion follows from these two portability controls.
