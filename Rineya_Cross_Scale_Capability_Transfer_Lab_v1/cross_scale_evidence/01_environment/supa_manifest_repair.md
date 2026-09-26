# Supa-A1.1 Manifest Repair

The base commit's `Rineya-Supa-A1.1-1.1.0-devfreeze/FILES.json` did not verify.
Only `CORE.json` differed: the manifest expected 4,422 bytes and SHA-256
`edde6662...`, while the file committed in the same original publication commit
is 4,866 bytes with SHA-256 `bf640ab3...`.

`git log -p` shows why: the publication filled the previously empty `checks`
block in `CORE.json` but did not regenerate the already-created `FILES.json`.
No model, config, tokenizer, evaluator, or result file changed. This research
branch corrects only the `CORE.json` bytes/hash entry. The initial architecture
audit that rejected the stale manifest is retained as `attempt_0001_no_go`.

The architecture audit uses the corrected tracked manifest as an explicit
override when verifying the local weight vault. Projection cannot receive GO
unless every listed file then verifies, including the published weight hash
`a99f079d...`.
