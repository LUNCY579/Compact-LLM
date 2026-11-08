# Publication limits and preserved lineage

Current executable/source checks passed: all133 accepted fresh-V2 dispatch files,
70 completed base sources,35 base tests and96 completed passive-probe bindings
match their recorded bytes. Scientific raw journals, corrected analyses, native
state/asset identities and failed-run lineage remain preserved.

Two older whole-directory manifests also captured mutable administrative files.
The current files differ because the campaign subsequently advanced. Most exact
earlier metadata copies were found in archives, but three original byte copies
were not found in this review:

| Historical manifest | Administrative file | Recorded SHA256 not recovered |
|---|---|---|
| `runs/native-bulk-restart-001/source_manifest.json` |`OWNERSHIP.json`|`1f7a989c282522ab92b8646b4fd48fa43a95c8aea8081cb697d90d10c0799c4b`|
| Same bulk manifest |`STATE.json`|`4c6f59b1df13f407ba40293301942535b4f3154774fffb6f7d6ab767c8a469ae`|
| `runs/native-cpu-causal-replay-003/source_manifest.json` |`STATE.json`|`84ef822e29accce4bcc7b6a65b6091eacaf521b27c78100de90bc8bac22d510f`|

Their commitments remain recorded. No reconstructed file is presented as an
original copy. This prevents a claim that every administrative byte in those
historical full-directory manifests can be recreated exactly. The review found
no corresponding executable-source, task-data or outcome corruption; that
does not erase the administrative retention gaps.

The publication uses `.gitattributes` to disable text normalization and checks
the staged Git blobs against local SHA256 values. Historical base HEADs,
absolute paths, timestamps, grants and source digests are not rewritten to the
publication commit. The metadata correction is additive and the original
scientific reports stay intact.

The publication manifest records each included or omitted file and its digest.
Only recreated caches, duplicate tokenizer assets and CPU fixture sandboxes are
omitted. Files remain locally intact. Raw measured `.npz` vectors and tiny
run-specific HF module caches are retained because native readers need the
actual loaded custom-model source bytes. Canonical weights are not included.

The newer October9 publication instruction authorizes this commit/push only.
It does not resume the blocked positive-gain goal or qualify the unrun432-case
native successor. Fresh qualification, full native measurement, preservation
grading and independent outcome review remain pending.
