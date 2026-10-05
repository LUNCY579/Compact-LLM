# Terminal preservation and successful audit commands

`preserve_terminal.py` performs only source/terminal-file IO. It never invokes a
process, Git/HEAD query, native API, PID/Job query, GPU, decoder or model. Root
supplies the frozen terminal file SHA and separately anchored current death
observations. The recorded supervisor cleanup/death evidence and Root's current
observations remain separately labeled. A cleanup refusal happens before any
evidence destination is created. A failed copy leaves an unsealed partial
destination, preserves the original attempt and never publishes a final receipt.

The script validates exact owned config/inventory/leaf paths, complete source131
and every config inventory map, the supervisor partial-output byte map, exact
lease terminal byte equality and recorded reservation/worker/resume/raw ownership
identities. Successful and failed launched workers require recorded Job cleanup,
no live owned worker, terminal return code, sampler/NVML cleanup and Root current
death proof covering the entire observed process birth roster. Admission-denied
and no-worker receipts require exact false lifecycle flags and no contradictory
worker/resume files; they never qualify for a successful audit.

For launched workers the current proof contains schema_version1,
kind `ROOT_CURRENT_OWNED_IDENTITY_DEATH_OBSERVATION_V1`, full `identity`,
`owned_identity_sha256`, sorted minimal `owned_identities` each containing exact
PID/FILETIME/create_time, `owned_identities_sha256`, and `identity_checks` for
every same identity with currently_alive false. It also contains currently_alive
false for the full primary identity, observer `native_supervisor._pid_alive`, its
config-pinned `observer_source_sha256`, source_commit, config_file_sha256,
supervisor_receipt_file_sha256, observed_utc after terminal finished_utc and body
sha256 excluding itself. The roster is independently reconstructed from the full
worker plus every CPU `owned_process_identities` record, keyed PID/FILETIME.
No descendant may be omitted. Root wrote the agreed proof outside the raw leaf:

`ROOT_CURRENT_OWNED_DEATH_BATCH04_A1_001.json`, file SHA
`f5e79cfa99052ff01f73cc06902bdc1f4a331896613a9918b8a32f662ebf1158`.
The preserver does not modify this proof or the terminal raw leaf.

The copied evidence leaf has flat raw files, config_snapshot.json,
audit_inventory_snapshot.json, complete repository-relative source_snapshots131,
separate auditor_dependency_snapshots for the six registered dependencies, a copy
of Root's death observation and the final preservation_receipt.json. Each copy
is exclusive, streamed in2MiB chunks, fsynced and read back. Before receipt seal,
every original and copied raw/config/inventory/source/auditor/death byte map is
checked again, as is the complete original lease map. Source drift on a failed
attempt is retained and explicitly recorded; success requires exact source pins.

Root holds exact clean launch HEAD
`dc4e52ef3a7b6b6cde51f1ab54a68b2d63be5510`, config file
`e6d56e4a592b68bc7e288022b98bd73e94a468f6243625579bfb3bcb7d3c0b4d`,
profile body
`2d5547910aa57b0509a532972d899ceb5c913e41274bfa773d25536e6d8c9676`,
final inventory file
`cc28dd3d02a8f550699892428022e0ebf61c2cdcbf2dd9286f948cd1aba4a6ea`.
The selected raw/evidence basename is
`short_m0_atomic_batch04_phase3_attempt1_20261005`.

Root's actual preservation command, only after terminal and independent script
review, from repository root:

```powershell
python -I -B experiments/rineya_synth_centralized_reasoning_v1/artifacts/phase3_launch_registration_work/preserve_terminal.py --root-terminal-frozen --supervisor-file-sha256 2571e27872e6206560cdedd1631b690f4f9420af6b3eb4c0dfc1cac30f619818 --current-death-evidence experiments/rineya_synth_centralized_reasoning_v1/artifacts/phase3_launch_registration_work/ROOT_CURRENT_OWNED_DEATH_BATCH04_A1_001.json --current-death-file-sha256 f5e79cfa99052ff01f73cc06902bdc1f4a331896613a9918b8a32f662ebf1158
```

The command prints preservation body SHA. Independently select that exact anchor,
then Root's successful full native raw/model/decoder/resource audit command is:

```powershell
python -I -B experiments/rineya_synth_centralized_reasoning_v1/artifacts/phase3_launch_registration_work/audit_preserved_success.py --root-terminal-and-preserved --preservation-body-sha256 EXACT_PRINTED_PRESERVATION_BODY_SHA256
```

This second command invokes the already reviewed actual native audit with every
exact external anchor. It is intentionally not run during CPU preparation.
Failed/admission-denied or CPU-only preservation receipts refuse this command.
Neither preservation nor CPU tests certify native mechanics, correctness,
enrollment, study training, transfer or retention. The complete three-arm study
still requires its actual results.
