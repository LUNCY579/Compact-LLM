# Byte-backup design and review gate

## Scope and identity

The only production source root is `C:/Users/clclc/BusinessProject/Rineya-Dataset`. The read-only metadata input is `backup_20261010/INVENTORY_METADATA003.json`, SHA256 `a034f6a694868503afc402f0d2bd925497a49c8e99c0a638395651da0e816640`, recorded status `METADATA_ONLY_NOT_PREPARED`.

All 102 rows with inventory roles `train_candidate_or_train_audit` or `declared_training_export_pending_source_status` are retained prospectively, without relabeling or outcome-based filtering. Their recorded byte total is 1,311,950,205. A conservative filename, role, ancestor, and size policy adds 38 README, data-card, source, provenance, split, and manifest metadata files from packages with declared TRAIN rows, plus the source-root README and DATA_CARD; their recorded total is 331,985 bytes. Examples, fixtures, protocols, evidence subtrees, other-owner program bodies, live state, caches, weights, databases, and opaque bundles are excluded from additional metadata selection.

Each file ID is SHA256 of canonical UTF-8 JSON `[original_relative_path, declared_role]`, assigned before rendering or reading any selected payload. Original relative paths and roles are kept verbatim. `training_eligible`, `training_ready`, and `training_authorized` remain false. Source owner, canonicality, rights-use, and native/model boundaries remain UNKNOWN; gains remain UNMEASURED. Full bytes are preserved; no JSONL parsing, response normalization, row truncation, semantic re-audit, dataset-world generation, model inference, or training occurs.

## Pre-read policy

`path_gate` is pure: it checks the original path, role, recorded size, forbidden tokens/ancestors, encoding, and metadata name without any stat, resolve, enumeration, or open. Blocked selected TRAIN rows remain in the selection and become explicit failed attempts. Allowed rows then undergo lstat checks on every ancestor, reject symlinks/reparse points and nonregular files, and resolve canonically within the exact source root. Output traversal is rejected before any filesystem access; output ancestors are also checked and the resolved destination must lie inside the owned backup directory and remain disjoint from source. Sources are opened read-only with no-follow where the OS offers it and compared against the expected descriptor fingerprint.

## Two independent reads

The first pass checks recorded size and hashes every raw byte. Raw bytes also pass the streaming credential signature scanner. For selected `.jsonl.gz` and `.txt.gz`, a streaming gzip decoder scans all expanded logical text, validates gzip CRC/EOF across concatenated members, and fails on malformed or truncated gzip. Memory is bounded by 1 MiB chunks. Expansion above the lesser of 32 GiB and `max(1 MiB, recorded_size * 1024)` fails the entire file explicitly; it never truncates or silently drops rows. A negative signature scan has a limited detector scope and does not establish absence of all secrets.

Source fingerprints contain device, inode, size, mtime_ns, and ctime_ns, compared before open, through the descriptor after read, and through a fresh gated path inspection. Every selected first-pass row gets an attempt receipt. Missing, denied, blocked, changed, or signature/encoding failures are explicit. All first-pass rows are checked even when earlier rows fail; any failure prohibits success and upload.

The second independent raw read happens through `tar.addfile`. It repeats both raw and expanded-text credential scans and gzip integrity/expansion checks, hashes the bytes supplied to tar, checks exact size plus a one-byte growth probe, and repeats source fingerprint checks. No first-pass digest is reused as second-pass evidence. A change after first scan or during archive writing fails capture. Two matching reads are stability evidence during each capture interval, not an atomic filesystem snapshot.

## Archive and full readback

The output is a deterministic streamed PAX tar inside gzip (`mtime=0`, empty gzip filename, fixed compression level). File member names are `source/<original_relative_path>`. Tar header timestamps and owner fields are fixed; source bytes are untouched. Original compressed TRAIN file bytes remain compressed file bytes inside the outer archive.

After creation, `verify_archive` streams the entire archive back. It validates every member's original identity, permitted path/role, type, size, and SHA256; rejects traversal, missing members, extras, duplicates, aliases, sparse or linked members; drains tarfile's internal read-ahead buffer and then gzip to reject nonzero post-EOF tails; validates outer gzip integrity; and computes the SHA256 and size of the whole compressed artifact. It never extracts to disk. A canonical content descriptor containing the inventory pin, preserved identities/roles, complete per-file byte hashes/sizes, and unchanged claims determines `content_sha256`.

Only after all source checks and readback pass are the archive and manifest given final names. Failure preserves partial local artifacts and receipts, with no successful manifest or upload. Late local publication failures demote final artifact names to failure names when storage permits. Output storage failure or uncatchable OS termination can prevent receipts; no missing receipt is interpreted as success.

## Producer receipts and review

`START.json` is produced by the executed target and pins actual PID, source root/mode, current tool SHA256, canonical selection SHA256, inventory SHA256, and configuration SHA256 including expansion bounds. Unavailable counts and observed process exit status start as UNKNOWN. `attempts.jsonl` records each attempt without payload text or matched credentials. `OUTCOME.json` records success or failure, declared claims, failed attempts, and the intended CLI exit code; observed process exit code stays UNKNOWN until a separate observer records it. Fatal exceptions and interruptions use UNKNOWN for unavailable counts.

Before production capture, the tool re-loads only the authorized pinned inventory metadata and re-derives the exact prospective selection; changed inventory/configuration is a hard failure. Both API and CLI require a current independent review receipt with passing synthetic controls, matching the exact tool, tests, selection file, inventory, and source-root pins. Synthetic fixture capture requires an explicitly passed disjoint temporary root. The review gate is an authorization/evidence check, not a security boundary against a malicious local caller manufacturing a receipt.

Synthetic controls cover exact raw/gzip bytes, both read passes, stale sizes, missing and blocked rows, purity before selected path access, source mutation despite mocked stat shortcuts, deterministic hashes, output traversal/reparse containment, gzip secrets/CRC/EOF/expansion, tar extras/duplicates/traversal/type/truncation/tails, and actual producer receipt pins. Test source is prepared here, but no test was executed during this builder task.

## HF publication boundary

No HF upload is implemented or executed by this tool. A separate authorized publisher must re-confirm PRIVATE visibility and apply the append-only content prefix `rineya-dataset-snapshots/<verified-content-sha256>/`. Archive, manifest, README/source evidence may live under this fresh prefix only. Existing differing prefix contents, dataset root README, and existing CRS snapshots must never be overwritten. The archive belongs on HF only and is Git-ignored; code, selection, source/evidence, and manifest are reviewable Git artifacts. No model/GPU/native/tokenizer/train/cloud-job/paid API/install/weights actions or protected CHECK/FINAL/eval/spent/privatebank/sealed payload reads belong to this workflow.
