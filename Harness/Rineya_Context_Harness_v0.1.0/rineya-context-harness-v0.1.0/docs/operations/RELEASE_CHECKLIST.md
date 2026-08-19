# Release Checklist

- [ ] Version values agree across package, API, schemas, changelog, and ZIP prefix.
- [ ] Generated schemas are current.
- [ ] Unit and integration tests pass.
- [ ] Coverage meets the configured threshold.
- [ ] Restaurant, Hanoi, and fixture-research demos pass.
- [ ] Pinned, dependency-breaking, stale-revision, and purge-disabled tests pass.
- [ ] Repository validation report is `pass`.
- [ ] Runtime databases, WAL/SHM files, traces, caches, bytecode, and secrets are absent.
- [ ] Manifest file count and aggregate hash are generated.
- [ ] ZIP integrity test passes.
- [ ] SHA-256 sidecar is generated.
- [ ] Clean-extraction smoke test passes.
- [ ] Known limitations and rollback path are documented.
- [ ] Fixture evidence is not presented as real-model benchmark evidence.
- [ ] Sealed evaluation was used only after candidate freeze, when applicable.
