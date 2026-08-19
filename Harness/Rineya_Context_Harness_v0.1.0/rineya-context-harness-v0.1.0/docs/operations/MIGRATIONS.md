# SQLite Migration Policy

## Purpose

The local state database is part of the harness contract. A schema change can affect recoverability, idempotency, provenance, and deletion guarantees, so migrations are ordered artifacts rather than startup-time ad hoc DDL.

## Source of truth

Repository migrations live in:

```text
migrations/
```

Wheel/runtime copies live in:

```text
src/rineya_harness/migrations/
```

Release validation requires the two sets to match byte-for-byte. Runtime initialization reads the packaged directory by default.

## Naming and ordering

Migrations use a four-digit monotonic prefix:

```text
0001_initial.sql
0002_idempotency_request_hash.sql
```

Rules:

1. Never modify a migration that has been released.
2. Add a new higher-numbered migration.
3. Keep each migration deterministic and forward-only.
4. Create or preserve `schema_migrations` before recording application.
5. Apply under an immediate transaction; a failed migration must roll back.
6. Do not start workflow execution until all discovered migrations succeed.

## Development procedure

1. Copy the current database before testing a schema change.
2. Add the migration under both source locations.
3. Add a fresh-database test.
4. Add an upgrade-from-prior-schema test.
5. Verify repeated initialization is idempotent.
6. Run schema export, repository validation, and the complete test suite.
7. Record behavioral consequences in `CHANGELOG.md`.

## Compatibility policy

Version 0.1 guarantees only ordered forward migration within the 0.1 line. It does not promise downgrade migration. Rollback means restoring a pre-migration database backup and the matching application build.

Before a future stable release, qualification must add:

- N-1 and N-2 upgrade fixtures;
- interrupted-migration recovery tests;
- database backup/restore rehearsal on Windows and Linux;
- maximum supported database-size measurements;
- explicit retention and purge migration semantics.

## Failure handling

If migration fails:

```text
stop startup
preserve the original database
emit the failing version and exception
never continue with a partially understood schema
restore backup or repair with a reviewed migration
```

No model is permitted to generate or apply migrations at runtime.
