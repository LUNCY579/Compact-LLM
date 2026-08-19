# ADR 0004: SQLite Materialized State Plus Events

**Status:** Accepted

## Decision

Use SQLite for the local-first v0.1 store, combining materialized memory tables with append-only transition events and checkpoints.

## Consequences

- Strong transactions with no external service.
- Simple Windows deployment and backup.
- Multi-user/high-throughput service deployment may require a future store implementation.
