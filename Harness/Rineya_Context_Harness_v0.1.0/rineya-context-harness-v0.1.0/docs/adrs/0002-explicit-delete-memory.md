# ADR 0002: Explicit Delete-Memory Primitive

**Status:** Accepted

## Decision

Provide an explicit `delete_memory` operation using stable IDs. The model-visible contract exposes reversible active-context eviction and archive. A guarded persistent-purge scope exists only in a separate administrative code path and is disabled by default.

## Rationale

Recursive-state forgetting must produce actual model-visible context reduction, not only an internal “obsolete” label. At the same time, ordinary model tool use must not expose irreversible deletion authority.

## Consequences

- Deletion becomes testable and traceable.
- Context must rebuild after commits.
- Dependency, pin, provenance, and active-evidence replacement guards are mandatory.
- Purge needs separate operational/privacy qualification and cannot be treated as ordinary RSF.
