# State Transition Contract

## Patch lifecycle

```text
PROPOSE → VALIDATE → COMMIT → VERIFY → REBUILD CONTEXT
```

## Required patch fields

- run ID;
- expected revision;
- actor;
- one or more typed mutations;
- idempotency key;
- human-readable reason;
- optional metadata.

## Technical invalidity

The patch is technically invalid when:

- schema parsing fails;
- referenced run or memory does not exist;
- revision is stale;
- ID is duplicated;
- an immutable field changes;
- a dependency is broken;
- pin rules are violated;
- evidence provenance is absent;
- purge policy is violated.

Technical invalidity is not a subjective quality score; the transition does not commit.
