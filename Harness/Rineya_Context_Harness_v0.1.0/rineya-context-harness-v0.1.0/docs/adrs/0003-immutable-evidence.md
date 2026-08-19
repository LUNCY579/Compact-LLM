# ADR 0003: Immutable Evidence, Mutable Working State

**Status:** Accepted

## Decision

Store retrieved passages and provenance separately from active memory. Evicting a passage from active context does not remove its evidence record.

## Consequences

- Citations and audits remain possible after forgetting.
- The state store contains compact references rather than complete webpages.
- Storage retention policy is separate from prompt-retention policy.
