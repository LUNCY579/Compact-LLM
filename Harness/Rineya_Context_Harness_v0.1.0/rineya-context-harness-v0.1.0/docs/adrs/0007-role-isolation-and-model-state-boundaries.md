# ADR 0007: Role Isolation and Model-Owned State Boundaries

**Status:** Accepted

## Decision

Preserve recent conversation as `user` and `assistant` messages; never serialize it as trusted system policy. Treat all model-proposed state changes as untrusted proposals. Restrict model updates to semantic content and provenance fields, and keep pins, global invariants, lifecycle status, purge, permissions, and commits under deterministic control.

## Rationale

A small model benefits from a compact prompt, but compaction must not change authority. Converting historical user or assistant text into a system message creates a privilege-escalation path. Likewise, allowing a model to alter lifecycle or pin fields would let it weaken the very invariants intended to constrain forgetting.

## Consequences

- Context compaction must preserve message roles and untrusted-data delimiters.
- The model cannot create or modify global invariants.
- The model-visible delete tool exposes only reversible scopes.
- Protected transitions require user/system/admin code paths.
- Additional validation and schemas are required, but failures become attributable and testable.
