# Project Governance

Rineya Context Harness is currently a maintainer-led research project.

## Decision authority

The repository owner or designated maintainers approve releases, public interfaces, persistence migrations, security-boundary changes, and research claims. Contributors may propose changes through issues or pull requests.

## Architecture decisions

A change requires an ADR when it modifies any of the following:

- trusted versus untrusted boundaries;
- memory or deletion semantics;
- database schema or recovery guarantees;
- model-facing tool contracts;
- workflow topology or termination rules;
- authentication, networking, or permissions;
- public API compatibility;
- evaluation or release gates.

ADRs record context, decision, alternatives, consequences, migration, and rollback. They are not silently rewritten after acceptance; later decisions supersede them with a new ADR.

## Release decisions

A release candidate must have:

- a versioned manifest;
- passing schema, compile, demo, and test checks;
- documented known limitations;
- a clean artifact without runtime databases, caches, secrets, or private traces;
- rollback instructions;
- an explicit distinction between fixture evidence and real-model evaluation.

Model claims and harness claims are versioned independently. A model improvement must not be attributed to the harness without an ablation, and a harness improvement must not be attributed to new weights without a frozen-checkpoint comparison.

## Security decisions

Security-sensitive changes require threat-model review. Irreversible operations, public network exposure, persistent personal memory, or side-effecting tools are out of scope for v0.1 and require a new release line or explicit architecture approval.
