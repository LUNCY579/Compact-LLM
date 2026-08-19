# Security Policy

## Supported versions

The project is currently an alpha research release. Security fixes are applied to the latest `0.1.x` line only.

## Reporting a vulnerability

Do not publish a suspected vulnerability, secret, exploit trace, or private dataset in a public issue. Report it privately to the repository owner or designated security contact once the public repository provides one. Include:

- affected version or commit;
- minimal reproduction steps;
- expected and observed behavior;
- security impact;
- whether secrets or personal data were exposed;
- suggested mitigation, when known.

Until a public contact is configured, keep the report local and do not expose the affected service to untrusted networks.

## Security posture of v0.1

Rineya Context Harness v0.1 is designed for local, read-only research use. It is **not** approved for direct public-Internet exposure or high-consequence autonomous actions.

Default protections include:

- API binding to `127.0.0.1`;
- no shell, email, purchase, browser-click, or deployment tools;
- permanent purge disabled by default;
- explicit optimistic revisions and idempotency keys;
- pinned invariant and dependency checks;
- immutable evidence records;
- outbound `http`/`https` restriction;
- localhost/private/link-local/reserved address blocking;
- port allowlisting, redirect limits, response-size limits, and timeouts;
- untrusted-source delimiters and prompt-injection flags;
- bounded model/search/tool loops;
- connected traces and state events.

## Deployment requirements

Before any non-local deployment, add and test:

- authentication and authorization;
- TLS termination;
- request and response size limits;
- rate limiting and abuse controls;
- isolated, credential-free network egress;
- DNS resolution and connection-address pinning to reduce DNS-rebinding risk;
- secret management rather than plaintext configuration;
- trace redaction and retention policy;
- encrypted storage where sensitive data is possible;
- backup, restore, and schema migration procedures;
- dependency and container scanning;
- model/runtime-specific prompt-injection evaluation;
- human approval for irreversible or consequential actions.

## Prompt injection and untrusted evidence

Search snippets, webpages, imported memories, and model-proposed state patches are untrusted. Source text never becomes policy. A fetched page cannot grant tools, alter budgets, disable verification, or authorize deletion. Pattern scanning is only an auxiliary signal; deterministic permissions and trust-boundary separation are the primary controls.

## SSRF limitations

Application-level URL validation is defense in depth, not a complete network sandbox. A production fetch service should run in a separate low-privilege process or container with restricted egress, no cloud metadata access, no ambient credentials, and explicit connect-time address validation.

## Data deletion

`delete_memory(scope="active_context")` is reversible context eviction. `archive` is also reversible. `persistent_purge` is a separate administrative operation, disabled by default, and should not be represented to users as secure erasure unless the deployment also handles database pages, WAL files, backups, traces, exports, and storage media according to an approved deletion policy.

## Secrets

Never commit API keys, tokens, credentials, private URLs, model licenses, user conversations, runtime databases, or traces containing personal information. The `.env.example` file contains names only, not usable secrets.
