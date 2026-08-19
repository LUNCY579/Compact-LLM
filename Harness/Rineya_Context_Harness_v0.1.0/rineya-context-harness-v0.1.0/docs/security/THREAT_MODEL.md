# Threat Model

## Assets

- user question and recent context;
- model configuration and API keys;
- local state database;
- evidence archive;
- trace files;
- local model endpoint;
- search endpoint;
- purge authorization token.

## Adversaries and failure sources

- malicious webpage content;
- prompt injection inside search results;
- SSRF through crafted URLs or redirects;
- stale DNS/private-address resolution;
- malformed model JSON;
- model-proposed deletion of required state;
- replayed or concurrent stale patches;
- accidental permanent purge;
- trace leakage;
- denial of service through large pages or loops;
- compromised local search/model service.

## Controls in v0.1

- read-only external tool surface;
- URL scheme/port/private-address restrictions;
- bounded redirects, bytes, calls, rounds, and wall time;
- content-type allowlist;
- untrusted evidence delimiters;
- heuristic injection flags;
- strict model schemas;
- stable memory/evidence IDs;
- optimistic revisions;
- idempotency keys;
- pin and dependency protection;
- purge disabled by default;
- local API binding;
- payload-bounded traces;
- checkpoints and export hashes.

## Residual risks

- DNS rebinding/connect-target mismatch is not fully eliminated by preflight resolution.
- Text-only extraction can miss visual context.
- Injection heuristics are not complete.
- Local model servers may handle JSON schema differently.
- SQLite files and traces are not encrypted by the application.
- No multi-user authorization model exists.

## Recommended deployment controls

- run fetch in an isolated, credential-free egress process;
- enforce network policy outside the Python process;
- encrypt the device or artifact volume;
- protect model/search endpoints with local firewall rules;
- disable trace payloads where sensitive data is possible;
- add authentication before non-loopback API exposure;
- perform dependency and container scanning in CI;
- rotate any purge/API secrets.
