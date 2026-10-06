# Offline library evidence replay: CPU component

The [source validator](evidence/library_evidence_reviewed_cpu_20261005/prefix_library_evidence.py)
replays a strict raw packet against externally selected source/policy/context,
PID/FILETIME, byte-map and ordered-boundary identities. Required checkpoints
include initial, each nested enter/exit, precleanup and final cleanup; each carries
complete snapshots, cumulative events and directory identities. The validator
recomputes transitions and counts separately from producer claims.

[Root13 controls](evidence/library_evidence_reviewed_cpu_20261005/ROOT_LIBRARY_EVIDENCE_REPLAY_003.json)
pass with unchanged source hashes, no errors/failures/skips. They use inert actual
files and explicitly injected event/process observations. Missing/reordered/
rewritten prefixes, absent loads, transient unloads, unregistered identities,
handle aliases, wrong boundary order, descriptor/context/source substitutions,
clock/directory drift, numeric boolean impersonation and false cleanup fail.
V2 permits only honest UNOBSERVED foreign registrations and refuses a complete
search-directory coverage claim. CPU identity and false native authority persist.

[Preservation](evidence/library_evidence_reviewed_cpu_20261005/ROOT_PRESERVATION.json)
retains the first fixture failure and both source variants. That fixture shared
an external descriptor object with its incoming identity; deep-copying the
independent selections corrected the test without weakening the validator.

This is a runnable offline component and prospective packet contract. The
current LibraryGuard emits summaries that cannot satisfy this packet. Actual
raw packet emission, failure accounting, final-drain persistence and worker/
terminal-auditor consumption are unimplemented. A packet checksum or replay
cannot qualify actual source/runtime bytes, observer completeness, OS bootstrap,
process resources or native authority. No native/model/GPU/WinAPI work ran.
The M0/M1/M2 fresh-family, RS-transfer and preservation comparison remains required.
