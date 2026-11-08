# Wave 3 — FWT-001 remote provenance observation (archived, not activated)

Original branch: `research/ced-full-width-remote-receipt-94fbe179-v1` at `1490dc05bb9bd8d363d214a51115685697851f8f`.

Original content preserved without modification:
- Original observation JSON `.github/receipts/fwt-001-94fbe179-remote-observation.json` retains its existing filename and original Git blob.
- Original GitHub Actions workflow `.github/workflows/fwt-001-source-provenance.yml` is preserved byte-for-byte under `docs/repository-consolidation/2026-10-22/inert_workflows/fwt-001-source-provenance.yml`. This deliberate relocation prevents activating an obsolete workflow on central.
- The source commit is the integration commit's second parent, preserving the full original file paths in its history.

**Evidence boundary:** the recorded two remote attempts failed before job steps (no runner, no remote tests, no provenance PASS, no model run). This is historical observation, not official P1/P2 or quality evidence. The companion source previously reported 75 local scoped tests; they are not rerun by this PR.

If a live workflow is desired, introduce it separately after reviewing event triggers, privileges, current targets and test requirements. Owner may retire the original branch name only after merge/reachability verification and any external branch-name consumer handoff. No baseline or CRS changes.
