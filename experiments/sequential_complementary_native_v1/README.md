# Current SCP boundary review routing

**Native execution is code-locked. Training and confirmation remain closed.**
Read [DEDUPLICATION_REVIEW.md](DEDUPLICATION_REVIEW.md) for the proposed single
implementation lineage and [BOUNDARY_REVIEW.md](BOUNDARY_REVIEW.md) for the current
execution boundary before using any historical instructions below.

The recommended continuation is `research/sequential-complementary-native-boundary-v1`
at `9551d7ba50784a5acd19e8512faaba6d0efc6aca`. The divergent
`research/scp-native-qualification-boundary-v1@7486a1490517e9bd0ed8e29631a601d713ec20e8`
is a retained, non-contained reference, not a redundant branch approved for deletion.
This documentation-only review does not merge implementations or grant authority.

The `propose`, `qualify`, and `_worker` CLI commands described below are historical
and refuse on the current boundary. No native launch command is available. Neither
source-branch guard is changed; `draft` also refuses on the documentation-review
branch rather than silently rebinding the pending request to new source.

---

The original integration-review README is preserved verbatim below.

# SCP-NQ-001 — Native integration review snapshot

**Published for code review, not native-qualified.** This package is the isolated
successor to `research/sequential-complementary-skills-v1` at
`3ad2769a9f8343a9697e1720fcdc9efa18c3845a`. Its dedicated branch is
`research/sequential-complementary-native-v1`.

The eight Python files in this publication are byte-for-byte copies of the local
implementation supplied in the preceding conversation. Publication adds this
review guide, continuation rules, and preserved test artifacts; it does not run
models or convert fixture evidence into a native PASS.

## Review map

| File | Responsibility |
| --- | --- |
| `contracts.py` | Exact Supa/trained-partial/Titan-Sparse asset identities, 310-tensor full inventory, 220-tensor update inventory, clean source binding, and a separate inference-only grant validator. |
| `native.py` | Exact FP32 subject reconstruction/materialization, zero-update student probes, and the fixed unscored REP/MR qualification backend. |
| `run.py` | MQ foreground supervisor/worker claim/lease/resource guards and the unchanged RRP persistent Journal; student-before-MR receipt checks. |
| `tests/test_native_contracts.py` | CPU asset, precision, scope, tokenizer, cache, reasoning and rejection fixtures. |
| `tests/test_supervision.py` | Journal/receipt fixtures and MQ child-process supervision/timeout tests. |
| `../sequential_complementary_v1/reasoning.py` | Narrow native adapter check; ordinary CPU callback behavior remains available. Native execution is restricted to the new fixed qualification lane. |

MQ/RRP implementation files, historical permits, registrations, source pins,
model catalogs and weight files are not changed by this publication. The parent
SCP workflow/status and historical README remain preserved; this package does not
claim to satisfy their full instrument, training or capability-experiment gates.

## Evidence and limitations

`evidence/prior_session_pytest.txt` and `evidence/prior_session_junit.xml.gz`
preserve the earlier local report: **112 passed**, no failures/errors/skips,
5.02 seconds in the text output. The XML reports 5.018 seconds and timestamp
`2026-09-18T20:32:57.704261+00:00`. These are imported prior-session artifacts,
not a fresh test execution at this publication commit. They contain no
independently verifiable complete-checkout source binding.

The Journal tests isolate the inherited budget module's unrelated training
preflight imports, as explained in `tests/test_supervision.py`. The report is
neither a complete RRP regression nor a rerun of the original 85 SCP tests.
Publication-time validation compiled all eight Python files in memory without
executing imports; it was a syntax check, not a runtime test.

`evidence/PUBLICATION_REVIEW.json` records source hashes, artifact hashes, these
scope limits, and the still-unexecuted native status. It is not an execution
approval or a native result. Remote publication must be verified separately by
reading the branch ref and changed-file blob hashes; no self-referential
"pushed" claim is embedded in this snapshot.

For a fresh local CPU review from the complete checkout, with the test
dependencies installed, keep generated records outside the checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  experiments/sequential_complementary_native_v1/tests -q \
  --junitxml=../scp-review-junit.xml
```

Use a new output filename for every attempt. This command is documented for the
reviewer; it was not executed during publication.

## Native gate remains separate

The public CLI provides `propose` and `qualify --phase student|mr`. `_worker` is
an internal supervisor-only entry point. `propose` requires a clean full checkout
and the actual exact local assets, and emits a **PENDING** budget request. No
approved grant is supplied here. Do not replace missing assets, invent an approval,
reuse a historical RRP training permit, or lower resource thresholds.

`student` is inference-only qualification, not training qualification: it performs
no backward probe or optimizer update. `mr` requires a settled subject-specific
student receipt with matching source, runtime, state and journal bindings. It
uses only the fixed unscored software prompt, not a research evaluation bank.
A successful future run would establish only the recorded mechanical scope.
Malformed/truncated generation observations are retained, not filtered successes.

Before any native launch: finish code review and full-checkout regression checks,
verify the exact assets and pinned runtime, create an explicit numeric
inference-only approval, and retain all records outside the checkout. Training,
teacher-dependent comparisons, protected-bank reads, confirmation, release and
quality claims remain closed. This publication starts none of them.
