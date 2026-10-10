# Publication tooling review record

The publisher and its fixtures are new standard-library tools for evidence storage. Their tests do not run archived research suites, solvers, native consumers or models. No source corpus is used as a developer fixture.

The non-author `publication_safety_review` agent read tooling and permitted repository metadata only. Its review identified and the coordinator repaired these concrete defects:

- Physical source roots and supplement paths were not gated before filesystem probes. Names now pass the policy before inspection; supplement basenames must match their four declared identities. Cumulative physical program paths also prevent rebasing an unreferenced program into the ordinary source namespace.
- The verifier rejected two omission classes emitted by the publisher. Both classes now round-trip without reading omitted bodies.
- Exact cross-program file references admitted descendants. Only the single owned program prefix admits a subtree; the four referenced files remain exact names.
- Cross-volume staging and partial part migration could leave uncatalogued artifacts. Staging now uses the archive volume; an object transaction rolls back only its exclusive new files and database rows. Failure while creating an object directory is fatal and preserves the incomplete archive rather than claiming a successful capture.
- Unsafe source names, including mixed unsafe/protected names, could generate unverifiable omission records. A pure name check runs before policy/stat. Such strings remain inert omission evidence and are never joined to a restore destination.

The documentation reviewer read only the new README, research summary and publisher policy. Its suggestions clarified heuristic credential detection, per-file stability rather than a global freeze, and RPQ2's exclusion from training rather than exclusion of its records from this archive.

Actual synthetic verification history:

- Initial 18-method run: 17 passed, one actual Windows symlink test skipped.
- After pilot/raw-corpus/method-source scope changes: 19 methods, 18 passed, one skipped.
- After principal safety repairs: the first 25-method run had one error because its mocked `os.scandir` also intercepted Python's temporary-directory cleanup. That test failure was retained as a finding; the mock was narrowed to the synthetic source. The next run passed 24 methods, one skipped.
- After final edge repairs: 27 methods, 26 passed, one skipped. Exact saved stdout/stderr and final source pins accompany this publication.

The archive capture began with the exact tool source preserved under `publication_tools/history/capture001/` and pinned in `CAPTURE_TOOL_PINS.json`. Subsequent edge repairs were made to the published verifier while that already loaded capture continued. The capture was not restarted. Capture-time and final-source identity must remain distinct; final archive integrity is established only by its actual completed verification receipt.

The skipped test reflects this account's inability to create actual symlinks. Windows junction behavior, malicious concurrent filesystem replacement and hostile operating-system enforcement are not qualified by these tests. Streaming chunks and disk indexes bound the implementation's working data; this is not a measured complete native runtime envelope or a research resource qualification.

Secret detection is heuristic. Included-file identity and restoration are tested; the publisher cannot prove that arbitrary unseen secret formats do not exist. No historical dataset correctness, admission or learner-gain credit follows from these tooling checks.

The first full archive verification was stopped through its exact owned execution session after an executed SQLite query-plan diagnostic identified two quadratic catalog scans. Its incomplete outputs and exact source remain under `validation_history/verify001/` and `publication_tools/history/verify001/`. The stopped run has no PASS; its completed phase and counts remain UNKNOWN. A filtered observation of the unique stopped verifier command found no matching live executor. That observation is not a complete process-tree or lifetime-budget qualification.

The separately reviewed repair adds a nonunique path-object lookup index and replaces escaped case-insensitive LIKE with a literal BINARY descendant-key range. Every object, part, path, alias, collision, omission and materialized-file check remains required. The new 28-method run passed 27 methods with the same Windows symlink skip; its new witness compares special-character and Unicode prefix behavior and executes query plans. `VERIFY002_*` records the separate repaired-source run's actual result; static equivalence is not a substitute for that result.

The first Git byte checker completed 180,735 file comparisons. Its exact source and captured index are preserved. Review found a general rename-scope omission and missing end-of-check index comparison; the corrected helper has source-review closure. The original actual run's index was independently shown unchanged afterwards, with outside-scope checking performed using `--no-renames`. Later changed files and reports require separate exact byte binding before commit; the historical run is not relabeled as an execution of the corrected helper.
