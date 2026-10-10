# Independent review of publication CPU failures

The combined publication run remains **647 passed, four failed, zero errors and zero skips** across 651 tests. Its failure log and JUnit must remain intact; this review does not convert that run into a full-suite pass.

The fresh bootstrap positive fixture used a basetemp outside `quality_pattern_v3`, while production correctly requires its run root inside that directory. Rerun only that node with a fresh V3-local basetemp.

The probe reader and fresh-bank no-Torch assertions inspect the entire interpreter's `sys.modules`. Other tests import Torch during combined collection. Run the probe reader import assertion alone in a clean process. The fresh-bank test also contains an unreached assertion that the fresh corpus does not exist; the reviewed corpus now legitimately exists. Use a separately named clean-import and unchanged-corpus-manifest check for its present-day requirements. Do not report the original whole fresh-bank node as passing.

The passive authority positive test requires the actual passive study to remain unused. That study completed, so its current refusal is correct. A new scoped witness should check scope acceptance and actual consumed-study refusal without modifying the ledger or replaying the model.

These four observed failures establish test context and lifecycle limitations; they do not establish a production source defect. The coordinator owns the justified follow-up checks, whose outcomes are not asserted here. Held source and tests remain unchanged, and no model or GPU work is needed.
