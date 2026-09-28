# HOTFIX-001 correction notice for preserved historical evidence

Original JSON and markdown in this directory are retained byte-for-byte. They are historical claims, not current qualification.

Three of four historical CED comparisons failed the strict elementwise metric. Exact argmax plus a global-scale bound does not erase those failures. The claims that the observed difference is definitively a benign architectural consequence, that the ordinary control excludes a conversion defect, or that FP32 cannot carry a 1e-4 absolute bound through these depths are not established by the supplied observations.

The original validator recorded parent hashes but did not recursively verify an actual receipt DAG. It did not require stage-specific check names. Four referenced checkpoint binaries are missing from the uploaded ZIP. Current schema-2 qualification requires fresh execution and the complete new checks/ancestry/artifacts.

The original implementation digest matches the uploaded code using Windows path keys. It must not be treated as an unexplained code-drift failure just because POSIX keys hash differently. This hotfix normalizes keys prospectively; no historical JSON has been rewritten.

See ../REVIEW_REPORT.md, ../RUNBOOK.md and ../amendments/HOTFIX_001.md. A failed or interrupted review is not a clean review. No new official-model recovery or efficacy result is supplied here.
