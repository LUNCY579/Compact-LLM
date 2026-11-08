# Wave 4 — CE8 skill acquisition preservation

Retained source commits: `codex/ce8-production-dev-20261002@bbb96fd59a192df1314d82fa461b0d5658bca8be` and `codex/ce8-reviewed-evidence-20261009@16f57f9dc387933f3b40f4c3ea596c5e530ec9c3`.

Two original, exact research subtrees from the reviewed source are imported:
- `experiments/skill_acquisition_method_lab_v1`
- `experiments/skill_acquisition_native_v1`

Existing `experiments/maxrl_*` versions in central are NOT overwritten. The production and reviewed source branches diverge; both original commit histories are retained as additional commit parents, preserving any differences without taking precedence over central. Do not infer that CE8 native qualification or effectiveness has passed from merging this historical package.

No CI/workflow modifications. Only retire old source branches after confirming central contains both commit histories and reviewing running native-host/launcher/branch-name consumers.
