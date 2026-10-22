# TWM-001 execution report

70 passed, 4 skipped, 0 failed. All executed tests ran on CPU.

The four optional actual tiny-Qwen3 tests were skipped because Transformers
was unavailable. Structural Qwen-like test doubles test wrapper contracts,
not the real architecture or model capabilities. Native 0.6B training,
retention, CUDA memory, and throughput were not run.

The standalone smoke exercised all five arms on fixed-tensor prefix-mean
regression. Its loss values are an optimization diagnostic, not a language
or reasoning benchmark and not evidence that triadic memory beats a control.

Raw evidence: `junit.xml`, `reference_smoke.json`, and
`QUALIFICATION_RECEIPT.json`. The receipt binds the source hash and records
actual per-arm parameter and buffer counts.
