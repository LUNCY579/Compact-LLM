# RAB-001 agent contract

Work only on an explicitly authorized new branch unless the user directs otherwise.
Never change or overwrite published cores, source weights, unrelated experiments,
existing tags, or main to make this benchmark pass.

Before editing, read README.md, protocol.json, source_lock.json, materials.json,
and the current source-bound receipts. Preserve null, failed, and blocked results.
Random-model tests, metadata-only parameter counts, and CI success are not official
model evidence. Neither a 25-step smoke nor this foundation tag establishes quality.

Keep the four-arm matrix and explicit 128-dimensional head convention. Calibrate
only on declared calibration data. Preserve one residual basis, complete GQA groups,
original layer order, tokenizer IDs, and tied embeddings. Never use mismatched-size
loading, stock Qwen/R2 fallback, a custom abstention readout, or undocumented donors.
Record every tensor selection and newly inserted layer. Source files are immutable.

G0 is unmodified official-model/readout qualification. G1 is mechanical conversion.
G2 is exactly 25 engineering steps, not recovery. G3-G5 are intentionally unimplemented
and fail closed. Do not add a bypass flag or fabricate a receipt. Before implementing
G3, freeze group-disjoint data and hashes, resource/training/search budgets, meaningful
effect and regression margins, controls and the cost ledger. C40 needs a recovered
ordinary donor and matched ordinary continuation; do not quietly give it extra training.

Run from the repository root:
    python -m pytest -q experiments/architecture_benchmark_001/tests
    python -m experiments.architecture_benchmark_001 matrix
    python -m experiments.architecture_benchmark_001 status

Native random-Qwen tests must actually execute in the pinned native environment;
skips are not native parity. Real source qualification additionally requires the exact
checkpoint and all asset hashes. Requalify after code/data/source/runtime changes.
Do not tune on sealed TEST, report a best seed alone, or turn analytic cache savings
into measured speed/memory claims. Preserve failures and use new output directories.
