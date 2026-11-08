# RAB-001 exploratory diagnostics — deliberately OUTSIDE the hashed experiment root

`gates.implementation_manifest()` globs `ROOT.rglob('*.py')` where `ROOT` is
`experiments/architecture_benchmark_001/`. Any `.py` placed there changes
`implementation_digest()` and invalidates every receipt bound to it.

These scripts are therefore kept in a sibling directory. They import the experiment package
read-only, produce no receipts, and authorize no stage. Run them from the repository root
with the pinned environment:

    PY=C:/Users/clclc/BusinessProject/.venv-rab001/Scripts/python.exe
    $PY experiments/rab001_exploration/arm_quality.py 156

| script | what it measures |
|---|---|
| `arm_quality.py` | teacher-forced CE, perplexity, KL-to-source and top-1 agreement per arm |
| `arm_generate.py` | greedy generation on the open pilot split per arm |
| `divergence.py` | per-encoder-layer and shared-bank chunk divergence for a CED checkpoint |
| `divergence_ctrl.py` | the same per-layer probe on the unmodified official B28 control |

Results are committed under `experiments/architecture_benchmark_001/evidence/`.
