# Completed validation and limits

## Scope

Operational successor `codex/model-quality-audit-v2`, parent `8c3ed88a821b83e531ab682b3201879ea4fc928b`. Tests ran with the existing `.venv-mq001` Python, `CUDA_VISIBLE_DEVICES=''` and `OMP_NUM_THREADS=1`. No official source weights were loaded, no GPU call or optimization was launched, and no scientific cell ran in this iteration. Tiny fixture optimizer updates are CPU mechanics.

## Completed tests

| Run | Result | Evidence |
|---|---|---|
| Historical MQ subject, complete scoped suite |94 passed, 1 skipped; 45.869 s|`mq-historical-tests.xml`|
| Successor mechanics and supervisor suite |111 passed, 1 skipped; 16.81 s|`mq-ops-tests-reviewed.log`, `.xml`|
| Successor optional random native-Qwen CPU cases |2 passed; 8.65 s|`mq-successor-native-random.xml`|

Together the two successor invocations cover all 114 scoped cases: **113 passed, 1 skipped**. The skip is the inherited Windows symlink-creation privilege case. Both random native-Qwen cases executed with the pinned Transformers runtime; they do not establish actual-checkpoint native qualification. Initial failed attempts are preserved: Windows venv child/interpreter identity and test selection of WSL instead of Git Bash were exposed and corrected, not omitted from history. The intermediate rerun records are retained as well.

The tests exercise public imports without Torch/NumPy/Transformers/safetensors/psutil; direct worker refusal before output/native import; one-use context, argument/parent/output/working-directory mismatch; Windows virtual-environment redirection; startup and post-spawn failures; nonzero child exit; timeout/interrupt; a real grandchild tree and surviving unrelated control; preservation of checkpoint bytes; create-only retry rejection; refusal of both obsolete wrappers. Existing masks, EOS, loss values/gradients, Adam parity, schedule boundaries, RNG/resume, source/plan/checkpoint mismatch, frozen tensors, update-group matching and final partial groups remain covered by the fixture suite.

## Invariants and evidence integrity

The source lock and scientific files `config.py`, `data.py`, `losses.py`, `optimizer.py`, `engine.py`, `native.py`, `checkpoints.py` and `evaluation.py` match the immutable parent. Only supervision/lazy imports, branch binding and retired wrapper behavior change. Resource minima remain qualification4.6/3.0 GiB host/GPU and training6.0/6.0 GiB. The qualification comment now acknowledges backward checks.

Historical source/code verification: both recorded MQ code digests reproduce from their exact Git commits; local Supa and Opallo weights hash to the recorded values (`mq-provenance-recheck.json`). All eight retained result sets match their referenced baseline/final/step files. The create-only archive preserves 113 non-weight files byte-for-byte; every archive member was checked. Eight prior document/wrapper Git blobs are preserved with hashes under `docs/history/model-quality-audit-2026-09-15`.

`successor-code-identity.json` binds final package bytes after documentation changes; evidence outputs themselves are excluded by the established digest rule. `ARTIFACT_INDEX.json` lists byte hashes of delivered audit records. Temporary recomputation scripts record how the offline audit was produced; their original absolute paths deliberately refer to the inspected historical checkout and scratch output. They are not native run commands or a reviewed training corpus.

## Limits

No fresh official-source/native GPU qualification, maximum-shape fit or official checkpoint resume has been run on this successor. The child context prevents accidental bypass, not hostile local environment forgery. Normal supervised timeout/interrupt cleanup was tested; forced supervisor termination is not an OS Job Object guarantee. If tree cleanup fails, the attempt remains failed and records cleanup notes; direct-child exit alone is not proof every descendant exited.

Six historical corrected cells lacked watchdog receipts and all eight historical checkpoint directories are absent. Stored outputs are useful development records; they cannot prove current resumability or certify a regenerated checkpoint. Corpus/semantic evaluator and scientific confirmation gates remain closed. No model-quality or release permission follows from these tests.
