# Active package initialization: CPU successor

The completed-parent package path checks could not resolve a child imported by
its parent's initializer. A staged CPU successor records the selected compiled
code object, module globals, loader and immutable source/root binding before
execution. A child lookup requires that exact initialization frame on the current
stack and the initializing spec, with current source bytes and unchanged paths.
The parent is not recorded as completed until its initializer returns successfully.
The active record is revoked in `finally`, including failures.

Real inert initializers import children and nested modules. Final five authored
controls pass, reproduced by Root in 1.740 seconds. They cover active-to-completed
transition, failure revocation, path/source mutation, forged active records without
the original frame and repeated loader execution refusal. The final refusal occurs
before source re-execution and preserves the existing completion record. The
intermediate version that re-entered execution is retained as a separate source
variant; reload support is not claimed. [Independent final five controls](evidence/tensor_active_initializer_reviewed_cpu_20261007/independent_review/INDEPENDENT_REVIEW_003.json)
pass in 1.763 seconds, including circular cached-parent imports, active failure
revocation, exact repeat refusal and installation/container restoration.

[Preservation](evidence/tensor_active_initializer_reviewed_cpu_20261007/ROOT_PRESERVATION_001.json)
and [handoff](evidence/tensor_active_initializer_reviewed_cpu_20261007/HANDOFF.md)
retain final, predecessor, fixture and receipt bytes. The frozen loader is
`df3ca0b59a740bbda773a0640d0415ea690dd8cc19b913e2d49248d61604dec5`.
It retains selected source/cache checks, installed context/checker/token and owned
FileFinder cache handling. This source-owned active record permits normal first
initialization; arbitrary modules in sys.modules do not acquire parent authority.
New originless namespaces and native entry remain closed.

Actual hooks, Torch, DLL/native APIs and model work were not executed. Public
sources remain unchanged. CPU frames and inert source execution do not qualify
current runtime closure, bootstrap execution, provider identity or a CUDA run.
Those current evidence slots and the exact tensor registration remain outstanding
before the separate two-tensor diagnostic. Complete scored M0, sustained training,
prior-correct guards and the M0/M1/M2 transfer comparison remain unmeasured.
