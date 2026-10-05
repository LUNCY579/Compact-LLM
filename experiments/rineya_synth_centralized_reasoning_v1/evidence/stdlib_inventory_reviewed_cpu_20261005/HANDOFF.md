# Finite interpreter/stdlib byte superset — CPU preparation

The helper inventories a declared source-execution scope, not complete machine
trust. It enumerates every regular Lib .py source excluding site-packages and
__pycache__, every DLLs .pyd/.dll, every immediate interpreter-root .exe/.dll/.zip,
and the actually executed Hermes launcher outside that base. It records the
optional python311.zip explicitly as present or absent. Empty Python sources
are included with zero bytes and SHA256-empty; executable binaries must be
nonempty. Versions, roots, executable, roster, file bytes and helper source are
explicit and compared before/after; no encoder/framework/model/GPU executes.

The actual sys.base_prefix is a known Windows directory junction. Exactly the
logical cpython-3.11-windows-x86_64-none path, its versioned physical target
cpython-3.11.15-windows-x86_64-none, attributes0x410 and tag0xa0000003 are bound.
Only regular files inside that physical target are enumerated/hashed; all other
symlink/reparse/escape rules remain closed. The logical alias and actual Hermes
venv Scripts/python.exe identity are separately bound. Changing the alias, tag,
target, version, file pointer or roster rejects. The initial general-reparse
rejection and original sources/tests were preserved before this explicit layout
policy. This is not a generic reparse allowance or a runtime relocation permit.

The observed inventory is779 files/36,184,313 bytes:738 Lib Python sources,
34 DLLs extension/runtime files,6 interpreter-root executables/runtime binaries,
and1 executed Hermes launcher. All5 actual empty Python sources are bound.
Optional stdlib ZIP is absent. This is an executable superset with unused test/
optional sources deliberately included; it does not infer arbitrary dynamic
imports or authorize optional packages outside that declared superset.

Synthetic tests cover omissions/extras, same-count substitutions, changed bytes,
version/root/executable/source pointers, symlink/escape, alias-target/tag drift,
optional ZIP changes and roster races. Fixture tests mock only no_execution
locally so prior unrelated test imports do not contaminate the fixtures; an
isolated -I -B child proves the unchanged production import gate and rejects an
injected forbidden module before any roster or file lookup. CPU fixtures cannot
issue native PASS or claim real OS resource/model qualification.

The proposed additive encoder-runtime contract is a separately reviewed V2
registration with the existing exact encoder package45 map preserved and an
independently frozen stdlib_registration descriptor/file SHA. It would validate
this exact finite policy/version/alias/executable/roster, merge inherited65,
encoder45 and these executable pins with no overwrite of conflicting hashes,
and place all maps and policy hashes inside the source-bound native recipe.
The admitted worker would hash every added regular file before imports and
after math, verify optional ZIP absence and exact alias/roster stability, and
publish the same actual maps in its immutable capsule. The independent auditor
would re-read those externally selected registration/source/runtime byte maps.

No V2 contract or worker edit is implemented here. No final native registration
or config is written; the currently incomplete encoder prerequisite stays closed
until that additive schema, consumption and audit are reviewed. Windows system
libraries/driver/NVML dependencies retain their separately inherited trust/pins;
this finite interpreter scope is not a claim to authenticate the whole machine,
arbitrary site packages, native fit, historical failure cause, study, scoring,
guard enrollment or new launch authority.
