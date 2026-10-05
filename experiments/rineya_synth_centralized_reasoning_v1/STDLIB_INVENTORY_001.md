# Finite interpreter and standard-library byte inventory

`stdlib_inventory.py` prepares a CPU-only executable-file superset for the pinned
Hermes Python3.11.15 runtime. It includes regular Lib Python sources excluding
site-packages/cache directories, DLLs Python extensions/runtime libraries,
interpreter-root executables/libraries/ZIPs and the executing Hermes launcher.
It records optional python311.zip presence or absence and includes valid empty
Python sources with the SHA256 of an empty file. Binary files must be nonempty.

The observed runtime has779 files totaling36,184,313 bytes:738 Python sources,
34 DLLs files, six interpreter-root files and one Hermes launcher. Five Python
sources are empty; the optional ZIP is absent. The exact known Windows junction
from the logical interpreter base to its versioned physical directory is bound
by target, attributes and tag. Files are hashed inside the regular physical
directory; other symlink/reparse/escape rules remain closed. Initial rejection
of the logical junction is retained with its explicitly reviewed successor.

Candidates bind roots, version, executable, roster, source and file hashes across
two full passes. Independent current-byte reconstruction also passes. This
inventories bytes without loading encoder, framework, model or GPU packages.
It does not claim complete machine trust or authenticate arbitrary site packages.
Windows system/driver dependencies retain their inherited trust and pins.

The [root12 controls](evidence/stdlib_inventory_root_cpu_validation_20261005.json)
pass with unrelated parent tokenizer imports present. Tests cover omissions,
extras, same-count substitutions, file/version/root/alias changes, optional ZIP,
empty sources, empty binaries, roster races and unchanged isolated import gates.
The [reviewed history](evidence/stdlib_inventory_reviewed_cpu_20261005)
retains the actual candidate, independent byte review, initial junction rejection
and fresh-root controls. Candidate and fixture outputs issue no native authority.

The additive encoder-runtime registration, contract and worker consumption are
still unimplemented. They must preserve the encoder45 and inherited65 maps,
merge these pins without conflicting overwrites, bind policy/alias/roster into
the native recipe, verify actual bytes before imports and after math, and join
the independent terminal audit. The incomplete encoder admission gate remains
closed. No final native registration, config or model run was created.
