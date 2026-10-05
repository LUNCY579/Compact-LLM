# Interpreter module classification repaired

The [source loader revision](evidence/frozen_bootstrap_loader_reviewed_cpu_20261005/prefix_registered_loader.py)
recognizes validated built-in/frozen importer identity before inspecting a
module's documentary source filename. Genuine frozen modules can expose
`__file__` without executing that file as cached Python source. The loader now
checks the module type, spec name, exact importer and independent interpreter
recognition; counterfeit origin/loader/name, unqualified fileless modules and
preloaded Torch still fail before source or cache access.

Root passed five new inert controls and19 existing contract/loader regressions.
The [independent24-control review](evidence/frozen_bootstrap_loader_reviewed_cpu_20261005/frozen_bootstrap_loader_review_1791237342636262800.json)
passes on unchanged selected source pins with no host runtime/model/cache reads
or native/framework imports. Previous loader bytes and lineage are retained.
Root also reproduced all24 controls from the
[preserved copy](evidence/frozen_bootstrap_loader_root_replay_20261005.json)
in0.393 seconds with unchanged source hashes. Four older workflow sources are
included solely to reproduce the metadata fixtures, with their auxiliary scope
recorded separately; this does not requalify their execution.

This verifies classification metadata only. It does not prove an existing
module namespace executed the selected interpreter code. Current interpreter
bytes, bootstrap assumptions and public native orchestration remain unqualified;
no native gate or study permission changes.
