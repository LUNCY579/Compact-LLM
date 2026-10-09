# Parent future-import correction

The import supervisor now resolves `__future__` through its exact already
inventoried stdlib source, compiling checked source bytes rather than an
unselected cache. This adds only the two necessary dependency/loader tuple
entries. The child bootstrap, resource rules, original supervisor and monitor
remain unchanged; both dispatcher supervisor descriptors are updated exactly.

Fourteen authored/Root and independent CPU controls pass, including fresh
resolution of the actual future import while executing the full original
supervisor with inert native dependencies. [Preserved review](evidence/import_parent_future_reviewed_cpu_20261009/ROOT_PRESERVATION.json)
retains the exact source and receipts. This is CPU evidence, not a native import
or host-fit result.

The actual007 failure remains retained. Refresh current source/runtime evidence,
bind a separate successor dispatch and continue under standing authorization.
M0/M1/M2 transfer and prior-correct preservation remain unmeasured.
