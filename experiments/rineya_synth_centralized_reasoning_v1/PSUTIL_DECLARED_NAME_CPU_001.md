# Exact observed psutil extension-name correction

The preload verifier now accepts the actual declared name `psutil_windows` only
for import namespace `psutil._psutil_windows`, the selected physical
`SITE/psutil/_psutil_windows.pyd`, and its exact70,656-byte descriptor/hash.
Spec/loader full namespace, origin, package/cache shape and byte checks remain
unchanged. Other modules still require their full declared names.

Five authored/Root and independent CPU controls pass using the actual observed
metadata as an inert fixture. Altered names, namespaces, paths, binary descriptors,
loaders and unrelated aliases reject. [Preserved review](evidence/psutil_declared_name_reviewed_cpu_20261009/ROOT_PRESERVATION.json)
retains the exact source, predecessor and receipts. The underlying actual
parent observation is separately preserved; these CPU checks do not reload the
extension, import Torch or establish native worker/transfer success.

Refresh current registration/bytes and continue the next separate import attempt
under standing authorization. Original resource, child isolation and historical
preload/DLL maps remain intact.
