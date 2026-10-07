# Registered package paths: CPU successor

The earlier top-level startup integration rejected cache entries created for a
parent package's `__path__`, because those directories are absent from `sys.path`.
A staged CPU successor now permits only search directories derived from an actual
registered parent package. Inert imports exercise a package, child, nested regular
package and deep module through RegisteredFinder and RegisteredLoader.

The loader captures immutable parent source origin, descriptor and root at
construction, then checks the original module/spec/loader, recorded source
execution and current bytes before resolving children. The request's exact roots
and previously owned cache keys bound the FileFinder population. Dotted-name
resolution without parent paths, rebound modules/paths, altered source and foreign
cache mutations refuse. The exact preloaded pywin32 namespace remains a separate
metadata exception; newly imported originless namespaces refuse.

Final four authored controls pass, reproduced by Root in 1.243 seconds. Original
failed fixture variants are retained: one lacked namespace spec name, another
omitted the real built-in sys module needed by NamespacePath. Corrections make
the tests reach the intended negative predicates. The earlier top-level eight and
independent seven results are not relabeled as full successor regressions.
[Independent final four controls](evidence/tensor_package_paths_reviewed_cpu_20261007/independent_review/INDEPENDENT_REVIEW_006.json)
pass in 1.191 seconds, including coherent parent rebinding, source/execution-record
mutation and foreign package-cache refusal.

[Preservation](evidence/tensor_package_paths_reviewed_cpu_20261007/ROOT_PRESERVATION_001.json)
and [handoff](evidence/tensor_package_paths_reviewed_cpu_20261007/HANDOFF.md) retain
exact sources, predecessors, fixtures and receipts. The frozen loader is
`47f8633b2b1d8ca153eba84e9ee3572ed15f5ec38210258ed0eca2cbc3b29f06`.
Native entry remains closed, public contracts are unchanged, and actual hooks,
framework/DLL/native APIs and model work did not run. Inert source execution does
not qualify actual Torch package closure, current bootstrap or a live provider.

The full amended native manifest reader, exact current source/runtime/cache/library
packet, live startup provider and separate tensor permission remain absent. The
pure registration Git join is reviewed independently and cannot fill those slots.
M0 completion/scoring, native sustained training, qualified guards and the actual
three-arm transfer/retention comparison remain outstanding.
