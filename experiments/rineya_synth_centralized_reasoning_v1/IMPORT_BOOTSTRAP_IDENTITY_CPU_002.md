# Additive preload module and loader identity checks

Bootstrap002 tightens current additive module validation: module/spec names must
match the selected name, loader must be the exact standard source/extension type,
module and spec share the same loader object, and loader name/path match selected
bytes. Regular-package path/spec roots must agree; ordinary modules cannot carry
an unregistered package path. File/cache/code equivalence and history-false
fields remain separate from these object metadata checks.

Five authored and Root controls pass. [Independent five controls](evidence/import_bootstrap_identity_reviewed_cpu_20261007/independent_review/INDEPENDENT_BOOTSTRAP_IDENTITY002.json)
pass in 0.088 seconds with 22 selected pins unchanged. They execute inert cache/
module checks, reject same-file foreign loaders and name/root mismatches, check
an authored tiny current-main frame and refuse native preload. The main fixture
is not actual worker initialization; package fixtures cover dual-root metadata,
while the downstream original loader retains its initializer-path check.

The frozen002 test lacked predecessor companions at its isolated location.
Independent review preserved that first initialization failure, then used a
recorded temporary mirror with exact frozen001 companions. Candidate functions
and bytes are unchanged. [Preservation](evidence/import_bootstrap_identity_reviewed_cpu_20261007/ROOT_PRESERVATION_002.json)
now separately includes those selected companion bytes: 34 files and 636,463
bytes. The less strict prior helper remains preserved.

No actual psutil preload, worker process, baseline39, DLL/native API, Torch or
CUDA ran. Full selected-reader/qualifier-before-context integration and actual
owned bootstrap still need review. This is source/metadata logic, not native
readiness or owner permission. The three-arm transfer result remains unmeasured.
