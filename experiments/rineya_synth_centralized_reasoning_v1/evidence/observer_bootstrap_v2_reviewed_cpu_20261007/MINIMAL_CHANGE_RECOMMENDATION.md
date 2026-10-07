# Finite observer startup correction proposal

CPU-only source/filesystem evidence, not namespace execution proof or native authority.

Original observer003 unknown rows: six source modules and `pywin32_system32` with null origin. The unchanged worker rejects all seven before observer binding/fixture work. Original failure bytes must stay unchanged.

The companion finite_bootstrap_candidate.json binds 14 files: six modules (python_source), six immediate .pth hooks (prospective startup_hook_source), and two namespace DLL leaves (extension_binary). Total 844,093 bytes. The candidate's namespace search_locations are source/filesystem-derived and were NOT measured in observer003.

## Minimum consumer changes

1. Preserve allowed_bootstrap_modules schema for source and extension rows; add its six exact module leaves. Do not allow arbitrary mapped editable package roots.
2. Extend cache_policy with startup_hook_sources and allowed_namespace_packages. startup_hook_sources should carry one exact physical site-packages directory and its complete immediate six .pth descriptor map. Validate a fresh files-only .pth roster and bytes; extra/missing hooks refuse. Reject aliases/reparse points.
3. allowed_namespace_packages is exactly pywin32_system32, with ordered singleton physical search_locations, complete immediate files-only descriptor map and its canonical digest. Refuse __init__.py, any extra file/directory, changed/missing DLL or alternate/extra search path. No arbitrary namespace-name allowlist.
4. Keep shared prefix_tensor_contract.runtime_role unchanged. Observer contract._shared_inventory_checks needs an exact observer-only role exception for this declared startup leaf set (.py, .pth and the namespace two DLLs), followed by original runtime_role for everything else. _schema needs startup_hook_source -> .pth; all new roles must remain explicit no-model assets.
5. Worker _native_bootstrap: a namespace_package branch only if the exact selected name has origin and __file__ both None, __spec__.origin is None, has_location false, and nonempty ordered __spec__.submodule_search_locations and module.__path__ agree with registration. Record both independently captured path lists and fresh namespace file roster/digest; file_descriptor verifies each selected DLL. Unknown originless modules stay unknown. Keep namespace_execution_proven false.
6. Audit replay_startup: require exact namespace row keys, path equality, roster bytes/digest and registered runtime joins. Preserve existing source_or_extension and builtin_or_frozen branches verbatim in behavior. An old five-key null-origin row cannot be reinterpreted as successful namespace evidence.
7. CPU controls: correct authored namespace succeeds; extra search path, spec/module path disagreement, changed file, added DLL/subdirectory/__init__.py, hidden error, missing namespace declaration, unknown originless module, extra .pth, changed hook byte and changed module bytes each refuse. No authored control can issue native authority.
8. Rebound config/runtime/registration/review and fresh external anchor must be distinct prospective packets under a new implementation commit then a distinct dispatch HEAD. Preserve ABI, resources, permission scope, imports and 12 operation sequence.

## Local startup provenance

The three editable hook files execute their corresponding finder install() function. That installs finder classes; their mapping target modules are not imported at installation. _virtualenv.pth imports _virtualenv (meta-path patch). distutils-precedence.pth conditionally imports _distutils_hack and installs shim. pywin32.pth adds win32, win32/lib, Pythonwin search paths then imports pywin32_bootstrap. Its top-level try imports pywin32_system32 namespace and adds the first physical namespace directory to DLL search. It does not load namespace DLLs itself.

No global virtualenv/pth edits, uninstallation or protected/model asset reads were performed. Initial inspector interpreter invocation used -I -B and therefore did not suppress automatic site startup processing; subsequent file/AST inspection uses -I -S -B. No task source imports or DLL loads were explicitly executed by the inspection script. This limitation is preserved in the candidate; it must not be represented as a sterile -S initial inspection.
