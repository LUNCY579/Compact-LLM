"""Finite DLL boundary used by the registered extension loader.

No OS calls occur at import. CPU suppliers execute this exact guard with inert
selected files. A native provider needs the separate native gate and a reviewed
fresh-bootstrap trace: Windows cannot enumerate other AddDllDirectory cookies.
Managed cookie state is therefore never represented as an independent complete
OS search-directory observation. Loaded-module paths/handles come from WinAPI;
their current bytes are checked separately against selected descriptors.
Load/unload event completeness requires a separately source-qualified native
observer, absent in the current contract. This guard checks selected prospective
dependencies before delegation and refuses unregistered observed execution; it
does not establish a preventive OS sandbox or inventory all System32 candidates.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import stat

KIND = 'FINITE_WINDOWS_DLL_BOUNDARY_V1'
DEFAULT_FLAGS = 0x00000400 | 0x00000800  # USER_DIRS | SYSTEM32
MAX_LIBRARIES = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _physical(path, directory=False):
    _require(type(path) is str and path and str(Path(path).absolute()) == path,
             'exact absolute selected library path')
    selected = Path(path)
    _require(selected.resolve() == selected, 'physical selected library path')
    for item in (selected, *selected.parents):
        metadata = item.lstat()
        _require(not item.is_symlink() and not (getattr(metadata, 'st_file_attributes', 0) & 0x400),
                 'library boundary refuses reparse aliases')
    metadata = selected.lstat()
    _require(stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode),
             'regular selected library directory/file')
    return selected


def validate_policy(policy):
    """Validate selected intent; this does not establish observed resolution."""
    _require(type(policy) is dict and set(policy) == {
        'kind', 'search_directories', 'system_directory', 'library_files',
        'initial_loaded_libraries', 'extension_transitions',
        'framework_transition', 'provider_source'}, 'exact enforced DLL boundary schema')
    _require(policy['kind'] == KIND, 'selected finite DLL boundary kind')
    directories = policy['search_directories']
    _require(type(directories) is list and 0 < len(directories) <= 128 and
             all(type(p) is str for p in directories) and len(set(directories)) == len(directories),
             'finite unique selected DLL search directories')
    _require(type(policy['system_directory']) is str and policy['system_directory'] in directories,
             'selected system directory belongs to finite DLL search directories')
    files = policy['library_files']
    _require(type(files) is dict and 0 < len(files) <= MAX_LIBRARIES, 'finite selected library identity map')
    for path, descriptor in files.items():
        _require(type(path) is str and type(descriptor) is dict and
                 set(descriptor) == {'bytes', 'sha256', 'role'} and
                 type(descriptor['bytes']) is int and 0 <= descriptor['bytes'] <= 2**31 and
                 type(descriptor['sha256']) is str and len(descriptor['sha256']) == 64 and
                 all(c in '0123456789abcdef' for c in descriptor['sha256']) and
                 descriptor['role'] in {'extension_binary', 'native_library', 'interpreter_executable',
                                        'torch_binary', 'interpreter_binary', 'windows_system_binary'},
                 'exact selected dependent library descriptor')
        _require(str(Path(path).absolute()) == path and Path(path).suffix.lower() in {'.dll', '.pyd', '.exe'} and
                 str(Path(path).parent) in directories,
                 'selected dependent library lies in an exact search directory')
    def paths(values):
        _require(type(values) is list and len(values) <= MAX_LIBRARIES and
                 all(type(p) is str and p in files for p in values) and len(set(values)) == len(values),
                 'finite exact library transition path list')
    paths(policy['initial_loaded_libraries'])
    _require(policy['initial_loaded_libraries'], 'nonempty observed bootstrap library obligation')
    paths(policy['framework_transition'])
    extensions = policy['extension_transitions']
    _require(type(extensions) is dict and len(extensions) <= MAX_LIBRARIES,
             'finite extension library transition map')
    for path, transition in extensions.items():
        _require(path in files and Path(path).suffix.lower() == '.pyd' and
                 files[path]['role'] in {'extension_binary', 'torch_binary'} and type(transition) is dict and
                 set(transition) == {'create', 'exec'}, 'exact extension DLL transition obligations')
        paths(transition['create']); paths(transition['exec'])
    source = policy['provider_source']
    _require(type(source) is dict and set(source) == {'path', 'descriptor'} and
             type(source['path']) is str and Path(source['path']).suffix == '.py' and
             type(source['descriptor']) is dict and set(source['descriptor']) == {'bytes', 'sha256', 'role'} and
             source['descriptor']['role'] == 'python_source', 'exact selected library provider source pin')
    return policy


def _provider_capability_factory():
    token = object()
    def capability():
        return token
    def check(candidate):
        _require(candidate is token, 'private CPU library supplier capability')
    return capability, check


_CPU_library_capability, _check_CPU_library_capability = _provider_capability_factory()


class _SupplierProvider:
    """Adapter around explicit observation/API suppliers; never native evidence."""
    fixture_only = True
    native_PASS = False
    native_authority_issued = False
    def __init__(self, supplier, capability):
        _check_CPU_library_capability(capability)
        self.supplier = supplier
        self.capability = capability
    def _call(self, name, *args):
        _check_CPU_library_capability(self.capability)
        return getattr(self.supplier, name)(*args)
    def begin(self, directories, system_directory):
        return self._call('begin', directories, system_directory)
    def scope(self):
        return self._call('scope')
    def libraries(self):
        return self._call('libraries')
    def events(self):
        return self._call('events')
    def close(self):
        return self._call('close')


class LibraryGuard:
    """Observe/check before and after every delegated native boundary.

    A failure poisons the guard even if caught by imported Python code. No
    completion or native claim can be emitted before successful cleanup.
    """
    def __init__(self, policy, contract, provider, budget=None):
        self.policy = contract.parse(contract.canonical(validate_policy(policy)))
        self.contract = contract
        self.provider = provider
        self.budget = budget
        self.fixture_only = provider.fixture_only
        self.native_PASS = False
        self.native_authority_issued = False
        self.active = False
        self.failed = False
        self.closed = False
        self.last = None
        self.observed = []
        self._depth = 0
        self._frames = []
        self._directory_identities = None
        self._event_log = []
        self._event_state = None
        self._source_pin()
        for directory in self.policy['search_directories']:
            _physical(directory, directory=True)
        self._directories()
        try:
            provider.begin(tuple(self.policy['search_directories']), self.policy['system_directory'])
            self.active = True
            self._scope()
            current = self._snapshot()
            _require(set(current) == set(self.policy['initial_loaded_libraries']),
                     'exact complete initial observed loaded-library identities')
            self.last = current
            self._event_state = dict(current)
            self._events(allowed_new=set())
        except BaseException:
            self.failed = True
            # begin can allocate cookies before reporting failure.
            self._cleanup_after_failure()
            raise

    def _source_pin(self):
        source = self.policy['provider_source']
        _require(source['path'] == str(Path(__file__).absolute()), 'actual shared library guard source identity')
        compiled = globals().get('__compiled_source_descriptor__')
        _require(compiled is not None and compiled == source['descriptor'],
                 'actual compiled library guard consumes selected provider source')
        self.contract.file_descriptor(source['path'], source['descriptor'], budget=self.budget)

    def _scope(self):
        scope = self.provider.scope()
        _require(type(scope) is dict and set(scope) == {
            'kind', 'complete', 'directories', 'system_directory', 'default_flags',
            'legacy_directory', 'unmanaged_directory_registration', 'completeness_basis', 'closed'},
            'complete actual DLL search-scope observation schema')
        expected_basis = 'explicit_injected_CPU_observation' if self.fixture_only else 'Win32_calls_and_verified_fresh_bootstrap_trace'
        _require(scope['kind'] == 'OBSERVED_DLL_DIRECTORY_SCOPE' and scope['complete'] is True and
                 scope['directories'] == self.policy['search_directories'] and
                 scope['system_directory'] == self.policy['system_directory'] and
                 scope['default_flags'] == DEFAULT_FLAGS and scope['legacy_directory'] == '' and
                 scope['unmanaged_directory_registration'] is False and
                 scope['completeness_basis'] == expected_basis and scope['closed'] is False,
                 'actual finite DLL search scope changed/incomplete')
        return scope

    def _directories(self):
        """Check selected physical directories; do not assert full OS inventory."""
        identities = {}
        for directory in self.policy['search_directories']:
            root = _physical(directory, directory=True)
            metadata = root.lstat()
            identities[directory] = (metadata.st_dev, metadata.st_ino)
            if self.budget is not None:self.budget.check()
        if self._directory_identities is not None:
            _require(identities == self._directory_identities,
                     'selected DLL search directory physical identities changed')
        self._directory_identities = identities

    def _events(self, allowed_new):
        observation = self.provider.events()
        _require(type(observation) is dict and set(observation) == {'kind', 'complete', 'events'} and
                 observation['kind'] == 'OBSERVED_PROCESS_LIBRARY_EVENTS' and observation['complete'] is True and
                 type(observation['events']) is list and len(observation['events']) <= 16 * MAX_LIBRARIES,
                 'complete actual library load/unload event observation required')
        rows = observation['events']
        _require(rows[:len(self._event_log)] == self._event_log,
                 'append-only actual library event trace conservation')
        state = dict(self._event_state)
        for index, row in enumerate(rows):
            _require(type(row) is dict and set(row) == {'sequence', 'event', 'path', 'module_handle'} and
                     type(row['sequence']) is int and row['sequence'] == index + 1 and
                     row['event'] in {'load', 'unload'} and type(row['path']) is str and
                     type(row['module_handle']) is int and row['module_handle'] > 0,
                     'ordered bounded actual library load/unload event schema')
            if index < len(self._event_log):continue
            path = row['path']
            _require(path in self.policy['library_files'], 'unregistered transient dependent library event observed')
            _require(row['event'] == 'load', 'library unload violates qualified lifetime conservation')
            _require(path in allowed_new and path not in state and row['module_handle'] not in state.values(),
                     'unplanned actual library load event before native/completion claim')
            self.contract.file_descriptor(path, self.policy['library_files'][path], budget=self.budget)
            state[path] = row['module_handle']
        self._event_log = self.contract.parse(self.contract.canonical(rows))
        self._event_state = state
        return state

    def _snapshot(self):
        snapshot = self.provider.libraries()
        _require(type(snapshot) is dict and set(snapshot) == {'kind', 'complete', 'libraries'} and
                 snapshot['kind'] == 'OBSERVED_PROCESS_LIBRARY_SNAPSHOT' and snapshot['complete'] is True and
                 type(snapshot['libraries']) is list and 0 < len(snapshot['libraries']) <= MAX_LIBRARIES,
                 'complete actual loaded-library snapshot required')
        result = {}
        handles = set()
        for row in snapshot['libraries']:
            _require(type(row) is dict and set(row) == {'path', 'module_handle'} and
                     type(row['path']) is str and type(row['module_handle']) is int and
                     row['module_handle'] > 0 and row['path'] not in result and row['module_handle'] not in handles,
                     'unique actual loaded-library path/handle identities')
            path = row['path']
            _require(path in self.policy['library_files'], 'unregistered dependent library observed')
            _physical(path)
            self.contract.file_descriptor(path, self.policy['library_files'][path], budget=self.budget)
            result[path] = row['module_handle']; handles.add(row['module_handle'])
        return result

    def _check_active(self):
        _require(self.active and not self.failed and not self.closed, 'active nonfailed DLL boundary required')
        self._source_pin(); self._scope(); self._directories()
        observed = self._snapshot()
        if observed != self.last and self._frames and self._frames[-1]['name'] == 'framework_import':
            frame = self._frames[-1]
            appeared = set(observed) - set(self.last)
            _require(all(observed.get(path) == handle for path, handle in self.last.items()) and
                     appeared <= set(frame['expected']) - frame['directly_observed'],
                     'loaded-library conservation before nested native delegation')
            frame['directly_observed'].update(appeared)
            self.last = observed
            self._events(allowed_new=appeared)
        else:
            self._events(allowed_new=set())
        _require(self._event_state == observed, 'actual library event/snapshot identity join')
        _require(observed == self.last, 'loaded-library conservation before native delegation')
        return observed

    @contextmanager
    def _boundary(self, name, expected_new):
        before = None
        try:
            for path in expected_new:
                _physical(path)
                self.contract.file_descriptor(path, self.policy['library_files'][path], budget=self.budget)
            before = self._check_active()
            self._depth += 1
            frame = {'name': name, 'expected': list(expected_new), 'directly_observed': set()}
            self._frames.append(frame)
            yield
            _require(not self.failed, 'poisoned native library boundary cannot return success')
            self._source_pin(); self._scope(); self._directories()
            after = self._snapshot()
            allowed = set(expected_new)
            self._events(allowed_new=allowed)
            _require(self._event_state == after, 'actual library event/snapshot identity join')
            _require(all(after.get(path) == handle for path, handle in before.items()),
                     'existing loaded-library identity conservation after native delegation')
            _require(all(after.get(path) == handle for path, handle in self.last.items()),
                     'nested verified loaded-library identity conservation')
            # Nested extension guards update last; framework adds only its own
            # direct libraries beyond those already verified by nested guards.
            acknowledged = set(self.last) - set(before) - frame['directly_observed']
            _require(set(after) - set(before) - acknowledged == set(expected_new) and
                     not (set(expected_new) & set(before)),
                     'exact observed dependent-library transition after native delegation')
            self.last = after
            self.observed.append({'boundary': name, 'new_libraries': sorted(set(after) - set(before)),
                                  'fixture_only': self.fixture_only,
                                  'actual_loaded_identity_observed': True,
                                  'native_qualification_proven': False})
        except BaseException:
            self.failed = True
            if before is not None:
                # A native delegate can fail after loading libraries. Retain a
                # mandatory post-failure observation without rescuing the import.
                try:
                    self._source_pin(); self._scope(); self._directories()
                    after_failure = self._snapshot()
                    self._events(allowed_new=set(expected_new))
                    self.observed.append({'boundary': name, 'new_libraries': sorted(set(after_failure) - set(before)),
                                          'fixture_only': self.fixture_only,
                                          'actual_loaded_identity_observed': True,
                                          'native_qualification_proven': False,
                                          'delegated_boundary_failed': True})
                except BaseException as observation_failure:
                    self.observed.append({'boundary': name, 'fixture_only': self.fixture_only,
                                          'actual_loaded_identity_observed': False,
                                          'native_qualification_proven': False,
                                          'delegated_boundary_failed': True,
                                          'post_failure_observation_complete': False,
                                          'observation_error_type': type(observation_failure).__name__})
            raise
        finally:
            if before is not None:
                self._depth -= 1
                self._frames.pop()

    def extension(self, path, operation):
        if operation not in {'create', 'exec'} or path not in self.policy['extension_transitions']:
            self.failed = True
            raise ValueError('selected actual extension DLL boundary before delegation')
        return self._boundary('extension:' + path + ':' + operation,
                              self.policy['extension_transitions'][path][operation])

    def framework_import(self):
        return self._boundary('framework_import', self.policy['framework_transition'])

    def poison(self, reason):
        self.failed = True
        self.observed.append({'boundary': reason, 'fixture_only': self.fixture_only,
                              'native_qualification_proven': False,
                              'loader_refusal': True})

    def _cleanup_after_failure(self):
        try:
            cleanup = self.provider.close()
            _require(type(cleanup) is dict and cleanup == {
                'kind': 'OBSERVED_DLL_DIRECTORY_CLEANUP', 'complete': True,
                'remaining_managed_directories': [], 'legacy_directory': '',
                'all_cookies_removed': True, 'closed': True},
                'complete failed-start DLL search-scope cleanup observation required')
        except BaseException as cleanup:
            raise ValueError('DLL search-scope cleanup failed; native/completion claims forbidden') from cleanup
        finally:
            self.active = False

    def close(self):
        _require(not self.closed, 'DLL search-scope cleanup exactly once')
        _require(self._depth == 0, 'cannot cleanup during delegated library boundary')
        primary = None
        try:
            if not self.failed:
                self._check_active()
        except BaseException as failure:
            self.failed = True
            primary = failure
        try:
            cleanup = self.provider.close()
            _require(type(cleanup) is dict and cleanup == {
                'kind': 'OBSERVED_DLL_DIRECTORY_CLEANUP', 'complete': True,
                'remaining_managed_directories': [], 'legacy_directory': '',
                'all_cookies_removed': True, 'closed': True},
                'complete actual DLL search-scope cleanup observation required')
            try:
                self._source_pin()
                self._events(allowed_new=set())
                final = self._snapshot()
                _require(self._event_state == final and final == self.last,
                         'final cleanup library event/snapshot lifetime conservation')
            except BaseException as observation_failure:
                if not self.failed:raise
                self.observed.append({'boundary': 'failed_boundary_final_cleanup',
                                      'fixture_only': self.fixture_only,
                                      'post_cleanup_observation_complete': False,
                                      'native_qualification_proven': False,
                                      'observation_error_type': type(observation_failure).__name__})
        except BaseException as failure:
            self.failed = True
            raise ValueError('DLL search-scope cleanup failed; native/completion claims forbidden') from failure
        finally:
            self.active = False
            self.closed = True
        if primary is not None:
            raise primary
        _require(not self.failed, 'failed DLL boundary cannot publish completion')
        return {'kind': 'DLL_BOUNDARY_CLEANUP_VERIFIED', 'fixture_only': self.fixture_only,
                'native_PASS': False, 'native_authority_issued': False,
                'native_qualification_proven': False, 'observed_boundaries': list(self.observed)}


def _CPU_make_guard(policy, contract, supplier, *, _capability, budget=None):
    _check_CPU_library_capability(_capability)
    return LibraryGuard(policy, contract, _SupplierProvider(supplier, _capability), budget)


class Win32LibraryProvider:
    """Concrete WinAPI provider; construction is forbidden by current stage gate.

    The supplied independently reviewed bootstrap verifier must bind this fresh
    process, this provider source and an interception/audit trace covering every
    DLL-directory registration from process creation through final cleanup. A
    bool or policy declaration is not accepted. It runs before importing ctypes.
    Trace observation is necessary because WinAPI cannot list foreign cookies.
    No current host, bootstrap trace or native qualification exists here.
    """
    fixture_only = False
    native_PASS = False
    native_authority_issued = False
    def __init__(self, contract, bootstrap_verifier=None, library_event_observer=None):
        contract.stage_native_gate()
        _require(callable(bootstrap_verifier), 'independently verified current fresh-bootstrap DLL-directory trace required')
        qualifier = getattr(contract, 'verify_native_library_event_observer', None)
        _require(callable(qualifier) and library_event_observer is not None,
                 'separately source-qualified complete native library event observer required')
        qualifier(library_event_observer)
        self.library_event_observer = library_event_observer
        self.bootstrap_verifier = bootstrap_verifier
        self._bootstrap('before_ctypes')
        import sys
        _require(all(name in sys.modules for name in ('ctypes', 'ctypes.wintypes', '_ctypes')),
                 'ctypes namespaces must be externally qualified preloaded bootstrap sources before provider construction')
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes; self.wintypes = wintypes
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.kernel.SetDefaultDllDirectories.argtypes = [wintypes.DWORD]
        self.kernel.SetDefaultDllDirectories.restype = wintypes.BOOL
        self.kernel.AddDllDirectory.argtypes = [wintypes.LPCWSTR]
        self.kernel.AddDllDirectory.restype = ctypes.c_void_p
        self.kernel.RemoveDllDirectory.argtypes = [ctypes.c_void_p]
        self.kernel.RemoveDllDirectory.restype = wintypes.BOOL
        self.kernel.GetDllDirectoryW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        self.kernel.GetDllDirectoryW.restype = wintypes.DWORD
        self.kernel.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
        self.kernel.GetSystemDirectoryW.restype = wintypes.UINT
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.psapi.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                                   wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
        self.psapi.EnumProcessModulesEx.restype = wintypes.BOOL
        self.psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        self.psapi.GetModuleFileNameExW.restype = wintypes.DWORD
        self.cookies = []; self.directories = None; self.system_directory = None; self.closed = False
        self._bootstrap('after_ctypes')

    def _bootstrap(self, stage):
        evidence = self.bootstrap_verifier(stage)
        _require(type(evidence) is dict and set(evidence) == {
            'kind', 'stage', 'fresh_process_identity_verified', 'complete_registration_trace',
            'unmanaged_directory_registration', 'provider_source_verified'} and
            evidence == {'kind': 'VERIFIED_CURRENT_DLL_DIRECTORY_BOOTSTRAP_TRACE', 'stage': stage,
                         'fresh_process_identity_verified': True, 'complete_registration_trace': True,
                         'unmanaged_directory_registration': False, 'provider_source_verified': True},
            'current independent DLL-directory bootstrap trace incomplete/unverified')

    def _string(self, function):
        buffer = self.ctypes.create_unicode_buffer(32768)
        length = function(len(buffer), buffer)
        _require(length < len(buffer), 'bounded WinAPI DLL-directory observation')
        if length == 0:
            _require(self.ctypes.get_last_error() == 0, 'successful WinAPI empty DLL-directory observation')
        return buffer.value

    def begin(self, directories, system_directory):
        _require(self.directories is None and not self.closed, 'Win32 finite search scope exactly once')
        self._bootstrap('before_scope')
        buffer = self.ctypes.create_unicode_buffer(32768)
        length = self.kernel.GetSystemDirectoryW(buffer, len(buffer))
        _require(0 < length < len(buffer) and buffer.value == system_directory,
                 'actual Win32 system directory matches selected finite scope')
        self.ctypes.set_last_error(0)
        _require(self._string(self.kernel.GetDllDirectoryW) == '', 'unmanaged legacy DLL directory refused')
        _require(self.kernel.SetDefaultDllDirectories(DEFAULT_FLAGS), 'SetDefaultDllDirectories failed')
        self.directories = list(directories); self.system_directory = system_directory
        for directory in directories:
            if directory != system_directory:
                cookie = self.kernel.AddDllDirectory(directory)
                _require(cookie, 'selected AddDllDirectory failed')
                self.cookies.append((directory, cookie))
        event_scope = self.library_event_observer.begin()
        _require(type(event_scope) is dict and event_scope == {
            'kind': 'OBSERVED_PROCESS_LIBRARY_EVENT_SCOPE_START', 'complete': True,
            'ordered_scope_log_started': True},
            'actual native library event observation scope did not start')
        return self.scope()

    def scope(self):
        _require(self.directories is not None and not self.closed, 'active actual Win32 search scope')
        self._bootstrap('scope_observation')
        self.ctypes.set_last_error(0)
        legacy = self._string(self.kernel.GetDllDirectoryW)
        return {'kind': 'OBSERVED_DLL_DIRECTORY_SCOPE', 'complete': True,
                'directories': list(self.directories), 'system_directory': self.system_directory,
                'default_flags': DEFAULT_FLAGS, 'legacy_directory': legacy,
                'unmanaged_directory_registration': False,
                'completeness_basis': 'Win32_calls_and_verified_fresh_bootstrap_trace', 'closed': False}

    def libraries(self):
        # Two independent enumerations detect loader-list races. A changed list
        # blocks; a future retry policy must be separately reviewed and bounded.
        def snapshot():
            process = self.kernel.GetCurrentProcess()
            modules = (self.wintypes.HMODULE * MAX_LIBRARIES)()
            needed = self.wintypes.DWORD()
            _require(self.psapi.EnumProcessModulesEx(process, modules, self.ctypes.sizeof(modules),
                      self.ctypes.byref(needed), 0x03), 'actual process module enumeration failed')
            width = self.ctypes.sizeof(self.wintypes.HMODULE)
            _require(needed.value % width == 0 and 0 < needed.value <= self.ctypes.sizeof(modules),
                     'complete bounded process module enumeration')
            result = []
            for index in range(needed.value // width):
                buffer = self.ctypes.create_unicode_buffer(32768)
                length = self.psapi.GetModuleFileNameExW(process, modules[index], buffer, len(buffer))
                _require(0 < length < len(buffer) - 1, 'complete bounded observed module path')
                result.append({'path': buffer.value, 'module_handle': int(modules[index])})
            return sorted(result, key=lambda row: row['path'])
        first = snapshot(); second = snapshot()
        _require(first == second, 'stable complete actual loaded-library enumeration')
        return {'kind': 'OBSERVED_PROCESS_LIBRARY_SNAPSHOT', 'complete': True, 'libraries': first}

    def events(self):
        # The observer must be a separately reviewed native interception or
        # notification implementation. Running a Python callback under the
        # Windows loader lock is not introduced by this source change.
        self._bootstrap('library_event_observation')
        return self.library_event_observer.events()

    def close(self):
        _require(not self.closed, 'Win32 library scope cleanup exactly once')
        failures = []
        remaining = []
        for directory, cookie in reversed(self.cookies):
            try:
                if not self.kernel.RemoveDllDirectory(cookie):
                    failures.append('RemoveDllDirectory:' + directory); remaining.append((directory, cookie))
            except BaseException as failure:
                failures.append('RemoveDllDirectory:' + type(failure).__name__); remaining.append((directory, cookie))
        self.cookies = list(reversed(remaining))
        self.closed = True
        legacy = None
        try:
            self._bootstrap('after_cleanup')
            self.ctypes.set_last_error(0)
            legacy = self._string(self.kernel.GetDllDirectoryW)
            _require(legacy == '', 'legacy DLL search directory changed at cleanup')
        except BaseException as failure:
            failures.append('post_scope_observation:' + type(failure).__name__)
        finally:
            try:
                observer_cleanup = self.library_event_observer.close()
                _require(type(observer_cleanup) is dict and observer_cleanup == {
                    'kind': 'OBSERVED_PROCESS_LIBRARY_EVENT_CLEANUP', 'complete': True, 'closed': True},
                    'actual native library event observer cleanup incomplete')
            except BaseException as failure:
                failures.append('event_observer_cleanup:' + type(failure).__name__)
        _require(not failures, 'actual Win32 library/event cleanup failed: ' + ','.join(failures))
        # Default search restriction persists until the isolated owned process
        # exits. There is no GetDefaultDllDirectories or safe unknown restoration.
        return {'kind': 'OBSERVED_DLL_DIRECTORY_CLEANUP', 'complete': True,
                'remaining_managed_directories': [], 'legacy_directory': legacy,
                'all_cookies_removed': True, 'closed': True}


def make_native_guard(policy, contract, *, bootstrap_verifier=None, library_event_observer=None, budget=None):
    contract.stage_native_gate()
    return LibraryGuard(policy, contract, Win32LibraryProvider(contract, bootstrap_verifier, library_event_observer), budget)
