"""Independent focused authored replay; no model/data/native APIs permitted."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
from collections import Counter
from unittest.mock import patch
import builtins, hashlib, io, json, os, sys, time, unittest

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == 'rineya_synth_centralized_reasoning_v1')
AUTHOR = PACKAGE / 'artifacts/long_guard_owned_runtime_preparation_work'
CORE = PACKAGE / 'evidence/long_guard_collection_reviewed_cpu_20261007/current001'
SOURCES = [AUTHOR / n for n in ('long_guard_owned_runtime.py', 'long_guard_owned_dispatch.py', 'test_long_guard_owned.py')]
SOURCES += [CORE / 'long_guard_collection.py', HERE / 'review_owned_long.py']
def pins():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}

reads, denials, imports = [], [], []
realio, realopen, realimport = io.open, builtins.open, builtins.__import__
def read(path, mode='r', *args, **kwargs):
    if isinstance(path, (str, bytes, Path)) and not any(c in str(mode) for c in ('w', 'a', 'x', '+')):
        p = Path(path).resolve(); reads.append(str(p))
        permitted = (p.is_relative_to(HERE) or p.is_relative_to(AUTHOR)
                     or p.is_relative_to(CORE) and p.suffix == '.py'
                     or p.parent == PACKAGE and p.suffix == '.py'
                     or p.parent == PACKAGE / 'tests' and p.suffix == '.py')
        if not permitted:
            denials.append(str(p)); raise AssertionError('independent actual input forbidden')
    return realio(path, mode, *args, **kwargs)
def import_checked(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'transformers', 'tokenizers', 'ctypes', 'psutil', 'safetensors', 'numpy'}:
        imports.append(name); raise AssertionError('independent native/framework import forbidden')
    return realimport(name, *args, **kwargs)

before = pins()
traps = [patch('io.open', side_effect=read),
         patch.object(builtins, 'open', side_effect=lambda p, *a, **k: read(p, *a, **k) if isinstance(p, (str, bytes, Path)) else realopen(p, *a, **k)),
         patch.object(builtins, '__import__', side_effect=import_checked)]
for t in traps: t.start()
author = ModuleType('selected_authored_long_controls')
author.__file__ = str(AUTHOR / 'test_long_guard_owned.py')
exec(compile((AUTHOR / 'test_long_guard_owned.py').read_bytes(), author.__file__, 'exec'), author.__dict__)

class IndependentControls(author.OwnedControls):
    def test_independent_activation_postguard_preserves_cleanup(self):
        src, plan, cfg, torch, model, decoder, scientific, seen = self.prepared(1)
        folder = self.root / 'after_install_refusal'; folder.mkdir()
        journal = self.H['Journal'](folder); report = {}
        def guard(stage):
            if stage == 'after_install_activation_hooks': raise RuntimeError('independent after-install refusal')
        journal.guard = guard
        try:
            with self.assertRaisesRegex(RuntimeError, 'after-install refusal'):
                self.R._CPU_loaded(cfg, src, plan, 1, self.L, self.component,
                    torch=torch, model=model, decoder=decoder, source_hashes={}, journal=journal,
                    guard=guard, helpers=scientific, device='cpu', report=report, capability=self.cap)
        finally: journal.stream.close()
        self.assertIn('activation_removed', seen)
        self.assertEqual(model.calls, [])
        self.assertEqual(model.hooks, [])
        self.assertEqual(report['cleanup_errors'], [])
        self.assertEqual(journal.snapshot()['operations']['install_activation_hooks'],
                         {'attempted': 1, 'returned': 1, 'failed': 0, 'pending': 0})

    def test_independent_cleanup_guard_refusal_retains_primary(self):
        src, plan, cfg, torch, model, decoder, scientific, seen = self.prepared(1)
        model.parity_bad = True
        folder = self.root / 'cleanup_refusal'; folder.mkdir()
        journal = self.H['Journal'](folder); report = {}
        def guard(stage):
            if stage == 'before_owned_hooks_remove': raise RuntimeError('independent cleanup boundary refusal')
        journal.guard = guard
        try:
            with self.assertRaisesRegex(ValueError, 'physical dense'):
                self.R._CPU_loaded(cfg, src, plan, 1, self.L, self.component,
                    torch=torch, model=model, decoder=decoder, source_hashes={}, journal=journal,
                    guard=guard, helpers=scientific, device='cpu', report=report, capability=self.cap)
        finally: journal.stream.close()
        self.assertEqual(model.hooks, [])
        self.assertIn('activation_removed', seen)
        self.assertEqual(report['cleanup_errors'], [])
        self.assertEqual(report['cleanup_boundary_error']['message'], 'independent cleanup boundary refusal')
        self.assertEqual(len(model.calls), 2)

    def test_independent_strict_heartbeat_and_physical_nonforward_refusal(self):
        path = self.root / 'heartbeat.jsonl'
        path.write_bytes(b'{"unix_seconds":1,"unix_seconds":2}\n')
        with self.assertRaisesRegex(ValueError, 'duplicate'): self.H['read_heartbeat'](path)
        path.write_bytes(b'{"unix_seconds":NaN}\n')
        with self.assertRaisesRegex(ValueError, 'nonfinite'): self.H['read_heartbeat'](path)
        batch = dict(context_cap=1024, maximum_attempted_forwards=2048,
                     maximum_input_positions=64*(32*992+496))
        torch = self.Torch(); dose = self.R.ForwardDose(torch, batch)
        call = dose.wrap(lambda operation, fn: fn())
        self.assertEqual(call('outer_asset', lambda: call('inner_asset', lambda: 7)), 7)
        ids = torch.tensor([[5]*419], dtype=torch.long, device='cpu')
        with self.assertRaisesRegex(ValueError, 'operation'):
            call('unregistered_asset_forward', lambda: dose.before(None, (), dict(input_ids=ids)))
        self.assertIsNone(dose.current)
        self.assertEqual(dose.snapshot()['attempted_calls'], 0)

    def test_independent_native_worker_gate_precedes_helpers(self):
        calls = []
        with patch.object(self.R, 'worker_helpers', side_effect=lambda **k: calls.append('helpers')):
            with self.assertRaisesRegex(ValueError, 'staged'):
                self.R.worker_body(None, None, None, None, None, None, None)
        self.assertEqual(calls, [])

    def test_independent_actual_compiled_supervisor_denial(self):
        external = PACKAGE.parents[1].parent
        config_path = self.root / 'CPU_CONFIG.json'
        config = {'config_path': str(config_path),
                  'output_directory': str(external / 'CPU_LONG_DO_NOT_CREATE')}
        saved = []; forbidden = []
        def deny(name):
            def invoke(*a, **k):
                forbidden.append(name); raise AssertionError('native supervisor boundary forbidden')
            return invoke
        namespace = dict(PACKAGE=PACKAGE, Path=Path, sys=sys,
            time=SimpleNamespace(monotonic=lambda: 10.0, sleep=deny('sleep')),
            _utc=lambda: '2026-10-07T00:00:00+00:00', BOUNDS={},
            CPU_INTERVAL=.1, CPU_FRESHNESS=.2, GPU_INTERVAL=.5, GPU_TIMEOUT=2.,
            GPU_FRESHNESS=2., ADMISSION_INTERVAL=.1, ADMISSION_DEADLINE=10.,
            _shutdown_gpu_probe=lambda: None, _gpu_pending_threads=lambda: [],
            _inventory=lambda output: [], _json_safe=lambda value: value,
            _json_new=lambda path, value: saved.append((str(path), value)),
            _LaunchFailure=RuntimeError, _bounds_valid=deny('bounds'),
            resource_snapshot=deny('resource_snapshot'), _WindowsOwnedWorker=deny('worker'))
        supervise, proof = self.D.supervisor_body(namespace, config, external)
        supervise.__globals__['_new_output'] = lambda output: self.root
        supervise.__globals__['_acquire_lease'] = deny('lease')
        result = supervise(config_path, self.root)
        self.assertEqual(proof['purpose_worker_changes'], {'worker': 2, 'scope': 1})
        self.assertEqual(proof['external_changes'], {'_new_output': 1, '_acquire_lease': 1})
        self.assertEqual(result['status'], 'ADMISSION_DENIED')
        self.assertIn('selected qualification/owner-approved byte preflight absent', result['failure'])
        self.assertFalse(result['worker_attempted'])
        self.assertFalse(result['worker_launched'])
        self.assertTrue(result['gpu_telemetry_cleanup_verified'])
        self.assertEqual(forbidden, [])
        self.assertEqual(len(saved), 1)

selected = ['test_three_caps_original64_collection_and2050_outer_meter',
            'test_cap32_actual_dose_and_original_stage_bound',
            'test_decoder_failure_stops_and_removes_hooks',
            'test_independent_activation_postguard_preserves_cleanup',
            'test_independent_cleanup_guard_refusal_retains_primary',
            'test_independent_strict_heartbeat_and_physical_nonforward_refusal',
            'test_independent_native_worker_gate_precedes_helpers',
            'test_independent_actual_compiled_supervisor_denial']
start = time.monotonic()
try:
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(n) for n in selected))
finally:
    for t in reversed(traps): t.stop()
after = pins()
receipt = dict(kind='INDEPENDENT_FOCUSED_LONG_OWNED_RUNTIME_CPU_REVIEW',
    tests=result.testsRun, success=result.wasSuccessful(), seconds=time.monotonic()-start,
    selected_controls=selected, source_before=before, source_after=after, source_unchanged=before==after,
    read_paths_and_counts=dict(Counter(reads)), independent_denials=denials,
    author_intentional_denials=IndependentControls.denials, forbidden_import_attempts=imports,
    author_forbidden_import_attempts=IndependentControls.imports, fixture_only=True,
    native_PASS=False, native_qualification_established=False, scoring_authorized=False,
    actual_model_data_API_access=False, native_success_branch_exercised=False)
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
target = HERE / ('INDEPENDENT_FOCUSED_'+str(time.time_ns())+'.json')
target.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n', encoding='utf8')
print(json.dumps(dict(path=str(target), tests=receipt['tests'], success=receipt['success'],
                     source_unchanged=receipt['source_unchanged'], sha256=receipt['sha256'])))
raise SystemExit(0 if result.wasSuccessful() and before==after and not imports else 1)
