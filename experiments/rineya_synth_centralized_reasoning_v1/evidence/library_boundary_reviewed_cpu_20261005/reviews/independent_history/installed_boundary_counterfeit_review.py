"""Independent source-only counterfeit selector reproduction; no native calls."""
import ast
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
STAGE = REPO / 'experiments/rineya_synth_centralized_reasoning_v1/artifacts/prefix_input_owned_execution_work'
WORK = Path(__file__).resolve().parent
NAMES = ('prefix_input_owned_worker.py', 'prefix_native_pipeline.py',
         'prefix_registered_loader.py', 'prefix_library_policy.py',
         'prefix_tensor_contract.py')
FORBIDDEN = {'torch', 'ctypes', 'psutil', 'native_worker', 'native_process',
             'native_nvml', 'transformers', 'safetensors', 'numpy'}
attempts = []
original_import = builtins.__import__

def imports(name, *args, **kwargs):
    if name.split('.', 1)[0] in FORBIDDEN:
        attempts.append(name)
        raise AssertionError('forbidden native/framework import: ' + name)
    return original_import(name, *args, **kwargs)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def hashes():
    return {name: hashlib.sha256((STAGE/name).read_bytes()).hexdigest() for name in NAMES}

before = hashes()
worker_path = STAGE/'prefix_input_owned_worker.py'
loader_path = STAGE/'prefix_registered_loader.py'
helper_path = STAGE/'prefix_library_policy.py'
pins = {str(STAGE/name): {'bytes': (STAGE/name).stat().st_size,
        'sha256': before[name], 'role': 'python_source'} for name in NAMES}
value = {'fixture_only': False, 'config': {'source_files': pins}}
C = types.SimpleNamespace(PACKAGE=STAGE, require=require, canonical=canonical)
tree = ast.parse(worker_path.read_text(encoding='utf-8'), filename=str(worker_path))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == '_installed_framework_boundary')
namespace = {'C': C}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(worker_path), 'exec'), namespace)

# Intentionally unrelated classes: selected filenames/descriptors are spoofed
# metadata only. No selected loader/helper body is executed for these objects.
guard_namespace = {'nullcontext': contextlib.nullcontext,
                   '__compiled_source_descriptor__': dict(pins[str(helper_path)])}
exec(compile('class CounterfeitGuard:\n'
             '    def framework_import(self):\n'
             '        return nullcontext()\n', str(helper_path), 'exec'), guard_namespace)
guard = guard_namespace['CounterfeitGuard']()
guard.active = True; guard.failed = False; guard.closed = False
finder_namespace = {}
exec(compile('class CounterfeitFinder:\n'
             '    def framework_import(self):\n'
             '        return self.library_guard.framework_import()\n', str(loader_path), 'exec'), finder_namespace)
finder = finder_namespace['CounterfeitFinder']()
finder.selected_loader_source = dict(pins[str(loader_path)])
finder.selected_context_projection = canonical(value)
finder.library_guard = guard; finder.selected_library_guard = guard
finder.fixture_only = False

with patch.object(builtins, '__import__', side_effect=imports), patch.object(sys, 'meta_path', [finder]):
    selected = namespace['_installed_framework_boundary'](value, finder.framework_import)
    require(selected.__self__ is finder, 'reproduction did not return counterfeit finder method')
    with selected():
        delegated = True

after = hashes()
require(before == after, 'source changed during focused reproduction')
require(not attempts, 'forbidden import attempted')
receipt = {
    'kind': 'INDEPENDENT_CALLER_SUPPLIED_COUNTERFEIT_DLL_BOUNDARY_SELECTOR_REPRODUCTION',
    'fixture_only': True, 'native_PASS': False, 'native_authority_issued': False,
    'native_stage_opened': False, 'root_source_edits': False,
    'source_before': before, 'source_after': after,
    'forbidden_import_attempts': attempts, 'native_or_host_api_calls': [],
    'counterfeit_installed_finder_accepted': True,
    'no_observed_library_guard_delegate_reached': delegated,
    'source_code_executed': 'exact AST-extracted current worker selector only',
    'injected_assumptions': ['plain context projection, no native context issuance',
                            'counterfeit finder and guard, selected filenames and copied source pins'],
    'finding': 'The selector accepts caller-created unrelated classes with copied writable pins/context/state and spoofed co_filename. No installed-source registry or exact compiled class provenance is consumed.',
    'recommended_fix': 'A source-authenticated closure validator must bind actual installation records to exact finder/guard classes/functions, original projection/pins and fixture/native mode; invalidate on removal/failure and require the current unique installation and active original guard.'
}
receipt['sha256'] = hashlib.sha256(canonical(receipt)).hexdigest()
path = WORK/('installed_boundary_counterfeit_review_' + str(time.time_ns()) + '.json')
with path.open('x', encoding='ascii', newline='\n') as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2, ensure_ascii=True); stream.write('\n')
print(json.dumps({'receipt': str(path), 'body_sha256': receipt['sha256'],
                  'counterfeit_accepted': True, 'source_stable': before == after,
                  'forbidden_imports': attempts}, sort_keys=True))
