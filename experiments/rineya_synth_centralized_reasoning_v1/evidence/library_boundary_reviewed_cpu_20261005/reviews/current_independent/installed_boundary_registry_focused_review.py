"""Finite installed-boundary controls; CPU only, no host/native imports."""
from pathlib import Path
import builtins, hashlib, importlib.util, io, json, sys, time, unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
STAGE=ROOT/'experiments/rineya_synth_centralized_reasoning_v1/artifacts/prefix_input_owned_execution_work'
OUT=Path(__file__).resolve().parent
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
before={p.name:sha(p) for p in STAGE.glob('*.py')}
forbidden={'torch','ctypes','psutil','native_worker','native_process','native_nvml','transformers','safetensors','numpy','tensorflow','jax'}
real_import=builtins.__import__;bad_imports=[]
def imports(name,*args,**kwargs):
    if name.split('.',1)[0] in forbidden:
        bad_imports.append(name);raise AssertionError('forbidden native/framework import: '+name)
    return real_import(name,*args,**kwargs)
real_open=Path.open;bad_reads=[]
def file_open(path,*args,**kwargs):
    if path.suffix.lower() in {'.dll','.pyd','.exe','.zip','.pyc','.safetensors','.bin','.pt','.pth'}:
        if not any(part.startswith('CPU_library_manifest_') for part in path.parts):
            bad_reads.append(str(path));raise AssertionError('real host/runtime/model/cache byte read')
    return real_open(path,*args,**kwargs)
def source(name):
    path=STAGE/(name+'.py');module=importlib.util.module_from_spec(importlib.util.spec_from_file_location(name,path))
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__);return module
names=(
 'test_actual_installed_finder_encloses_framework_supplier_and_cleans_up',
 'test_actual_installed_CPU_finder_cannot_satisfy_native_selector',
 'test_recorded_CPU_installation_cannot_be_promoted_by_mutable_flags_and_projection',
 'test_private_selector_refuses_forged_capability_and_non_CPU_projection',
 'test_counterfeit_finder_and_guard_selected_filenames_pins_state_cannot_create_registry_record',
 'test_actual_installed_finder_refuses_nullcontext_and_arbitrary_callbacks',
 'test_actual_installed_finder_refuses_wrong_context_projection',
 'test_actual_installed_finder_refuses_wrong_compile_descriptor',
 'test_actual_finder_removed_from_importer_cannot_satisfy_native_selector',
 'test_actual_installed_finder_refuses_method_from_another_finder',
 'test_actual_installed_finder_refuses_missing_library_guard',
 'test_actual_installed_finder_refuses_inactive_library_guard',
 'test_actual_installed_finder_refuses_noop_bound_method_replacement',
 'test_actual_installed_guard_refuses_noop_bound_method_replacement',
 'test_actual_installed_guard_refuses_joint_counterfeit_identity_swap',
 'test_actual_installed_guard_refuses_compiled_provider_descriptor_drift',
 'test_installed_finder_refuses_library_drift_before_framework_delegate',
 'test_CPU_selected_boundary_cannot_enter_public_native_loader_or_bindings')
output=io.StringIO();started=time.monotonic()
sys.path.insert(0,str(STAGE))
with patch.object(builtins,'__import__',side_effect=imports),patch.object(Path,'open',file_open):
    manifest=source('test_library_boundary_manifest')
    framework=source('test_framework_library_enclosure')
    suite=unittest.TestSuite([manifest.LibraryBoundaryManifestControls(name) for name in names])
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(framework.FrameworkLibraryEnclosureControls))
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
elapsed=time.monotonic()-started
after={p.name:sha(p) for p in STAGE.glob('*.py')}
out=OUT/('installed_boundary_registry_output_'+str(time.time_ns())+'.txt')
with out.open('x',encoding='utf-8') as stream:stream.write(output.getvalue())
receipt={'kind':'INDEPENDENT_SOURCE_OWNED_LIBRARY_INSTALLATION_REGISTRY_FOCUSED_CPU_REVIEW',
 'source_before':before,'source_after':after,'sources_unchanged':before==after,
 'passed':result.wasSuccessful(),'tests_run':result.testsRun,'skips':len(result.skipped),
 'selected_manifest_controls':list(names),'shared_factory_controls':5,
 'elapsed_seconds':elapsed,'output_file':str(out),'output_sha256':sha(out),
 'forbidden_import_attempts':bad_imports,'actual_host_runtime_model_cache_byte_attempts':bad_reads,
 'root_source_edits':False,'fixture_only':True,'native_PASS':False,'native_authority_issued':False,
 'source_assessment':'The shared factory and registered entry consume the installed-selector check before native suppliers. Fresh bounded/pinned loader and helper source authenticate finder/guard/validator executable methods and class/globals ties. Installation records conserve original classes/functions/guard/context and CPU mode; removal invalidates the record. The prior counterfeit objects/no-op callbacks and mutable CPU promotion are refused.',
 'limits':['Only caller-supplied counterfeit objects/callbacks and finite metadata/method substitutions are covered.',
 'Hostile reconstruction or mutation of every trusted Python global/class/closure is outside the same-process authority threat model.',
 'All library snapshots/events/directories and historical runtime/bootstrap obligations here use explicit inert suppliers; actual native observer, fresh directory trace and current runtime qualification remain absent.',
 'No native stage opened, native context authority issued, model/evaluation membership or success qualification.']}
receipt['sha256']=hashlib.sha256(canonical(receipt).encode('ascii')).hexdigest()
path=OUT/('installed_boundary_registry_review_'+str(time.time_ns())+'.json')
with path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'receipt':str(path),'body_sha256':receipt['sha256'],'passed':result.wasSuccessful(),
 'tests_run':result.testsRun,'sources_unchanged':before==after,'forbidden_imports':bad_imports,'bad_byte_reads':bad_reads}))
if not result.wasSuccessful() or result.skipped or before!=after or bad_imports or bad_reads:
    print(output.getvalue());raise SystemExit(1)
