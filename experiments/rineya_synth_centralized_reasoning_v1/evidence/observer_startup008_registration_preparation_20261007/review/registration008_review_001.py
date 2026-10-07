"""Read-only exact registration007-to008 delta, no native execution."""
from pathlib import Path
import hashlib, json, subprocess, sys
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
P = WORK.parents[1]
IMPL = 'ab6ad25e71b5d63365655e917229f55c4e0bd1c3'
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
refs = {}
def sealed(q):
    q = Path(q); raw = q.read_bytes(); x = json.loads(raw); body = dict(x); seal = body.pop('sha256'); assert digest(body) == seal
    refs[str(q)] = {'file_sha256': hashlib.sha256(raw).hexdigest(), 'body_sha256': seal, 'bytes': len(raw)}
    return x
def load(q): return json.loads(Path(q).read_bytes())
c = sealed(P / 'configs/SYNTH_OBSERVER_STARTUP_CONFIG_008.json')
r = sealed(P / 'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_008.json')
g = sealed(P / 'configs/SYNTH_OBSERVER_STARTUP_REGISTRATION_008.json')
v = sealed(P / 'evidence/observer_startup_byte_admission_review_008.json')
a = sealed(P / 'evidence/observer_startup008_actual_byte_review_20261007.json')
c7 = load(P / 'configs/SYNTH_OBSERVER_STARTUP_CONFIG_007.json'); r7 = load(P / 'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_007.json'); g7 = load(P / 'configs/SYNTH_OBSERVER_STARTUP_REGISTRATION_007.json')
assert c['implementation_commit'] == r['implementation_commit'] == g['implementation_commit'] == v['implementation_commit'] == a['implementation_commit'] == IMPL
assert len(c['source_files']) == 27 and len(r['files']) == 2983
assert c['source_files'] == g['source_files'] and c['runtime_files'] == g['runtime_files'] == r['files'] == r7['files']
assert g['config_body_sha256'] == c['sha256'] and g['runtime_body_sha256'] == r['sha256']
assert refs[str(P / 'configs/SYNTH_OBSERVER_STARTUP_CONFIG_008.json')]['file_sha256'] == '7082bc7a0881b7d088312cdec95e6fe2bbce8c3ed5dae4dc25512e69c1d6e299'
assert refs[str(P / 'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_008.json')]['file_sha256'] == '138ff38107c18610cc2f3dbb31aa5606d9babb579b972b8bd6a6401f14ced7dd'
assert refs[str(P / 'configs/SYNTH_OBSERVER_STARTUP_REGISTRATION_008.json')]['file_sha256'] == '2a341170c9fd7075ebf1ca86284efe39cb3008f6b40394d30437cf561906e014'
for key in c:
    if key not in {'implementation_commit', 'source_files', 'sha256'}: assert c[key] == c7[key], key
for key in g:
    if key not in {'implementation_commit', 'config_body_sha256', 'runtime_body_sha256', 'source_files', 'sha256'}: assert g[key] == g7[key], key
for key in r:
    if key not in {'implementation_commit', 'cache_policy', 'qualification_review', 'sha256'}: assert r[key] == r7[key], key
assert set(c['source_files']) == set(c7['source_files'])
changed = {path: d for path, d in c['source_files'].items() if d != c7['source_files'][path]}
assert set(changed) == {c['worker_path']}
assert a['changed_source_descriptors'] == {path: {'before': c7['source_files'][path]['sha256'], 'after': descriptor['sha256']} for path, descriptor in changed.items()}
assert changed[c['worker_path']]['sha256'] == '0b4b3aae9182e2f8b1f820ad3a4e39fafc609d1a5cb4c9648069e1fd1b8edfa0'
assert Path(c['worker_path']).read_bytes() == (WORK / 'worker.py').read_bytes()
source_before = {}
for path, descriptor in c['source_files'].items():
    q = Path(path); raw = q.read_bytes(); assert len(raw) == descriptor['bytes'] and hashlib.sha256(raw).hexdigest() == descriptor['sha256']
    blob = subprocess.run(['git', 'cat-file', 'blob', IMPL + ':' + q.relative_to(P.parents[1]).as_posix()], cwd=P.parents[1], capture_output=True, check=True).stdout; assert blob == raw
    source_before[path] = hashlib.sha256(raw).hexdigest()
cache = r['cache_policy']; old = r7['cache_policy']
for key in cache:
    if key != 'allowed_bootstrap_modules': assert cache[key] == old[key], key
assert set(cache['allowed_bootstrap_modules']) == set(old['allowed_bootstrap_modules'])
assert {name for name in cache['allowed_bootstrap_modules'] if cache['allowed_bootstrap_modules'][name] != old['allowed_bootstrap_modules'][name]} == {'__main__'}
assert cache['allowed_bootstrap_modules']['__main__'] == {**old['allowed_bootstrap_modules']['__main__'], 'source_sha256': changed[c['worker_path']]['sha256']}
assert len(cache['allowed_cache_files']) == 57
plan = sealed(cache['tensor_prelude']['path']); assert refs[cache['tensor_prelude']['path']]['file_sha256'] == cache['tensor_prelude']['file_sha256'] and plan['sha256'] == cache['tensor_prelude']['body_sha256']
packet = sealed(plan['packet_path']); assert refs[plan['packet_path']]['file_sha256'] == plan['packet_file_sha256'] and packet['sha256'] == plan['packet_body_sha256']
assert packet['native_launch_authorized'] is packet['native_PASS'] is packet['native_authority_issued'] is False and all(value is None for value in packet['pending'].values())
for key, expected in {'source_inventory_sha256': digest(c['source_files']), 'runtime_inventory_sha256': digest(r['files']), 'observer_registration_sha256': digest(r['library_event_observer']), 'allowed_initial_loaded_libraries_sha256': digest(r['allowed_initial_loaded_libraries']), 'bootstrap_policy_sha256': digest(cache)}.items(): assert v[key] == expected, key
assert r['qualification_review']['file_sha256'] == refs[str(P / 'evidence/observer_startup_byte_admission_review_008.json')]['file_sha256'] and r['qualification_review']['body_sha256'] == v['sha256']
selection = {**c['source_files'], **r['files']}
assert a['selected_descriptor_map_sha256'] == digest(selection) and a['actual_hashed_bytes'] == sum(d['bytes'] for d in selection.values()) == 3959198287
assert a['reader_sha256'] == hashlib.sha256((P / 'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py').read_bytes()).hexdigest() == '5c4c47a2257d2c62bef875528ac0dc940d83eefcbeb4a9f2f7fd6c4973122892'
assert a['reader_roles_changed'] is a['cache_source_equivalence_qualified'] is a['current_tensor_bootstrap_qualified'] is a['resources_observed'] is a['native_PASS'] is False
source_after = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in c['source_files']}; assert source_before == source_after
receipt = {'kind': 'INDEPENDENT_PROSPECTIVE_OBSERVER_REGISTRATION008_REVIEW_V1', 'implementation_commit': IMPL, 'sealed_input_count': len(refs), 'document_refs': refs, 'source_before': source_before, 'source_after': source_after, 'source_implementation_blobs_verified': 27, 'only_changed_source': changed, 'runtime2983_descriptors_and_cache57_exact007': True, 'only_changed_bootstrap_row': '__main__ source SHA', 'scope_owner_permission_resource_ABI_operations_preserved': True, 'plan_profile_windows_library_stdlib_maps_unchanged': True, 'root_full_byte_receipt_joined': True, 'full_runtime_rehashed_by_this_review': False, 'new_native_measurement_pending': True, 'failed007_not_tensor_or_cache_proof': True, 'dispatch_packet_uncommitted_at_review': True, 'site_disabled': sys.flags.no_site == 1, 'native_API_DLL_model_called': False, 'native_PASS': False, 'native_authority_issued': False}
receipt['sha256'] = digest(receipt)
out = HERE / 'REGISTRATION008_REVIEW001.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'source_blobs': 27, 'sealed_inputs': len(refs), 'path': str(out)}, sort_keys=True))
