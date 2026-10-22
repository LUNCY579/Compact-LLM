"""Prospective isolated source contracts; not an execution grant."""
import copy

_MANIFEST = {'contracts': {'backend_ref_frame': {'children': {'file_hash_leaf': 1, 'guard': 1}, 'kind': 'input_io', 'local_s_upper': 30, 'owners': ['qualification'], 'wall_s_upper': 300}, 'cp_data_hash': {'children': {'guard': 1}, 'kind': 'fingerprint', 'local_s_upper': 60, 'owners': ['qualification'], 'wall_s_upper': 240}, 'cp_file_hash': {'children': {'guard': 1}, 'kind': 'fingerprint', 'local_s_upper': 60, 'owners': ['qualification'], 'wall_s_upper': 240}, 'cp_frozen_hash': {'children': {'guard': 1}, 'kind': 'fingerprint', 'local_s_upper': 60, 'owners': ['qualification'], 'wall_s_upper': 240}, 'cp_hash_leaf': {'children': {'guard': 1}, 'kind': 'fingerprint', 'local_s_upper': 60, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 240}, 'cp_publication_hash': {'children': {'guard': 1}, 'kind': 'fingerprint', 'local_s_upper': 60, 'owners': ['qualification'], 'wall_s_upper': 240}, 'cp_restore': {'children': {'cp_hash_leaf': 623, 'guard': 627}, 'kind': 'checkpoint_restore', 'local_s_upper': 120, 'owners': ['qualification'], 'wall_s_upper': 262144}, 'cp_restore_adapter': {'children': {'guard': 2, 'opaque_checkpoint_restore': 1}, 'kind': 'checkpoint_adapter_io', 'local_s_upper': 120, 'owners': ['qualification'], 'wall_s_upper': 1048576}, 'cp_save_adapter': {'children': {'guard': 2, 'opaque_checkpoint_save': 1}, 'kind': 'checkpoint_adapter_io', 'local_s_upper': 120, 'owners': ['qualification'], 'wall_s_upper': 32768}, 'cp_save_fresh': {'children': {'cp_hash_leaf': 6, 'guard': 13}, 'kind': 'checkpoint_save', 'local_s_upper': 120, 'owners': ['qualification'], 'wall_s_upper': 4096}, 'cp_tensor_pair_hash': {'children': {'guard': 1}, 'kind': 'fingerprint', 'local_s_upper': 60, 'owners': ['qualification'], 'wall_s_upper': 240}, 'cuda_probe_leaf': {'children': {}, 'kind': 'resource_cuda_probe', 'local_s_upper': 1, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 1}, 'file_hash_leaf': {'children': {}, 'kind': 'input_io', 'local_s_upper': 60, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 60}, 'guard': {'children': {'cuda_probe_leaf': 1, 'lease_parent_check': 1, 'resource_probe_leaf': 1}, 'kind': 'composed_source_guard', 'local_s_upper': 19, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 150}, 'lease_parent_check': {'children': {'worker_os_leaf': 2}, 'kind': 'supervisor_lease', 'local_s_upper': 10, 'owners': ['bootstrap'], 'wall_s_upper': 60}, 'model_fingerprint': {'children': {'guard': 1}, 'kind': 'fingerprint', 'local_s_upper': 60, 'owners': ['qualification'], 'wall_s_upper': 240}, 'opaque_checkpoint_restore': {'children': {'cp_hash_leaf': 640, 'guard': 644}, 'kind': 'checkpoint_restore', 'local_s_upper': 1200, 'owners': ['qualification'], 'wall_s_upper': 524288}, 'opaque_checkpoint_save': {'children': {'cp_hash_leaf': 12, 'guard': 28}, 'kind': 'checkpoint_save', 'local_s_upper': 1200, 'owners': ['qualification'], 'wall_s_upper': 16384}, 'opaque_input_frame': {'children': {'file_hash_leaf': 1, 'guard': 1}, 'kind': 'input_io', 'local_s_upper': 100, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 360}, 'opaque_lease_frame': {'children': {'model_fingerprint': 1, 'worker_os_leaf': 8}, 'kind': 'supervisor_lease', 'local_s_upper': 120, 'owners': ['bootstrap'], 'wall_s_upper': 600}, 'opaque_tokenizer_frame': {'children': {'guard': 1}, 'kind': 'tokenizer_render', 'local_s_upper': 60, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 240}, 'prepared_row': {'children': {'guard': 1}, 'kind': 'tokenizer_render', 'local_s_upper': 60, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 240}, 'resource_probe_leaf': {'children': {}, 'kind': 'resource_probe', 'local_s_upper': 10, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 10}, 'teacher_row': {'children': {'guard': 1}, 'kind': 'input_io', 'local_s_upper': 60, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 240}, 'worker_os_leaf': {'children': {}, 'kind': 'worker_os_binding', 'local_s_upper': 5, 'owners': ['qualification', 'bootstrap'], 'wall_s_upper': 5}, 'opaque_cp_adapter_frame': {'kind': 'checkpoint_adapter_io', 'owners': ['qualification'], 'wall_s_upper': 1048576, 'local_s_upper': 120, 'children': {'cp_hash_leaf': 640, 'guard': 644}}}, 'publication_tail_s_upper': 10, 'schema': 'CE8-SOURCE-OPERATION-CONTRACTS-1'}

def manifest():
    return copy.deepcopy(_MANIFEST)

def apply_wall_maxima(profiles, owner):
    value=manifest()
    for row in value["contracts"].values():
        if owner in row["owners"] and row["kind"] in profiles:
            profiles[row["kind"]]["wall_s_upper"]=max(profiles[row["kind"]]["wall_s_upper"],row["wall_s_upper"])
    return profiles


_GUARDED_COUNTS = {
    'module_import': 1, 'configured_runtime': 1, 'process_rng_seed': 1,
    'tokenizer_load': 1, 'model_load': 2, 'inventory_configuration': 1,
    'journal_construct': 1, 'checkpoint_construct': 3,
    'qualification_artifact': 1, 'qualification_tensor_write': 1,
    'qualification_mask': 1, 'qualification_likelihood': 1,
    'qualification_rng': 1, 'qualification_gradient_reset': 1,
    'qualification_checkpoint_configuration': 2,
    'qualification_reference_forward': 1, 'qualification_candidate_forward': 1,
    'qualification_backward': 1, 'qualification_gradient_validation': 1,
    'qualification_state_fingerprint': 1,
    'qualification_reducer_overhead': 2,
    # Canonical metadata census retained by attempt14: full slots 3,006,530,048
    # bytes / 312 names => <=3179 one-MiB chunks; selected 880,932,864 bytes /
    # 154 names => <=994 chunks. Reference/write have one additional gate;
    # roundtrip calls references() per chunk without a separate entry gate.
    'donor_reference': 3180, 'donor_write': 995, 'donor_roundtrip': 994,
}

def compose_guarded_profiles(profiles):
    """Prospective self allowances plus explicit nested guard/publication costs."""
    m=manifest(); unit=m['contracts']['guard']['wall_s_upper']+m['publication_tail_s_upper']
    basis={}
    for kind,count in _GUARDED_COUNTS.items():
        own=profiles[kind]['wall_s_upper']
        profiles[kind]['wall_s_upper']=own+count*unit
        basis[kind]={'source_self_allowance_s':own,'guard_count':count,
            'inclusive_upper_s':profiles[kind]['wall_s_upper']}
    return basis
