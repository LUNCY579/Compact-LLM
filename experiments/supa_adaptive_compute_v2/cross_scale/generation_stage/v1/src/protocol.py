"""Prospective Track B small-screen contract; no configuration is emitted on import."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from . import upstream

STAGE = Path(__file__).resolve().parents[1]
REPO = upstream.REPO
DATASET = upstream.PROGRAMME / 'data/generated/v2/cross_scale/manifest.json'
GATES = (.03, .10, .20, .40)
ARMS = ('XG', 'XS', 'XR', 'XM')
PRESERVATION = ('ordinary_qa_preservation', 'instruction_evidence', 'retrieval_multihop')
BANK_CONTRACT = '6857158ee7a360bdf2f72a21c5ee78041ab0a5ecda264c0387e1e5facf638a46'
SCORER = '62937db0530db2f8b43672ff1ea3e9ffa96dce5e8b7c9cf5bac6996b3d29c11e'
CHECKPOINT = 'a99f079d32212bbcbb703c2e2bad801ea32d43afea52004f38bf94eec5dc4026'


def cell_id(arm, gate=0.):
    if arm == 'X0' and gate == 0.:
        return 'X0'
    if arm not in ARMS or type(gate) not in (int, float) or gate not in GATES:
        raise ValueError('only the fixed four doses and five experimental arms are permitted')
    return f'{arm}_g{gate:.2f}'


def _relative(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPO.resolve()):
        raise ValueError('artifact must belong to this repository')
    return path.relative_to(REPO.resolve()).as_posix()


def _resolve(relative, root):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('canonical repository-relative artifact path required')
    path = (REPO / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or _relative(path) != relative:
        raise ValueError('artifact outside its authorized stage subtree or aliased path')
    return path


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA256 required')


def _fixed(device):
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('only explicit CPU or qualified CUDA:0 FP32 is supported')
    return {
        'schema_version': 'cross_scale_generation_screen_v1', 'track': 'cross_scale',
        'stage': 'architecture_generation_screen', 'role': 'architecture_dev', 'analysis_role': 'candidate_selection',
        'seed': 2702, 'threads': 4, 'device': device, 'dtype': 'float32',
        'attention_implementation': 'eager', 'use_cache': False, 'tf32': False,
        'max_new_tokens': 48, 'max_input_tokens': 256, 'max_prompt_seconds': 900,
        'max_rss_gib': 8., 'min_available_gib': 3.5 if device == 'cpu' else .75,
        'min_available_vram_gib': 4., 'max_cuda_memory_gib': 6.5, 'top_k': 10,
        'hidden_rms_ratio_bounds': [.25, 4.], 'checkpoint_sha256': CHECKPOINT,
        'model_class': 'Qwen3AbstainForCausalLM', 'num_hidden_layers': 28,
        'donor_revision': '1cfa9a7208912126459214e8b04321603b3df60c',
        'selected_pair': {'supa_layer': 19, 'donor_layer': 24},
        'selected_objective': 'B3', 'baseline_cell': 'X0', 'shard_count': 8,
        'cells': {'X0': {'arm': 'X0', 'gate': 0.}, **{
            cell_id(arm, gate): {'arm': arm, 'gate': gate} for gate in GATES for arm in ARMS}},
        'gate_grid': list(GATES), 'dataset_manifest': _relative(DATASET),
        'dataset_manifest_contract_sha256': BANK_CONTRACT, 'scorer_sha256': SCORER,
        'data': {'track': 'cross_scale', 'allowed_roles': ['architecture_dev'],
                 'examples': 128, 'independent_scenarios': 128, 'variant': 'original',
                 'calibration_may_fit_control_only': True, 'sealed_access': False},
        'native_comparator': {'arm': 'XN', 'available': False,
            'reason': 'No independently selected best native operator and completed native confirmation exists at this freeze.',
            'native_efficacy_inputs_permitted': False},
        'screen_policy': {
            'minimum_paired_wins_vs_X0': 2, 'minimum_net_correct_vs_X0': 1,
            'maximum_additional_preservation_errors': 1,
            'preservation_families': list(PRESERVATION),
            'maximum_additional_invalid_outputs': 0, 'maximum_safety_stops': 0,
            'require_all_same_dose_controls_complete': True,
            'require_XG_accuracy_at_least_each_control': True,
            'hard_failure_policy': 'terminate_cell_preserve_failed_receipt_continue_other_fixed_cells',
            'qualification_failure_policy': 'reject_entire_dose',
            'ranking': ['max_XG_minus_best_control', 'max_XG_minus_X0', 'lower_gate'],
            'bootstrap_replicates': 10000, 'bootstrap_seed': 2703,
            'identity_or_kl_threshold': None, 'efficacy_claim_licensed': False},
        'runtime_qualification': 'cross_scale_generation_screen_gpu_fp32_v1',
        'qualification': {'examples_per_family': 1, 'fixtures_per_cell': 8,
            'capture_layers': [2, 19, 27], 'cpu_min_available_gib': 3.5,
            'max_abs_logit_error': .002, 'max_relative_logit_l2': .0002,
            'minimum_logit_cosine': .999999, 'max_abs_nll_error': .002,
            'max_relative_hidden_l2': .0002, 'minimum_hidden_cosine': .999999,
            'require_greedy_identity': True, 'require_gpu_repeat_identity': True,
            'require_disabled_restored_identity': True, 'baseline_pass_required': True,
            'failed_cell_rejects_entire_dose': True},
        'confirmatory_generation_licensed': False,
        'closed_gates': ['confirmatory_generation', 'sealed_evaluation', 'qwen8b', 'routing',
                         'MLA', 'MHA', 'MTP', 'PEFT', 'backbone_training', 'distillation'],
    }


def _control_fixed():
    return {'kind': 'independent_frozen_swiglu', 'insertion_layer': 19,
        'hidden_size': 1024, 'intermediate_size': 34560, 'seed': 2702,
        'weight_std': .02, 'unit_rms_epsilon': 1e-6,
        'additional_parameters': 106168320, 'additional_active_parameters': 106168320,
        'additional_matmul_flops_per_token': 212336640, 'training': False,
        'calibration': 'track_B_fit_only_scenario_balanced_ordinary_residual_energy'}


def _verify_export(path, *, full_upstream=False):
    """Verify a committed exact subset and its current complete Stage 1 provenance."""
    from .provenance import verify_export
    path = upstream._within(path, upstream.EXPORT_ROOT)
    if path.name != 'manifest.json':
        raise ValueError('export manifest.json required')
    manifest = verify_export(path, require_committed=True, verify_upstream_now=full_upstream)
    expected = {'schema_version': 'cross_scale_generation_bridge_export_v1', 'status': 'COMPLETE',
        'source_objectives': {'XG': 'B3', 'XS': 'BS', 'XR': 'BR'},
        'subset_byte_identical_on_reload': True, 'dtype': 'float32', 'device': 'cpu',
        'frozen': True, 'trainable_parameters': 0, 'gradient_tensors': 0,
        'refit_performed': False, 'weights_changed': False, 'tensor_file': 'bridges.pt'}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('complete unchanged XG/XS/XR same-pair export required')
    if (manifest['context']['supa_layer'], manifest['context']['donor_layer'],
            manifest['context']['selected_objective']) != (19, 24, 'B3'):
        raise ValueError('only the frozen independently selected B3 S19/D24 is admitted')
    tensor_path = path.parent / 'bridges.pt'
    if upstream._sha(tensor_path) != manifest['tensor_file_sha256']:
        raise ValueError('exported bridge bytes changed')
    return {'manifest_path': _relative(path), 'manifest_sha256': upstream._sha(path),
        'tensor_path': _relative(tensor_path), 'tensor_sha256': upstream._sha(tensor_path),
        'source_objectives': manifest['source_objectives'], 'tensor_hashes': manifest['tensor_sha256']}


def _verify_calibration(path, export):
    from .control_calibration import verify_calibration_artifact
    from .provenance import require_committed
    path = upstream._within(path, STAGE / 'evidence')
    if path.name != 'report.json':
        raise ValueError('committed stage-owned calibration report.json required')
    digest = upstream._sha(path)
    report = verify_calibration_artifact(path, expected_sha256=digest)
    require_committed(path)
    # The verifier is stage-owned and must bind precisely the same selected graft.
    if (report['export_manifest_sha256'] != export['manifest_sha256']
            or Path(REPO / report['export_manifest_path']).resolve()
            != (REPO / export['manifest_path']).resolve()):
        raise ValueError('XM calibration belongs to a different upstream export')
    return {**_control_fixed(), 'alpha': report['alpha'],
        'state_sha256': report['control_state_sha256'], 'artifact_path': _relative(path),
        'artifact_sha256': digest}


def build_config(*, export_manifest_path, calibration_artifact_path, device='cpu'):
    """Derive a config only from committed verified artifacts; never write a file.

    The caller must exclusively write and commit this result and its stage-owned
    registration before qualification/generation. There is no scalar override.
    """
    export = _verify_export(Path(export_manifest_path), full_upstream=True)
    control = _verify_calibration(Path(calibration_artifact_path), export)
    config = {**_fixed(device), 'dataset_manifest_sha256': upstream._sha(DATASET),
              'upstream_export': export, 'matched_control': control}
    validate_config(config)
    return config


def validate_config(config, *, verify_artifacts=True):
    """Reject policy/geometry drift; real callers leave artifact verification on."""
    if not isinstance(config, dict):
        raise ValueError('configuration must be an object')
    fixed = _fixed(config.get('device'))
    if set(config) != set(fixed) | {'dataset_manifest_sha256', 'upstream_export', 'matched_control'}:
        raise ValueError('unexpected or missing generation screen configuration fields')
    # Canonical comparison distinguishes booleans from integers and rejects NaN.
    if upstream._canonical({key: config[key] for key in fixed}) != upstream._canonical(fixed):
        raise ValueError('prospective screen contract changed')
    control = config['matched_control']
    expected_control = _control_fixed()
    if not isinstance(control, dict) or set(control) != set(expected_control) | {
            'alpha', 'state_sha256', 'artifact_path', 'artifact_sha256'}:
        raise ValueError('matched control contract incomplete')
    if upstream._canonical({key: control[key] for key in expected_control}) != upstream._canonical(expected_control):
        raise ValueError('matched control geometry or calibration changed')
    alpha = control['alpha']
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('finite positive artifact-derived alpha required')
    _digest(control['state_sha256']); _digest(control['artifact_sha256'])
    artifact = _resolve(control['artifact_path'], STAGE / 'evidence')
    export = config['upstream_export']
    if not isinstance(export, dict) or set(export) != {
            'manifest_path', 'manifest_sha256', 'tensor_path', 'tensor_sha256',
            'source_objectives', 'tensor_hashes'}:
        raise ValueError('exact upstream export binding required')
    if export['source_objectives'] != {'XG': 'B3', 'XS': 'BS', 'XR': 'BR'}:
        raise ValueError('same-pair mandatory controls may not be replaced')
    manifest = _resolve(export['manifest_path'], upstream.EXPORT_ROOT)
    tensor = _resolve(export['tensor_path'], upstream.EXPORT_ROOT)
    if manifest.name != 'manifest.json' or tensor != manifest.parent / 'bridges.pt':
        raise ValueError('export paths must identify one unchanged subset')
    for key in ('manifest_sha256', 'tensor_sha256'):
        _digest(export[key])
    if not isinstance(export['tensor_hashes'], dict) or set(export['tensor_hashes']) != {'XG', 'XS', 'XR'}:
        raise ValueError('all three tensor hash inventories required')
    for hashes in export['tensor_hashes'].values():
        if not isinstance(hashes, dict) or set(hashes) != set(upstream.SHAPES):
            raise ValueError('all four bridge tensor hashes required')
        for value in hashes.values():
            _digest(value)
    _digest(config['dataset_manifest_sha256'])
    if verify_artifacts:
        from .provenance import require_committed
        bank = upstream._read(DATASET)
        if (upstream._sha(DATASET) != config['dataset_manifest_sha256']
                or bank['manifest_sha256'] != BANK_CONTRACT or bank['scorer_sha256'] != SCORER):
            raise ValueError('frozen Track B bank changed')
        require_committed(DATASET)
        actual_export = _verify_export(manifest)
        if actual_export != export or _verify_calibration(artifact, actual_export) != control:
            raise ValueError('artifact-derived configuration changed')
    return copy.deepcopy(config)
