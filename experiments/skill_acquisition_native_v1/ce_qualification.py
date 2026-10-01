"""Scope-bound CE qualification requests and model-free producer interfaces.

No framework, model, worker or native dispatch is imported or selected here.
Completed qualification is an OUTPUT, never a prerequisite of qualification.
Metadata alone cannot establish execution: registered backends and retained raw
artifacts/cost seals remain required. Pure orchestration fixtures cannot qualify
a native model or supply measured prices.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import uuid

REQUEST = 'CE8-CONTROLLER-REQUEST-1'
CONTEXT = 'CE8-OUTER-ADMISSION-CONTEXT-1'
SCOPE = 'CE8-NATIVE-QUALIFICATION-SCOPE-1'
RECEIPT = 'CE8-NATIVE-QUALIFICATION-1'
STAGES = ('ingestion_tokenizer', 'runtime_anchor', 'longest_masks',
          'likelihood_gradient_parity', 'adam_clipping', 'checkpoint_resume',
          'delta_roundtrip', 'resource_io_measurement')
REFERENCES = ('selectedoperatorchoice_ref', 'recipe_ref', 'scope_ref', 'source_ref',
    'runtime_ref', 'anchors_ref', 'evaluation_ref', 'preparation_ref', 'plan_ref',
    'ingestion_ref', 'resources_ref', 'attempt_caps_ref', 'scope_authorization_ref')
COST_FIELDS = ('wall_seconds_upper', 'operation_attempts_upper', 'io_read_bytes_upper',
              'io_write_bytes_upper', 'identity_hash_attempts_upper',
              'identity_hash_bytes_upper', 'metadata_bytes_upper')
CHOICES = ('canonical_only', 'paired_canonical_permuted')
MAX_JSON = 4 * 1024 ** 2
NATIVE_BACKEND_IMPLEMENTED = False  # this request harness never supplies a native execution authority
# Only a reviewed source change can register an actual verifier. Neither a
# request Boolean nor an injected callback/opaque object can populate this.
_REGISTERED_NATIVE_VERIFIERS = frozenset()
ACTUAL_ACTIVATION_GAPS = {
    'observation_artifact_readback': 'Recompute each numerical/state observation from its retained raw artifact; unrelated blob hashes do not prove observations.',
    'retained_ledger_replay': 'Replay the retained attempt/hash chain and prove all failed/unknown reservations remain charged; a ledger hash alone is insufficient.',
    'verification_budget': 'Reserve and enforce artifact/ledger readback and verifier hash/IO/time/metadata costs before that verification work.',
    'native_execution_bitwise_proof': 'Verify source-bound live/terminal native execution handles and fresh-process parameter/Adam/module/CUDA/RNG proofs against raw artifacts.',
    'registered_actual_verifier': 'Implement and source-register that actual verifier; this module contains no implementation or registration.',
}
WARM_STOP = 'dceec5aba2cf06645f19ebb9c9afc0f10ebd618c4cd3fd625af88b3de0999784'
PINS = {'torch': '2.10.0+cu126', 'transformers': '4.57.1',
        'tokenizers': '0.22.2', 'safetensors': '0.7.0'}


class RequestRejected(ValueError): pass
def require(condition, message):
    if not condition: raise RequestRejected(message)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def hex64(value): return type(value) is str and len(value) == 64 and not set(value) - set('0123456789abcdef')
def integer(value, minimum=0): return type(value) is int and value >= minimum
def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0 and (not positive or value > 0)


def _costs(value, *, stage=False):
    expected = set(COST_FIELDS) | ({'max_attempts'} if stage else set())
    require(type(value) is dict and set(value) == expected
            and finite(value['wall_seconds_upper'], True)
            and all(integer(value[k], 1 if k in ('operation_attempts_upper', 'metadata_bytes_upper', 'max_attempts') else 0)
                    for k in expected - {'wall_seconds_upper'}), 'explicit finite qualification cost bounds')
    return value


def _ref_shape(reference):
    require(type(reference) is dict and set(reference) >= {'path', 'sha256'}
            and type(reference['path']) is str and hex64(reference['sha256']), 'explicit bound reference')


def _read(reference, observer, costs):
    _ref_shape(reference)
    path = Path(reference['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'regular absolute bound JSON reference')
    size = path.stat().st_size
    require(size <= MAX_JSON and costs['io_read_bytes_upper'] >= size
            and costs['identity_hash_attempts_upper'] >= 1 and costs['identity_hash_bytes_upper'] >= size,
            'bootstrap read/hash bound before work')
    identifier = uuid.uuid4().hex
    observer('qualification_metadata', 'started', operation_id=identifier, path=str(path),
             read_bytes_upper=size, identity_hash_attempts_upper=1, identity_hash_bytes_upper=size)
    start = time.perf_counter()
    try:
        with path.open('rb') as stream: raw = stream.read(size + 1)
        require(len(raw) == size and hashlib.sha256(raw).hexdigest() == reference['sha256'], 'bound metadata bytes changed')
        def unique(pairs):
            out = {}
            for key, value in pairs:
                require(key not in out, 'duplicate JSON key'); out[key] = value
            return out
        value = json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(RequestRejected('nonfinite JSON')))
        require(type(value) is dict, 'bound JSON object required')
    except BaseException:
        observer('qualification_metadata', 'failed', operation_id=identifier,
                 known_bytes_read=size, wall_s=time.perf_counter() - start)
        raise
    observer('qualification_metadata', 'returned', operation_id=identifier,
             bytes_read=size, identity_hash_attempts_actual=1, identity_hash_bytes_actual=size,
             wall_s=time.perf_counter() - start)
    costs['io_read_bytes_upper'] -= size
    costs['identity_hash_attempts_upper'] -= 1; costs['identity_hash_bytes_upper'] -= size
    require(costs['identity_hash_attempts_upper'] >= 0 and costs['identity_hash_bytes_upper'] >= 0,
            'bootstrap hash allowance exhausted')
    return value


def request_preimage(request):
    return {key: value for key, value in request.items() if key != 'scope_authorization_ref'}


def admit_qualification(request, *, code_sha256, outer_admission, io_observer):
    """Verify qualification-only metadata after priced outer bootstrap admission.

    Unapproved/missing choice/scope refuses before reference IO or framework
    imports. There is deliberately no completed qualification input dependency.
    The returned context is evidence for later backend/resource admission,
    never an implicit model-load or production-training permit.
    """
    require(type(request) is dict and request.get('schema') == REQUEST and request.get('phase') == 'qualification', 'qualification request schema/phase')
    require(request.get('approved') is True and request.get('qualification_authorized') is True
            and request.get('training_authorized') is False
            and request.get('status') == 'AUTHORIZED_QUALIFICATION_ONLY', 'qualification scope unapproved; no imports or IO')
    require(request.get('arm') in ('canonical', 'permuted')
            and request.get('selectedoperatorchoice_ref') is not None and request.get('scope_ref') is not None,
            'operator choice and separate qualification scope required')
    require(request.get('qualifiedreceipt_ref') is None, 'qualification cannot require its own completed output')
    for key in REFERENCES: _ref_shape(request.get(key))
    bounds = dict(_costs(request.get('bootstrap_bounds')))
    require(callable(outer_admission) and callable(io_observer), 'priced outer admission and IO observer required')
    whole_sha = digest(request_preimage(request))
    ref_map = {key: request[key] for key in REFERENCES}
    outer = outer_admission('qualification_metadata', ref_map)
    require(type(outer) is dict and outer.get('schema') == 'CE8-OUTER-ADMISSION-1'
            and outer.get('admitted') is True and outer.get('mode') in ('synthetic_fixture', 'registered_scope')
            and outer.get('phase') == 'qualification' and outer.get('arm') == request['arm']
            and outer.get('request_sha256') == whole_sha
            and outer.get('scope_sha256') == request['scope_ref']['sha256']
            and outer.get('source_sha256') == request['source_ref']['sha256']
            and outer.get('runtime_sha256') == request['runtime_ref']['sha256'], 'outer admission provenance')
    bindings = {key[:-4]: _read(request[key], io_observer, bounds) for key in REFERENCES}
    choice, scope = bindings['selectedoperatorchoice'], bindings['scope']
    require(choice.get('schema') == 'CE8-OPERATOR-CHOICE-1' and choice.get('approved') is True
            and choice.get('choice') in CHOICES and choice.get('new_experiment') is True
            and choice.get('preserved_stop_receipt_sha256') == WARM_STOP,
            'new operator decision preserving the stopped study required')
    require(choice['choice'] != 'canonical_only' or request['arm'] == 'canonical', 'permuted arm outside selected experiment')
    require(scope.get('schema') == SCOPE and scope.get('role') == 'qualification_only'
            and scope.get('approved') is True and scope.get('qualification_authorized') is True
            and scope.get('production_training_authorized') is False and scope.get('arm') == request['arm']
            and scope.get('choice_sha256') == request['selectedoperatorchoice_ref']['sha256'], 'separate arm-bound qualification scope')
    auth = bindings['scope_authorization']
    require(auth.get('approved') is True and auth.get('qualification_authorized') is True
            and auth.get('training_authorized') is False
            and auth.get('request_preimage_sha256') == digest(request_preimage(request)), 'exact qualification authorization')
    source = bindings['source']
    require(type(code_sha256) is dict and code_sha256 and all(hex64(v) for v in code_sha256.values())
            and source.get('code_sha256') == code_sha256, 'frozen source bundle changed')
    recipe = bindings['recipe'].get('recipe')
    require(type(recipe) is dict and digest(recipe) == scope.get('static_recipe_sha256')
            == outer.get('static_recipe_sha256'), 'static CE recipe scope/source binding')
    require(recipe.get('accumulation_rows') == 8 and type(recipe['accumulation_rows']) is int
            and recipe.get('microbatch_rows') == 1 and recipe.get('loss_reduction') == 'mean_labeled_tokens_per_row_then_equal_rows'
            and recipe.get('gradient_checkpointing') == {'use_reentrant': False}
            and recipe.get('gradient_clipping') == {'max_norm': 1.0, 'norm_type': 2.0, 'foreach': False, 'error_if_nonfinite': True}, 'CE recipe mechanics')
    optimizer = recipe.get('optimizer') or {}
    require(optimizer.get('class') == 'torch.optim.AdamW' and finite(optimizer.get('lr'), True)
            and finite(optimizer.get('eps'), True) and optimizer.get('weight_decay') == 0
            and type(optimizer.get('betas')) is list and len(optimizer['betas']) == 2
            and all(finite(v) and v < 1 for v in optimizer['betas'])
            and optimizer.get('foreach') is False and optimizer.get('capturable') is True
            and recipe.get('optimizer_origin') == 'FRESH_PER_ARM'
            and recipe.get('target_eos_id') == 151645 and recipe.get('eos_ids') == [151645, 151643]
            and recipe.get('pad_id') == 151643 and recipe.get('max_target_tokens') == 128,
            'explicit native CE Adam/reset/target contract')
    runtime = bindings['runtime']
    preimage = runtime.get('decoding_preimage') or {}
    require(preimage.get('device') == 'cuda:0' and preimage.get('dtype') == 'float32'
            and preimage.get('packages') == PINS and preimage.get('deterministic_algorithms') is True
            and preimage.get('tf32_cuda_matmul') is False and preimage.get('tf32_cudnn') is False
            and runtime.get('decoding_sha256') == digest(preimage), 'pinned FP32 deterministic runtime')
    anchors = bindings['anchors']
    require(anchors.get('own_start_sha256') == anchors.get('common_anchor_sha256')
            and all(hex64(anchors.get(key)) for key in ('own_start_sha256', 'common_anchor_sha256', 'base_weights_sha256', 'warm_delta_sha256', 'tokenizer_receipt_sha256')), 'exact common deployed anchor/tokenizer binding')
    plan = bindings['plan']
    from . import ce_plan
    rows = ce_plan.validate_plan(plan)
    require(plan['mode'] == 'unapproved_proposal' and plan['label_binding']['arm'] == request['arm']
            and len(rows) == 375, 'full selected 3000-row proposal plan binding')
    preparation = bindings['preparation']
    require(plan['preparation']['sha256'] == request['preparation_ref']['sha256']
            and preparation.get('approved') is False and preparation.get('training_authorized') is False, 'prepared target source binding; preparation is not a grant')
    specification = bindings['ingestion']
    require(specification.get('schema') == 'CE8-INGESTION-SPEC-1'
            and specification.get('arm') == request['arm'] and specification.get('plan_sha256') == plan['plan_sha256']
            and specification.get('preparation_ref') == request['preparation_ref']
            and specification.get('tokenizer_receipt_sha256') == anchors['tokenizer_receipt_sha256'], 'declared ingestion specification binding')
    expected = specification.get('expected_census')
    require(type(expected) is dict and expected == declared_census(preparation, request['arm']), 'actual preparation census specification')
    caps = bindings['attempt_caps']
    require(caps.get('schema') == 'CE8-QUALIFICATION-CAPS-1' and set(caps.get('stage_caps') or {}) == set(STAGES), 'all bounded qualification stages')
    for value in caps['stage_caps'].values(): _costs(value, stage=True)
    _costs(caps.get('global_caps'))
    require(scope.get('attempt_caps_sha256') == request['attempt_caps_ref']['sha256']
            and scope.get('resources_sha256') == request['resources_ref']['sha256'], 'approved caps/resource envelope binding')
    require(type(scope.get('gradient_reference')) is dict
            and finite(scope['gradient_reference'].get('absolute_tolerance'), True)
            and finite(scope['gradient_reference'].get('relative_tolerance'), True)
            and scope['gradient_reference'].get('norm') == 'l2', 'explicit authorized gradient metrics/tolerances')
    require(finite(scope.get('likelihood_tolerance_nats'), True)
            and scope.get('allowed_data_roles') == ['train'], 'bounded likelihood criterion and TRAIN-only qualifier')
    return {'schema': CONTEXT, 'mode': outer['mode'], 'phase': 'qualification', 'arm': request['arm'], 'request_sha256': whole_sha,
        'scope_sha256': request['scope_ref']['sha256'], 'operator_choice': choice['choice'], 'refs': ref_map,
        'static_recipe_sha256': scope['static_recipe_sha256'],
        'bindings': bindings, 'caps': caps, 'stage_plan': stage_plan(caps), 'outer_admission': outer,
        'admission_evidence_verified': True, 'production_training_authorized': False,
        'requires_separate_backend_admission': True, 'bootstrap_remaining': bounds}


def declared_census(preparation, arm):
    source = preparation['census'][arm]
    return {key: source[key] for key in ('rows_rendered', 'total_prompt_tokens', 'total_supervised_tokens',
            'total_sequence_positions', 'max_sequence_positions', 'prompt_render_sha256', 'sequence_ids_sha256')} | {
        'max_prompt_tokens': source['prompt_tokens']['max'], 'max_labeled_tokens': source['labelled_tokens']['max']}


def stage_plan(caps):
    return [{'stage': stage, 'ordinal': index, 'caps': dict(caps['stage_caps'][stage]),
             'required_predecessors': list(STAGES[:index])} for index, stage in enumerate(STAGES)]


def validate_actual_ingestion(context, value):
    """Select independent actual witnesses; never fabricate maxP+maxC rows."""
    anchors, plan = context['bindings']['anchors'], context['bindings']['plan']
    require(value.get('schema') == 'CE8-INGESTION-1' and value.get('tokenizer_rendered') is True
            and value.get('model_loads') == value.get('optimizer_steps') == 0
            and value.get('native_qualified') is False and value.get('training_authorized') is False
            and value.get('arm') == context['arm'] and value.get('plan_sha256') == plan['plan_sha256']
            and value.get('static_recipe_sha256') == context['bindings']['scope']['static_recipe_sha256']
            and value.get('tokenizer_receipt_sha256') == anchors['tokenizer_receipt_sha256']
            and value.get('preparation_ref') == context['refs']['preparation_ref']
            and value.get('plan_ref') == context['refs']['plan_ref'], 'actual ingestion receipt provenance')
    expected = context['bindings']['ingestion']['expected_census']
    census = value.get('census') or {}
    mapped = dict(census, rows_rendered=census.get('rows'))
    require(all(mapped.get(key) == actual for key, actual in expected.items()), 'rendered census differs from prepared specification')
    rows = value.get('prepared_row_bindings')
    ids = [rid for boundary in plan['ordered_plan'] for rid in boundary['row_ids']]
    require(type(rows) is list and [r.get('id') for r in rows] == ids, 'all exact ordered actual token rows')
    targets = dict(pair for boundary in plan['label_binding']['boundaries'] for pair in boundary['row_target_pairs'])
    for row in rows:
        require(all(integer(row.get(k), 1) for k in ('prompt_tokens', 'labeled_tokens', 'full_sequence_tokens'))
                and row['full_sequence_tokens'] == row['prompt_tokens'] + row['labeled_tokens']
                and all(hex64(row.get(k)) for k in ('prompt_ids_sha256', 'completion_ids_sha256', 'prepared_row_sha256', 'render_sha256', 'target_sha256')), 'actual prepared row binding')
        require(row['target_sha256'] == targets[row['id']], 'actual prepared row teacher differs from bound label plan')
    totals = {'total_prompt_tokens': sum(row['prompt_tokens'] for row in rows),
              'total_supervised_tokens': sum(row['labeled_tokens'] for row in rows),
              'total_sequence_positions': sum(row['full_sequence_tokens'] for row in rows)}
    require(all(census.get(key) == actual for key, actual in totals.items()), 'actual prepared row counts differ from summed census')
    for field in ('max_prompt_tokens', 'max_labeled_tokens', 'max_sequence_positions'):
        key = {'max_prompt_tokens': 'prompt_tokens', 'max_labeled_tokens': 'labeled_tokens', 'max_sequence_positions': 'full_sequence_tokens'}[field]
        maximum = max(row[key] for row in rows); witness = value['witnesses'][field]
        require(maximum == census[field] == witness.get('value')
                and witness.get('row_ids') == [row['id'] for row in rows if row[key] == maximum], 'actual independent maximum witness identity')
    require(value.get('receipt_sha256') == digest({k: v for k, v in value.items() if k != 'receipt_sha256'}), 'ingestion receipt digest')
    return value['witnesses']


def build_stage_command(context, stage, *, request_path, output_dir):
    """Build only a caller-registered worker command; this never launches it."""
    require(context.get('schema') == CONTEXT and context.get('admission_evidence_verified') is True
            and stage in STAGES, 'admitted qualification stage')
    worker = (context['bindings']['source'].get('stage_workers') or {}).get(stage)
    require(type(worker) is dict and type(worker.get('python_executable')) is str
            and Path(worker['python_executable']).is_absolute() and type(worker.get('module')) is str
            and worker['module'] and all(part.isidentifier() for part in worker['module'].split('.'))
            and type(worker.get('extra_args')) is list and all(type(arg) is str for arg in worker['extra_args']),
            'registered stage worker is absent; no default backend')
    require(Path(request_path).is_absolute() and Path(output_dir).is_absolute(), 'owned absolute request/output paths')
    require(not any(arg.split('=', 1)[0] in ('--request', '--stage', '--output-dir') for arg in worker['extra_args']),
            'registered worker cannot override bound request/stage/output arguments')
    return [worker['python_executable'], '-B', '-u', '-m', worker['module'], '--request', str(request_path),
            '--stage', stage, '--output-dir', str(output_dir), *worker['extra_args']]


def _verify_artifact(reference, observer):
    _ref_shape(reference)
    path = Path(reference['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and integer(reference.get('bytes'), 1) and path.stat().st_size == reference['bytes'], 'retained artifact path/size')
    identifier = uuid.uuid4().hex
    observer('qualification_artifact', 'started', operation_id=identifier, path=str(path),
             read_bytes_upper=reference['bytes'], identity_hash_attempts_upper=1,
             identity_hash_bytes_upper=reference['bytes'])
    h = hashlib.sha256(); count = 0
    try:
        with path.open('rb') as stream:
            while block := stream.read(min(1024 ** 2, reference['bytes'] + 1 - count)):
                count += len(block); require(count <= reference['bytes'], 'artifact grew'); h.update(block)
        require(count == reference['bytes'] and h.hexdigest() == reference['sha256'], 'retained artifact bytes/hash changed')
    except BaseException:
        observer('qualification_artifact', 'failed', operation_id=identifier, known_bytes_read=count)
        raise
    observer('qualification_artifact', 'returned', operation_id=identifier, bytes_read=count,
             identity_hash_attempts_actual=1, identity_hash_bytes_actual=count)


def validate_stage_observations(context, stage, observed, ingestion=None):
    """Required numerical/state observations, backed by retained worker artifacts."""
    require(type(observed) is dict, 'qualification stage observations required')
    anchors, scope = context['bindings']['anchors'], context['bindings']['scope']
    if stage == 'ingestion_tokenizer': return validate_actual_ingestion(context, observed['ingestion_receipt'])
    if stage == 'runtime_anchor':
        require(observed.get('actual_anchor_sha256') == anchors['own_start_sha256']
                and observed.get('tokenizer_receipt_sha256') == anchors['tokenizer_receipt_sha256']
                and observed.get('actual_runtime_preimage') == context['bindings']['runtime']['decoding_preimage']
                and observed.get('decoder_checkpointing') == {'use_reentrant': False}
                and observed.get('head_checkpointing') is True, 'actual pinned runtime/anchor/checkpoint paths')
    elif stage == 'longest_masks':
        require(ingestion is not None, 'actual ingestion precedes mask qualification')
        rows = {row['id']: row for row in ingestion['prepared_row_bindings']}
        wanted = set(rid for value in ingestion['witnesses'].values() for rid in value['row_ids'])
        masks = observed.get('rows')
        require(type(masks) is list and {r.get('id') for r in masks} == wanted, 'all actual maxP/maxC/maxS witness masks')
        for mask in masks:
            row = rows[mask['id']]; P, C = row['prompt_tokens'], row['labeled_tokens']
            require(mask.get('prepared_row_sha256') == row['prepared_row_sha256']
                    and mask.get('predictor_positions') == list(range(P - 1, P + C - 1))
                    and mask.get('target_positions') == list(range(P, P + C))
                    and mask.get('prompt_selected') == mask.get('pad_selected') == 0 and mask.get('eos_selected') == 1,
                    'actual one-shift target/EOS mask')
    elif stage == 'likelihood_gradient_parity':
        require(ingestion is not None, 'actual witnesses precede likelihood comparison')
        targets = {row['id']: row['labeled_tokens'] for row in ingestion['prepared_row_bindings']}
        wanted = set(rid for value in ingestion['witnesses'].values() for rid in value['row_ids'])
        comparisons = observed.get('per_token_comparisons')
        tolerance = scope.get('likelihood_tolerance_nats')
        require(finite(tolerance, True) and type(comparisons) is list
                and {r.get('id') for r in comparisons} == wanted, 'authorized likelihood tolerance/all actual witnesses')
        for row in comparisons:
            require(integer(row.get('positions'), 1) and row['positions'] == targets[row['id']]
                    and finite(row.get('max_abs_nats')) and row['max_abs_nats'] <= tolerance,
                    'per-token independent likelihood parity including EOS')
        metric = scope['gradient_reference']; reference = observed.get('gradient_reference') or {}
        # The absolute-plus-relative criterion applies to the declared L2
        # norm, including a zero reference; max-coordinate error is separate.
        require(reference.get('norm') == metric['norm'] and reference.get('fixed_loss_denominator') == 8
                and all(finite(reference.get(k)) for k in ('reference_norm', 'difference_norm', 'max_abs_error'))
                and reference['max_abs_error'] <= metric['absolute_tolerance']
                and reference['max_abs_error'] <= reference['difference_norm']
                and reference['difference_norm'] <= metric['absolute_tolerance']
                    + metric['relative_tolerance'] * reference['reference_norm'], 'independent equal-row gradient reference')
    elif stage == 'adam_clipping':
        require(integer(observed.get('actual_optimizer_steps'), 2)
                and observed.get('applied_U') == list(range(1, observed['actual_optimizer_steps'] + 1))
                and observed.get('finite_confined_gradients') is True and observed.get('frozen_state_unchanged') is True
                and observed.get('clip_before_adam') is True and observed.get('clip_max_norm') == 1.0,
                'actual finite scoped clipped Adam steps')
    elif stage == 'checkpoint_resume':
        require(integer(observed.get('checkpoint_saves'), 2) and integer(observed.get('checkpoint_restores'), 1)
                and integer(observed.get('actual_processes'), 2)
                and all(observed.get(k) is True for k in ('parameters_bitwise', 'adam_bitwise', 'module_modes_bitwise',
                    'python_rng_bitwise', 'numpy_rng_bitwise', 'torch_cpu_rng_bitwise', 'torch_cuda_rng_bitwise',
                    'fresh_fence_installed', 'stale_work_rejected', 'failed_unknown_costs_retained', 'corruption_fails_closed')),
                'native fresh-process checkpoint/RNG/fence proof')
    elif stage == 'delta_roundtrip':
        require(observed.get('own_start_sha256') == anchors['own_start_sha256']
                and observed.get('common_reference_sha256') == anchors['common_anchor_sha256']
                and observed.get('delta_dtype') == 'float32' and observed.get('delta_rebuilt_verified') is True
                and all(hex64(observed.get(k)) for k in ('delta_file_sha256', 'delta_data_sha256', 'final_master_sha256', 'rebuilt_endpoint_sha256'))
                and finite(observed.get('rounding_residual_max_abs'))
                and finite(observed.get('rounding_residual_l2')), 'common-anchor delta and reported deployed-rounding proof')
    elif stage == 'resource_io_measurement':
        require(observed.get('measured') is True and type(observed.get('operation_measurements')) is list
                and observed['operation_measurements'] and all(type(row) is dict
                    and finite(row.get('elapsed_seconds')) and all(integer(row.get(k)) for k in
                        ('io_read_bytes', 'io_write_bytes', 'identity_hash_attempts', 'identity_hash_bytes'))
                    for row in observed['operation_measurements']), 'actual elapsed/hash/IO measurements')
        require(all(finite(observed.get(k), True) for k in ('peak_worker_rss_gib', 'gpu_total_gib', 'peak_reserved_gib', 'min_gpu_free_gib', 'min_disk_free_bytes'))
                and observed.get('allocator_fraction') == .78 and observed.get('reserved_fraction_limit') == .95
                and observed['peak_reserved_gib'] <= .95 * .78 * observed['gpu_total_gib']
                and observed['min_gpu_free_gib'] >= .5, 'measured native resource fit')
    else: raise RequestRejected('unknown qualification stage')


def producer_receipt(context, stage_records, cost_seal, *, backend_mode, outer_admission=None, io_observer=None,
                     native_verifier=None):
    """Assemble registered execution evidence; fixtures never become native PASS.

    Controller-owned records retain each started/failed/returned attempt, its
    raw artifacts and reservation. Missing, pending or over-cap work refuses.
    Stage observation interpretation is a registered backend responsibility,
    bound by code, scope and raw artifact digests, not inferred from CPU tests.
    """
    require(backend_mode == 'registered_scope', 'synthetic orchestration is not native qualification')
    if not NATIVE_BACKEND_IMPLEMENTED or not any(native_verifier is verifier for verifier in _REGISTERED_NATIVE_VERIFIERS):
        return {'schema': RECEIPT, 'status': 'DRAFT_UNQUALIFIED', 'passed': False,
            'native_qualified': False, 'production_training_authorized': False,
            'request_sha256': context.get('request_sha256'), 'scope_sha256': context.get('scope_sha256'),
            'arm': context.get('arm'), 'attempts': stage_records, 'cost_seal': cost_seal,
            'activation_gap': 'No implemented registered native backend/execution-handle verifier. Mode/booleans cannot establish execution.',
            'actual_activation_gaps': dict(ACTUAL_ACTIVATION_GAPS)}
    require(callable(outer_admission) and callable(io_observer), 'priced registered artifact verification required')
    # This is an unimplemented, source-owned execution verifier interface.
    # The empty registry above keeps it unreachable. A future registration must
    # implement all gaps, rather than infer truth from stage dictionary flags.
    require(callable(native_verifier), 'implemented registered native verifier required')
    native_verifier(context, stage_records, cost_seal,
                    outer_admission=outer_admission, io_observer=io_observer)
    authority = outer_admission('qualification_finalize', context['refs'])
    require(authority.get('schema') == 'CE8-OUTER-ADMISSION-1' and authority.get('mode') == 'registered_scope'
            and authority.get('admitted') is True and authority.get('request_sha256') == context['request_sha256']
            and authority.get('scope_sha256') == context['scope_sha256'], 'registered finalization authority')
    require(type(stage_records) is list and stage_records, 'retained qualification attempts required')
    successful = {}; total = dict.fromkeys(COST_FIELDS, 0); attempts = dict.fromkeys(STAGES, 0)
    for row in stage_records:
        require(row.get('stage') in STAGES and row.get('request_sha256') == context['request_sha256']
                and row.get('scope_sha256') == context['scope_sha256'] and row.get('arm') == context['arm']
                and row.get('status') in ('returned', 'failed'), 'sealed stage attempt provenance; unknown work cannot certify')
        costs = _costs(row.get('reserved_costs'))
        attempts[row['stage']] += 1
        declared = context['caps']['stage_caps'][row['stage']]
        require(attempts[row['stage']] <= declared['max_attempts']
                and all(costs[k] <= declared[k] for k in COST_FIELDS), 'declared stage attempt/cost ceiling')
        for field, value in costs.items(): total[field] += value
        artifacts = row.get('raw_artifacts')
        require(type(artifacts) is list and artifacts and all(type(a) is dict and hex64(a.get('sha256'))
                and type(a.get('path')) is str and Path(a['path']).is_absolute() for a in artifacts), 'retained raw stage artifacts')
        for artifact in artifacts: _verify_artifact(artifact, io_observer)
        if row['status'] == 'returned':
            require(row.get('passed') is True and row.get('native_actual') is True
                    and all(stage in successful for stage in STAGES[:STAGES.index(row['stage'])]), 'native returned ordered stage proof')
            require(row.get('code_sha256') == context['bindings']['source']['code_sha256']
                    and row.get('static_recipe_sha256') == context['bindings']['scope']['static_recipe_sha256'], 'executed backend source/recipe binding')
            ingestion = successful.get('ingestion_tokenizer', {}).get('observations', {}).get('ingestion_receipt')
            validate_stage_observations(context, row['stage'], row.get('observations'), ingestion)
            successful[row['stage']] = row
    require(set(successful) == set(STAGES)
            and all(total[k] <= context['caps']['global_caps'][k] for k in COST_FIELDS), 'all measured qualification stages within declared caps')
    require(type(cost_seal) is dict and cost_seal.get('schema') == 'CE8-QUALIFICATION-COST-SEAL-1'
            and cost_seal.get('request_sha256') == context['request_sha256'] and cost_seal.get('scope_sha256') == context['scope_sha256']
            and cost_seal.get('attempts_sha256') == digest(stage_records) and cost_seal.get('reserved_totals') == total
            and cost_seal.get('completed') is True and hex64(cost_seal.get('retained_ledger_sha256')), 'qualification execution cost seal')
    ledger_ref = cost_seal.get('retained_ledger_ref')
    require(type(ledger_ref) is dict and ledger_ref.get('sha256') == cost_seal['retained_ledger_sha256'], 'actual retained cost ledger binding')
    _verify_artifact(ledger_ref, io_observer)
    body = {'schema': RECEIPT, 'phase': 'qualification', 'passed': True, 'native_actual': True,
        'production_training_authorized': False, 'request_sha256': context['request_sha256'],
        'scope_sha256': context['scope_sha256'], 'arm': context['arm'], 'source_ref': context['refs']['source_ref'],
        'runtime_ref': context['refs']['runtime_ref'], 'anchors_ref': context['refs']['anchors_ref'],
        'static_recipe_sha256': context['bindings']['scope']['static_recipe_sha256'],
        'attempts': stage_records, 'cost_seal': cost_seal,
        'classification': 'controlled registered native qualification evidence; not acquisition/retention or a training grant'}
    return dict(body, receipt_sha256=digest(body))
