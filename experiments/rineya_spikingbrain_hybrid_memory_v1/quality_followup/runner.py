"""Exact prospective quality pilot; defaults are offline and model-free.

The eleven-command proposal is separate from SBHM. This module does not import
torch, pretrained-model code, or GPU utilities until the native grant passes.
Each native subprocess has an exclusive output and every failure stops the run.
"""
from pathlib import Path
from functools import wraps
import subprocess
import time

from . import CONDITIONS, RECIPE, REGISTRATION, SCHEMA
from .protocol import (artifact, authorize, bind, load_config, native_admission,
                       native_config, require_ownership, require_qualification,
                       require_review)
from .utils import (Blocked, PACKAGE_ROOT, code_inventory, digest, external_path, now,
                    read_json, require, sha, write_json)


def _translate_native_guards(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except RuntimeError as exc:
            # Avoid importing the heavy SBHM protocol merely to catch its class.
            if exc.__class__.__module__ == 'sbhm.protocol' and exc.__class__.__name__ == 'Blocked':
                raise Blocked(str(exc)) from exc
            raise
    return guarded


def _commands(config_path, config):
    root = external_path(config['native_run_root'])
    prefix = [str(Path(config['native_python']).resolve()), '-B', '-m', 'quality_followup']
    commands = []

    def add(name, stage, condition, qualification=None, delta=None):
        output = root / name
        args = prefix + ['stage', '--config', str(Path(config_path).resolve()),
                         '--stage', stage, '--condition', condition, '--output', str(output),
                         '--allow-model-load']
        if stage == 'train':
            args += ['--allow-training']
        if qualification:
            args += ['--qualification', str(root / qualification / 'receipt.json')]
        if delta:
            args += ['--delta', str(root / delta)]
        commands.append({'name': name, 'stage': stage, 'condition': condition,
                         'output': str(output), 'qualification': None if not qualification else
                         str(root / qualification / 'receipt.json'),
                         'delta': None if not delta else str(root / delta), 'argv': args})

    add('anchor-qualification', 'qualify', 'anchor')
    add('anchor-evaluation', 'evaluate', 'anchor', 'anchor-qualification')
    add('control-initial-qualification', 'qualify', 'control')
    add('candidate-initial-qualification', 'qualify', 'candidate')
    for condition in ('control', 'candidate'):
        add(f'{condition}-train', 'train', condition, f'{condition}-initial-qualification')
        add(f'{condition}-trained-qualification', 'qualify', condition, delta=f'{condition}-train')
        add(f'{condition}-evaluation', 'evaluate', condition,
            f'{condition}-trained-qualification', f'{condition}-train')
    reports = [str(root / f'{c}-evaluation' / 'receipt.json') for c in CONDITIONS]
    commands.append({'name': 'comparison', 'stage': 'compare', 'condition': None,
                     'output': str(root / 'comparison.json'), 'qualification': None,
                     'delta': None, 'argv': prefix + ['compare', '--config', str(Path(config_path).resolve()),
                     '--reports'] + reports + ['--output', str(root / 'comparison.json')]})
    return commands


def build_plan(config_path):
    """Validate and hash the exact proposal without creating any native output."""
    config = load_config(config_path)
    base = bind(config, 'anchor')
    bindings = {condition: {**base, 'condition': condition,
                'arm': 'A' if condition == 'anchor' else 'B'} for condition in CONDITIONS}
    proposal = {'schema': SCHEMA + '/proposal', 'study_id': config['study_id'],
                'status': 'PLAN_ONLY', 'config_path': str(Path(config_path).resolve()),
                'native_run_root': str(external_path(config['native_run_root'])),
                'bindings': bindings, 'code_inventory': code_inventory(),
                'recipe': RECIPE, 'registration': REGISTRATION,
                'commands': _commands(config_path, config),
                'counts': {'qualifications': 5, 'training_stages': 2, 'steps_per_training_stage': 512,
                           'dev_measurements': 3, 'rows_per_measurement': 672, 'comparisons': 1},
                'guards': {'admission_available_ram_gib': 7.4, 'runtime_available_ram_gib': 6.0,
                           'maximum_worker_rss_gib': 4.0, 'minimum_free_vram_gib': 1.0,
                           'device': 'cuda:0', 'qualification_max_age_hours': 24,
                           'ownership_max_age_minutes': 5},
                'grant_requirements': {'schema': SCHEMA + '/native-grant',
                    'grant_path': config['grant'], 'maximum_lifetime_days': 7,
                    'separate_authorization_required': True,
                    'actions': [{'stage': stage, 'condition': condition} for condition in CONDITIONS
                                for stage in ('qualify', 'train', 'evaluate')
                                if not (condition == 'anchor' and stage == 'train')]},
                'execution': False, 'model_loaded': False, 'promotion': False,
                'warnings': ['One synthetic DEV panel and one seed; no broad quality or retention claim.',
                             'Stop on failure; no automatic retries, expansion, or follow-up campaign.',
                             'Implementation approval does not authorize these native commands.']}
    proposal['proposal_sha256'] = digest(proposal)
    return proposal


def _require_exact_operation(proposal, stage, condition, output, qualification, delta):
    out = external_path(output)
    candidates = [item for item in proposal['commands'] if item['stage'] == stage and
                  item['condition'] == condition and Path(item['output']).resolve() == out]
    require(len(candidates) == 1, 'stage output is not an exact reviewed proposal operation')
    operation = candidates[0]
    for field, value in (('qualification', qualification), ('delta', delta)):
        expected = operation[field]
        require((value is None and expected is None) or
                (value is not None and expected is not None and
                 Path(value).resolve() == Path(expected).resolve()),
                f'{field} does not match the exact reviewed proposal')
    require(not out.exists(), 'refusing to overwrite or resume a native stage')
    return operation


def _anchor_gate(config, proposal):
    """The common panel stays frozen even when this instrument is inadequate."""
    from .corpus import load_packet
    from .metrics import anchor_retention_gate, score_prediction
    path = external_path(config['native_run_root']) / 'anchor-evaluation' / 'receipt.json'
    require(path.is_file(), 'frozen anchor DEV measurement must precede any training')
    receipt = read_json(path)
    expected = proposal['bindings']['anchor']
    require(receipt.get('status') == 'DEV_MEASUREMENT_ONLY' and receipt.get('binding') == expected and
            receipt.get('training_record') is None and receipt.get('delta_record') is None,
            'anchor measurement is unbound or adapted')
    packet = load_packet(config['corpus_packet'])
    rows = receipt.get('predictions', [])
    require(len(rows) == len(packet['dev']) == 672, 'anchor DEV panel incomplete')
    for observed, row in zip(rows, packet['dev']):
        require((observed.get('id'), observed.get('family_id'), observed.get('target')) ==
                (row['id'], row['family_id'], row['target']), 'anchor row alignment changed')
        scores = score_prediction(observed.get('prediction'), row['target'])
        require(all(observed.get(key) == value for key, value in scores.items()), 'anchor scoring changed')
    qualification = receipt.get('qualification', {})
    require(qualification.get('binding') == expected, 'anchor qualification mismatch')
    refs = receipt.get('artifact_refs', {})
    require(refs.get('qualification') == artifact(external_path(config['native_run_root']) /
                'anchor-qualification' / 'receipt.json'), 'anchor qualification artifact changed')
    require_qualification(refs['qualification']['path'], expected)
    gate = anchor_retention_gate(rows, packet['dev'], packet['design'])
    require(gate['adequate'], 'inadequate frozen-anchor retention instrument: need 16 source-correct cases '
            'across four skills separately under strict and typed scoring; stop without changing panel')
    return {'receipt': artifact(path), 'gate': gate}


def runtime_tokens(tokenizer, row, training, ledger=None, condition=None):
    """Check actual input/label hashes immediately before any row forward."""
    from .corpus import encode_lists
    ids, labels = encode_lists(tokenizer, row, training)
    cap = RECIPE['max_train_tokens' if training else 'max_prompt_tokens']
    require(0 < len(ids) <= cap and len(labels) == len(ids), 'actual row geometry exceeds registration')
    record = {'input_tokens': len(ids), 'supervised_positions': sum(x != -100 for x in labels[1:]),
              'input_token_sha256': digest(ids), 'label_token_sha256': digest(labels)}
    if training:
        require(ledger is not None and condition in ('control', 'candidate'), 'bound exposure slot required')
        require(row['id'] == ledger['slot_id'] and digest(row) == ledger[f'{condition}_row_sha256'],
                'training row differs from its frozen exposure slot')
        require(record['input_tokens'] == ledger['input_tokens'] and
                record['supervised_positions'] == ledger['supervised_positions'] and
                record['input_token_sha256'] == ledger['input_sha256'][condition] and
                record['label_token_sha256'] == ledger['labels_sha256'][condition],
                'actual runtime input/label hashes differ from frozen exposure')
    return ids, labels, record


def _cache_expected(prefix_length):
    return 28 * 2 * 8 * 128 * prefix_length * 4


def _training(model, tokenizer, config, binding, packet, out, qualification, qualification_path):
    import torch
    from sbhm import native
    from sbhm.pipeline import recovery_loss
    from sbhm.protocol import CACHE_RELEASE_EVENTS, resources
    condition = binding['condition']
    rows = {row['id']: row for row in packet[condition]}
    require(len(packet['ledger']) == 512 and len(rows) == 512, 'registered training exposure incomplete')
    parameters = native.trainables(model)
    require(parameters and all(f'model.layers.{RECIPE["layer"]}.self_attn.' in name for name in parameters),
            'only declared same-site LoRA parameters may train')
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=RECIPE['learning_rate'],
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0, foreach=False)
    model.eval()
    started_utc, started = now(), time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    records, resource_samples = [], []
    for step, slot in enumerate(packet['ledger']):
        require(slot['step'] == step and slot['slot_id'] in rows, 'ledger order or row inventory changed')
        resource_samples.append(resources(RECIPE['device'], runtime=True))
        row = rows[slot['slot_id']]
        ids, labels, exposure = runtime_tokens(tokenizer, row, True, slot, condition)
        input_ids = torch.tensor([ids], dtype=torch.long, device=RECIPE['device'])
        label_ids = torch.tensor([labels], dtype=torch.long, device=RECIPE['device'])
        optimizer.zero_grad(set_to_none=True)
        loss = recovery_loss(model, input_ids, label_ids)
        require(bool(torch.isfinite(loss)), 'nonfinite training loss')
        loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
                'nonfinite adaptation gradient')
        require(not any(p.grad is not None for p in model.parameters() if not p.requires_grad),
                'a frozen source parameter received a gradient')
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in parameters.values()), 'nonfinite adaptation tensor')
        records.append({'step': step, 'slot_id': slot['slot_id'], 'row_id': row['id'],
                        **exposure, 'loss': loss.item(), 'gradient_norm': float(norm)})
        resource_samples.append(resources(RECIPE['device'], runtime=True))
    receipt = {'status': 'TRAINING_COMPLETED_NOT_EFFICACY', 'stage': 'train', 'binding': binding,
               'started_utc': started_utc, 'created_utc': now(), 'records': records,
               'qualification': qualification, 'qualification_ref': artifact(qualification_path),
               'corpus_sha256': binding['corpus_sha256'], 'ledger_sha256': binding['ledger_sha256'],
               'parameter_names': sorted(parameters),
               'trainable_parameters': sum(p.numel() for p in parameters.values()),
               'optimizer': RECIPE['optimizer'], 'steps': len(records), 'resume_supported': False,
               'training_seconds': time.perf_counter() - started,
               'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(),
               'rss_sample_max_bytes': max(item['rss_bytes'] for item in resource_samples),
               'resource_samples': resource_samples,
               'allocator_cache_release_events': list(CACHE_RELEASE_EVENTS)}
    native.save_delta(model, out, binding, receipt)
    return receipt


def _evaluate(model, tokenizer, config, binding, packet, qualification, qualification_path, delta):
    import torch
    from sbhm.pipeline import generate
    from sbhm.protocol import CACHE_RELEASE_EVENTS, resources
    from .metrics import anchor_retention_gate, score_prediction, summarize_rows
    with torch.no_grad():
        model(input_ids=torch.full((1, 8), 10, device=RECIPE['device'], dtype=torch.long),
              use_cache=False, logits_to_keep=1)
    require(len(packet['dev']) == 672, 'common DEV panel incomplete')
    predictions = []
    for row in packet['dev']:
        ids, _, hashes = runtime_tokens(tokenizer, row, False)
        metadata = packet['design']['dev'][row['id']]
        require(hashes['input_token_sha256'] == metadata['input_token_sha256'] and
                hashes['label_token_sha256'] == metadata['label_token_sha256'] and
                len(ids) == metadata['token_geometry']['prompt_tokens'],
                'actual DEV input/label hashes differ from frozen common panel')
        tensor = torch.tensor([ids], dtype=torch.long, device=RECIPE['device'])
        result = generate(model, tensor, tokenizer, native_config(config), binding['arm'])
        cached_prefix = len(ids) + len(result['generated_ids']) - 1
        require(result['cache_storage_bytes'] == _cache_expected(cached_prefix),
                'full-attention cache storage does not match frozen source geometry')
        result.update(id=row['id'], family_id=row['family_id'], target=row['target'],
                      **score_prediction(result['prediction'], row['target']), **hashes,
                      cache_geometry={'cached_prefix_tokens': cached_prefix,
                                      'expected_cache_storage_bytes': _cache_expected(cached_prefix)})
        predictions.append(result)
        resources(RECIPE['device'], runtime=True)
    refs = {'qualification': artifact(qualification_path)}
    delta_record, training_record = None, None
    if delta is not None:
        directory = external_path(delta)
        delta_record = read_json(directory / 'delta.json')
        training_record = read_json(directory / 'receipt.json')
        require(delta_record.get('record') == training_record, 'training/delta record mismatch')
        refs.update(training=artifact(directory / 'receipt.json'),
                    delta_manifest=artifact(directory / 'delta.json'),
                    delta=artifact(directory / 'delta.safetensors'),
                    initial_qualification=training_record['qualification_ref'])
    report = {'status': 'DEV_MEASUREMENT_ONLY', 'stage': 'evaluate', 'binding': binding,
              'created_utc': now(), 'predictions': predictions,
              'summary': summarize_rows(predictions, packet['dev'], packet['design']),
              'qualification': qualification, 'training_record': training_record,
              'delta_record': delta_record, 'artifact_refs': refs,
              'corpus_sha256': binding['corpus_sha256'],
              'semantic_accuracy': None, 'speedup_claim': False, 'energy_measured': False,
              'promotion': False, 'allocator_cache_release_events': list(CACHE_RELEASE_EVENTS)}
    if binding['condition'] == 'anchor':
        report['retention_instrument_gate'] = anchor_retention_gate(predictions, packet['dev'], packet['design'])
    return report


@_translate_native_guards
def execute_stage(config_path, stage, condition, output, qualification=None, delta=None,
                  allow_model_load=False, allow_training=False):
    # Flags precede even source hashing, so accidental CLI invocation is cheap
    # and cannot import model code or create a partial run directory.
    require(allow_model_load, 'separate native permission and --allow-model-load required')
    require(stage in ('qualify', 'train', 'evaluate') and condition in CONDITIONS, 'unknown operation')
    require(stage != 'train' or (allow_training and condition != 'anchor'),
            'explicit --allow-training required; anchor is never trained')
    config = load_config(config_path)
    proposal = build_plan(config_path)
    operation = _require_exact_operation(proposal, stage, condition, output, qualification, delta)
    binding = bind(config, condition, delta)
    grant = authorize(config, stage, binding, proposal, allow_model_load, allow_training)
    ownership = require_ownership(config)
    # Corpus parsing is pure/offline and must complete before allocating a model.
    from .corpus import load_packet
    packet = load_packet(config['corpus_packet'])
    require(packet['manifest']['files']['ledger']['sha256'] == binding['ledger_sha256'], 'ledger changed')
    require(digest(packet['manifest']['tokenizer']['files']) == binding['tokenizer_sha256'],
            'corpus tokenization is not the exact frozen source tokenizer/config')
    qualified = None if stage == 'qualify' else require_qualification(qualification, binding)
    anchor_gate = _anchor_gate(config, proposal) if stage == 'train' else None
    resources_before = native_admission(config)
    # Permission, owner, exact data/variant, qualification and guards all passed.
    from sbhm import native
    from sbhm.protocol import CACHE_RELEASE_EVENTS, resources
    out = external_path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'request.json', {'created_utc': now(), 'stage': stage, 'binding': binding,
               'operation': operation, 'proposal_sha256': proposal['proposal_sha256'],
               'grant_ref': artifact(config['grant']), 'ownership': ownership,
               'review_ref': artifact(config['review']), 'admission_resources': resources_before,
               'anchor_gate': anchor_gate})
    load_started = time.perf_counter()
    model, tokenizer = native.load_host(native_config(config))
    loading_seconds = time.perf_counter() - load_started
    if stage == 'qualify':
        receipt = native.qualify(model, native_config(config, qualification=True), binding['arm'], binding, delta)
        for key, item in receipt['parity'].items():
            length = int(key.split(':')[1])
            require(item['cache_storage_bytes'] == _cache_expected(length), 'qualification cache geometry changed')
        receipt['allocator_cache_release_events'] = list(CACHE_RELEASE_EVENTS)
        if delta is not None:
            receipt['artifact_refs'] = {'delta': artifact(external_path(delta) / 'delta.safetensors'),
                                       'delta_manifest': artifact(external_path(delta) / 'delta.json'),
                                       'training': artifact(external_path(delta) / 'receipt.json')}
    else:
        native.install(model, native_config(config), binding['arm'])
        native.load_delta(model, delta, binding)
        if stage == 'train':
            receipt = _training(model, tokenizer, config, binding, packet, out, qualified, qualification)
        else:
            receipt = _evaluate(model, tokenizer, config, binding, packet, qualified, qualification, delta)
    # The saved training record must stay byte-for-byte equal to delta.record;
    # stage loading costs therefore live in request/resource receipt, not train.
    final_resources = resources(RECIPE['device'], runtime=True)
    write_json(out / 'receipt.json', receipt)
    write_json(out / 'stage-costs.json', {'created_utc': now(), 'binding': binding,
               'model_loading_seconds': loading_seconds, 'final_resources': final_resources,
               'allocator_cache_release_events': list(CACHE_RELEASE_EVENTS),
               'capability_claim': False})
    return receipt


@_translate_native_guards
def execute_plan(config_path, allow_model_load=False, allow_training=False):
    require(allow_model_load and allow_training, 'native plan needs separate permission and both explicit flags')
    config = load_config(config_path)
    proposal = build_plan(config_path)
    root = external_path(config['native_run_root'])
    require(not root.exists(), 'native run root exists; no overwrite, resume or automatic retry')
    for condition in CONDITIONS:
        for stage in ('qualify', 'evaluate', 'train'):
            if stage == 'train' and condition == 'anchor':
                continue
            authorize(config, stage, proposal['bindings'][condition], proposal, True, True)
    ownership = require_ownership(config)
    native_admission(config)
    root.mkdir(parents=True, exist_ok=False)
    import psutil
    process = psutil.Process()
    write_json(root / 'run-owner-lock.json', {'schema': SCHEMA + '/run-owner-lock',
               'created_utc': now(), 'owner_thread_id': config['owner_thread_id'],
               'config_sha256': digest(config), 'proposal_sha256': proposal['proposal_sha256'],
               'acknowledgement_sha256': ownership['acknowledgement']['sha256'],
               'orchestrator_pid': process.pid, 'orchestrator_create_time': process.create_time()})
    write_json(root / 'execution-request.json', {'created_utc': now(), 'proposal': proposal,
               'grant_ref': artifact(config['grant']), 'promotion': False})
    for command in proposal['commands']:
        # subprocesses isolate loaded models and CUDA allocator state; check=True
        # stops on the first failed command without spawning any successor.
        subprocess.run(command['argv'], cwd=str(PACKAGE_ROOT), check=True)
        if command['name'] == 'anchor-evaluation':
            _anchor_gate(config, proposal)
    return read_json(root / 'comparison.json')
