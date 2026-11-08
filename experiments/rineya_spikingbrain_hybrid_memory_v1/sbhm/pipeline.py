"""Native staged runner: capture -> attention distill -> LM recovery -> evaluation.

All stages are individually authorized and receipt-bound. No implicit downloads,
protected evaluation, full-model training, checkpoint overwrite, or automatic
promotion. Sequential attention backend is for correctness, not speed claims.
"""
from pathlib import Path
import math
import time
import shutil
import torch
from safetensors.torch import save, load_file
from . import native
from .core import cache_bytes
from .data import load_corpus, ordered, encode, validate_evaluation_groups
from .protocol import (Blocked, bind, authorize, require_qualification, resources,
                       read_json, write_json, sha, safe_file, digest, now, CACHE_RELEASE_EVENTS)


def prepare(args):
    config = read_json(args.config)
    binding = bind(config, args.arm, args.delta)
    authorize(config, args.command, binding, args.allow_model_load, args.allow_training)
    if args.command != 'qualify':
        require_qualification(args.qualification, binding)
    if args.command == 'teacher' and (args.arm != 'A' or args.delta):
        raise Blocked('teacher capture must use the exact frozen baseline')
    if args.command in ('distill', 'recover') and args.arm == 'A':
        raise Blocked('arm A is never trained')
    if args.command != 'qualify':
        corpus(config)  # validate role/hash/disjointness BEFORE allocating a model
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'request.json', {'created_utc': now(), 'binding': binding, 'stage': args.command})
    model, tokenizer = native.load_host(config)
    if args.command != 'qualify':
        native.install(model, config, args.arm)
        native.load_delta(model, args.delta, binding)
    return config, binding, out, model, tokenizer


def corpus(config):
    return load_corpus(config.get('corpus_manifest'))


def tokens(config, tokenizer, row, training):
    ids, labels = encode(tokenizer, row, training)
    cap = config['limits']['max_train_tokens' if training else 'max_prompt_tokens']
    if ids.shape[1] > cap:
        raise Blocked(f'row {row["id"]} has {ids.shape[1]} tokens, exceeds {cap}; truncation forbidden')
    return ids.to(config['device']), labels.to(config['device'])


def recovery_loss(model, ids, labels):
    """Pinned causal mean CE, projecting only predictors of supervised labels.

    The complete teacher-forced prefix still traverses the same backbone. Qwen3
    4.51.3 selects hidden-state rows before its vocabulary head; ignored prompt
    labels and the unused final input position need no vocabulary-sized logits.
    This matches its default ForCausalLMLoss (no num_items_in_batch override),
    including the last prompt token predicting the first answer and answer EOS.
    """
    if ids.ndim != 2 or labels.shape != ids.shape or ids.shape[0] != 1:
        raise Blocked('recovery requires matching batch-one input IDs and labels')
    shifted = labels[:, 1:]
    positions = torch.nonzero(shifted[0] != -100, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise Blocked('recovery requires at least one supervised next-token label')
    targets = shifted.index_select(1, positions)
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits
    if logits.ndim != 3 or logits.shape[:2] != (1, positions.numel()):
        raise Blocked('model did not project exactly the supervised recovery positions')
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1).to(logits.device),
        ignore_index=-100, reduction='mean')


def capture(model, tokenizer, config, binding, out):
    data, data_hash = corpus(config)
    captured, entries, written_bytes = {}, [], 0
    consumed = list({r['id']: r for r in ordered(data['train'], config['seed'], config['limits']['steps'])}.values())
    def hook(_module, _args, kwargs, output):
        cos, sin = kwargs['position_embeddings']
        captured.update(hidden=kwargs['hidden_states'].detach().cpu().clone(),
                        cos=cos.detach().cpu().clone(), sin=sin.detach().cpu().clone(),
                        target=output[0].detach().cpu().clone())
    module = model.model.layers[config['layer']].self_attn
    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            for i, row in enumerate(consumed):
                ids, _ = tokens(config, tokenizer, row, True)
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
                serialized = save({n: t.contiguous() for n, t in captured.items()})
                size = len(serialized)  # Count the actual file, including its header.
                if written_bytes + size > 512*2**20 or shutil.disk_usage(out).free < size + 2**30:
                    raise Blocked('teacher cache exceeds 512 MiB cap or 1 GiB disk reserve')
                written_bytes += size
                path = out/f'target-{i:06d}.safetensors'
                with path.open('xb') as stream:
                    stream.write(serialized)
                entries.append({'id': row['id'], 'row_sha256': digest(row), 'path': path.name, 'sha256': sha(path)})
                captured.clear()
                del serialized
                resources(config['device'], runtime=True)
    finally:
        handle.remove()
    receipt = {'status': 'TEACHER_TARGETS_CAPTURED', 'binding': binding, 'corpus_sha256': data_hash,
               'layer': config['layer'], 'entries': entries, 'split': 'train', 'created_utc': now(),
               'teacher_cache_file_bytes': written_bytes,
               'allocator_cache_release_events': list(CACHE_RELEASE_EVENTS)}
    write_json(out/'teacher.json', receipt)
    return receipt


def training(model, tokenizer, config, binding, out, stage, teacher=None, parent_delta=None):
    data, data_hash = corpus(config)
    rows = ordered(data['train'], config['seed'], config['limits']['steps'])
    target_index = {}
    if stage == 'distill':
        if not teacher:
            raise Blocked('--teacher directory is required')
        manifest = read_json(Path(teacher)/'teacher.json')
        if manifest['corpus_sha256'] != data_hash or manifest['split'] != 'train':
            raise Blocked('teacher targets belong to another corpus/split')
        for key in ('source_sha256', 'code_sha256', 'config_sha256'):
            if manifest['binding'][key] != binding[key]:
                raise Blocked(f'teacher target {key} mismatch')
        if manifest['binding']['arm'] != 'A' or manifest['binding']['delta_sha256'] is not None:
            raise Blocked('teacher must be unadapted arm A')
        target_index = {item['id']: item for item in manifest['entries']}
        if len(target_index) != len(manifest['entries']) or set(target_index) != {r['id'] for r in rows}:
            raise Blocked('teacher inventory does not exactly cover consumed TRAIN rows')
    parameters = native.trainables(model)
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=config['learning_rate'],
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0, foreach=False)
    module = model.model.layers[config['layer']].self_attn
    records = []
    model.eval()  # deterministic, no dropout; autograd remains enabled
    for step, row in enumerate(rows):
        resources(config['device'], runtime=True)
        optimizer.zero_grad(set_to_none=True)
        if stage == 'distill':
            item = target_index[row['id']]
            path = safe_file(teacher, item['path'])
            if item['row_sha256'] != digest(row) or sha(path) != item['sha256']:
                raise Blocked('teacher row/activation hash changed')
            batch = {k: v.to(config['device']) for k, v in load_file(str(path)).items()}
            length = batch['hidden'].shape[1]
            if length > config['limits']['max_train_tokens']:
                raise Blocked('cached target exceeds registered training shape')
            prediction = module(hidden_states=batch['hidden'], position_embeddings=(batch['cos'], batch['sin']),
                                attention_mask=None, past_key_value=None)[0]
            target = batch['target'].float()
            loss = (prediction.float()-target).square().mean() / target.square().mean().clamp_min(1e-8)
            supervised = length  # positions, not vocabulary labels
        else:
            ids, labels = tokens(config, tokenizer, row, True)
            length = ids.shape[1]
            loss = recovery_loss(model, ids, labels)
            supervised = int((labels[:, 1:] != -100).sum())
        if not torch.isfinite(loss):
            raise Blocked('nonfinite loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters.values()):
            raise Blocked('nonfinite gradients')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise Blocked('frozen parameter received a gradient')
        gradient_norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters.values()):
            raise Blocked('nonfinite adaptation parameter')
        records.append({'step': step, 'row_id': row['id'], 'loss': loss.item(), 'input_tokens': length,
                        'supervised_positions': supervised, 'gradient_norm': float(gradient_norm)})
        resources(config['device'], runtime=True)
    receipt = {'status': 'TRAINING_COMPLETED_NOT_EFFICACY', 'stage': stage, 'created_utc': now(),
               'binding': binding, 'corpus_sha256': data_hash, 'records': records,
               'optimizer': 'AdamW(.9,.95), wd=0, eps=1e-8, clip=1', 'resume_supported': False,
               'parent_record': read_json(Path(parent_delta)/'delta.json')['record'] if parent_delta else None,
               'parameter_names': sorted(parameters), 'trainable_parameters': sum(p.numel() for p in parameters.values()),
               'allocator_cache_release_events': list(CACHE_RELEASE_EVENTS)}
    native.save_delta(model, out, binding, receipt)
    return receipt


def _sync():
    torch.cuda.synchronize()


@torch.no_grad()
def generate(model, ids, tokenizer, config, arm):
    """Instrumented loop timing, including guards; tokenization/loading excluded."""
    length = ids.shape[1]
    reset_at = config.get('reset_recurrent_at')
    if reset_at is not None and (arm != 'D' or not 0 < reset_at < length):
        raise Blocked('state-reset diagnostic requires arm D and an interior prompt boundary')
    cache, output, offset = None, None, 0
    torch.cuda.reset_peak_memory_stats()
    _sync()
    start = time.perf_counter()
    while offset < length:
        end = min(length, offset + config['limits']['prefill_chunk'])
        if reset_at is not None and offset < reset_at < end:
            end = reset_at
        output = native.model_forward(model, ids[:, offset:end], cache)
        cache = output.past_key_values
        if not torch.isfinite(output.logits).all():
            raise Blocked('nonfinite prefill logits')
        resources(config['device'], runtime=True)
        offset = end
        if offset == reset_at:
            cache._sbhm[config['layer']].reset_recurrent()
    first = int(output.logits[0, -1].argmax())
    _sync()
    first_time = time.perf_counter()
    generated, stop = [first], 'max_new_tokens'
    rss_max = resources(config['device'], runtime=True)['rss_bytes']
    for _ in range(1, config['limits']['max_new_tokens']):
        if generated[-1] == tokenizer.eos_token_id:
            stop = 'eos'
            break
        next_ids = ids.new_tensor([[generated[-1]]])
        output = native.model_forward(model, next_ids, cache)
        cache = output.past_key_values
        if not torch.isfinite(output.logits).all():
            raise Blocked('nonfinite decoding logits')
        generated.append(int(output.logits[0, -1].argmax()))
        rss_max = max(rss_max, resources(config['device'], runtime=True)['rss_bytes'])
    if generated[-1] == tokenizer.eos_token_id:
        stop = 'eos'
    _sync()
    end_time = time.perf_counter()
    native.check_cache(cache, config, arm, length + len(generated)-1)
    elapsed = end_time - first_time
    return {'prediction': tokenizer.decode(generated, skip_special_tokens=True), 'generated_ids': generated,
            'stop_reason': stop, 'prompt_tokens': length, 'generated_tokens': len(generated),
            'model_ttft_seconds': first_time-start, 'decode_seconds_excluding_first': elapsed,
            'decode_tokens_per_second': (len(generated)-1)/elapsed if len(generated)>1 and elapsed>0 else None,
            'cache_storage_bytes': cache_bytes(cache), 'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(), 'rss_sample_max_bytes': rss_max,
            'reset_recurrent_at': reset_at, 'backend': 'native_sdpa' if arm in ('A','B') else 'mixed_sdpa_and_sequential_reference'}


def evaluate(model, tokenizer, config, binding, out, arm, delta=None):
    data, data_hash = corpus(config)
    manifest = read_json(config['corpus_manifest'])
    if sha(config['corpus_manifest']) != data_hash:
        raise Blocked('corpus manifest changed while reading evaluation groups')
    groups = validate_evaluation_groups(manifest.get('evaluation_groups'), data['dev'])
    # Fixed synthetic warmup; no evaluation examples used to tune warmup.
    with torch.no_grad():
        model(input_ids=torch.full((1, 8), 10, device=config['device'], dtype=torch.long),
              use_cache=False, logits_to_keep=1)
    predictions = []
    for row in data['dev']:
        ids, _ = tokens(config, tokenizer, row, False)
        result = generate(model, ids, tokenizer, config, arm)
        result.update(id=row['id'], family_id=row['family_id'], target=row['target'],
                      strict_exact_match=result['prediction'] == row['target'],
                      normalized_exact_match=result['prediction'].strip().casefold() == row['target'].strip().casefold())
        predictions.append(result)
        resources(config['device'], runtime=True)
    return {'status': 'DEV_MEASUREMENT_ONLY', 'binding': binding, 'created_utc': now(),
            'corpus_sha256': data_hash, 'predictions': predictions,
            'evaluation_groups': groups,
            'adaptation_record': read_json(Path(delta)/'delta.json')['record'] if delta else None,
            'strict_exact_match': sum(x['strict_exact_match'] for x in predictions)/len(predictions),
            'normalized_exact_match': sum(x['normalized_exact_match'] for x in predictions)/len(predictions),
            'semantic_accuracy': None, 'speedup_claim': False, 'energy_measured': False,
            'allocator_cache_release_events': list(CACHE_RELEASE_EVENTS)}


def compare(paths):
    """Descriptive D/C contrast, with original-model retention kept separate."""
    metrics = ('strict_exact_match', 'normalized_exact_match')
    loaded = [read_json(p) for p in paths]
    if any(r.get('binding', {}).get('arm') not in set('ABCD') for r in loaded):
        raise Blocked('completed DEV reports require a valid arm binding')
    reports = {r['binding']['arm']: r for r in loaded}
    if len(paths) != 4 or set(reports) != set('ABCD'):
        raise Blocked('exactly one completed DEV report per arm A/B/C/D is required')
    base = reports['A']
    if base['binding']['delta_sha256'] is not None or base.get('adaptation_record') is not None:
        raise Blocked('baseline A must remain unadapted')

    def validate_rows(report):
        rows = report.get('predictions')
        if not isinstance(rows, list) or not rows:
            raise Blocked('completed DEV report must contain nonempty predictions')
        for row in rows:
            if not isinstance(row, dict):
                raise Blocked('prediction rows must be objects')
            if any(not isinstance(row.get(key), str) or not row[key] for key in ('id', 'family_id', 'target')):
                raise Blocked('prediction rows require nonempty id/family_id/target strings')
            if any(type(row.get(metric)) is not bool for metric in metrics):
                raise Blocked('prediction exact-match metrics must be booleans')
            if row['strict_exact_match'] and not row['normalized_exact_match']:
                raise Blocked('strict correctness cannot contradict normalized correctness')
            prediction = row.get('prediction')
            if not isinstance(prediction, str):
                raise Blocked('raw prediction text must be a string')
            expected = (prediction == row['target'],
                        prediction.strip().casefold() == row['target'].strip().casefold())
            if any(row[metric] != value for metric, value in zip(metrics, expected)):
                raise Blocked('prediction exact-match metrics contradict prediction/target text')
            if row.get('reset_recurrent_at') is not None:
                raise Blocked('state-reset diagnostic cannot be pooled with main comparison')
        if len({row['id'] for row in rows}) != len(rows):
            raise Blocked('duplicate prediction row IDs')
        for metric in metrics:
            reported = report.get(metric)
            actual = sum(row[metric] for row in rows) / len(rows)
            if (type(reported) not in (int, float) or not math.isfinite(reported) or
                    not math.isclose(reported, actual, rel_tol=0, abs_tol=1e-12)):
                raise Blocked(f'reported {metric} does not match prediction rows')
        return rows

    base_rows = validate_rows(base)
    families = sorted({row['family_id'] for row in base_rows})
    groups = validate_evaluation_groups(base.get('evaluation_groups'), base_rows)

    def retention(baseline, candidate, metric):
        baseline_correct = sum(row[metric] for row in baseline)
        retained = sum(b[metric] and r[metric] for b, r in zip(baseline, candidate))
        return {'baseline_correct': baseline_correct, 'retained': retained,
                'damages': baseline_correct - retained,
                'repairs': sum(not b[metric] and r[metric] for b, r in zip(baseline, candidate)),
                'retention_rate': retained / baseline_correct if baseline_correct else None}

    def arm_summary(baseline, candidate):
        return {'rows': len(candidate),
                'repairs': sum(not b['strict_exact_match'] and r['strict_exact_match'] for b, r in zip(baseline, candidate)),
                'damages': sum(b['strict_exact_match'] and not r['strict_exact_match'] for b, r in zip(baseline, candidate)),
                **{metric: sum(row[metric] for row in candidate) / len(candidate) for metric in metrics},
                'original_model_retention': {metric: retention(baseline, candidate, metric) for metric in metrics}}

    def family_rows(rows, family):
        return [row for row in rows if row['family_id'] == family]

    exposure = None
    result = {}
    for arm, report in reports.items():
        if report['status'] != 'DEV_MEASUREMENT_ONLY' or report['corpus_sha256'] != base['corpus_sha256']:
            raise Blocked('incompatible measurement/corpus')
        for key in ('source_sha256', 'code_sha256', 'config_sha256', 'environment'):
            if report['binding'][key] != base['binding'][key]:
                raise Blocked(f'unpaired {key}')
        if arm != 'A':
            if not isinstance(report['binding'].get('delta_sha256'), str) or not report['binding']['delta_sha256']:
                raise Blocked('trained controls require a bound adaptation delta')
            recovery = report.get('adaptation_record')
            if not recovery or recovery.get('stage') != 'recover':
                raise Blocked('all trained controls require completed recovery receipts')
            distill = recovery.get('parent_record')
            if not distill or distill.get('stage') != 'distill':
                raise Blocked('all trained controls require the registered distillation stage')
            for stage in (distill, recovery):
                if stage.get('status') != 'TRAINING_COMPLETED_NOT_EFFICACY' or not stage.get('records'):
                    raise Blocked('adaptation stages require nonempty completed training receipts')
                if stage.get('corpus_sha256') != report['corpus_sha256']:
                    raise Blocked('adaptation stage has incompatible corpus')
                for key in ('arm', 'source_sha256', 'code_sha256', 'config_sha256', 'environment'):
                    if stage.get('binding', {}).get(key) != report['binding'][key]:
                        raise Blocked(f'adaptation stage has incompatible {key}')
                for index, record in enumerate(stage['records']):
                    if (record.get('step') != index or not isinstance(record.get('row_id'), str) or
                            not record['row_id'] or any(type(record.get(key)) is not int or record[key] < 1
                                                       for key in ('input_tokens', 'supervised_positions')) or
                            record['supervised_positions'] > record['input_tokens'] or
                            any(type(record.get(key)) not in (int, float) or not math.isfinite(record[key])
                                for key in ('loss', 'gradient_norm'))):
                        raise Blocked('invalid or nonfinite training exposure record')
            trace = [[(x['row_id'], x['input_tokens'], x['supervised_positions']) for x in stage['records']]
                     for stage in (distill, recovery)]
            if exposure is not None and trace != exposure:
                raise Blocked('training exposure/order/steps differ across controls')
            exposure = trace
        rows = validate_rows(report)
        if [(r['id'], r['family_id'], r['target']) for r in rows] != [(r['id'], r['family_id'], r['target']) for r in base_rows]:
            raise Blocked('row/order/family/target mismatch')
        if validate_evaluation_groups(report.get('evaluation_groups'), rows) != groups:
            raise Blocked('unpaired evaluation_groups')
        result[arm] = arm_summary(base_rows, rows)
        result[arm]['per_family'] = {family: arm_summary(family_rows(base_rows, family), family_rows(rows, family))
                                     for family in families}

    def contrast(local, hybrid):
        summary = {'paired_rows': len(local)}
        for metric in metrics:
            d_only = sum(not c[metric] and d[metric] for c, d in zip(local, hybrid))
            c_only = sum(c[metric] and not d[metric] for c, d in zip(local, hybrid))
            summary[metric] = {'D_only_correct': d_only, 'C_only_correct': c_only,
                               'both_correct': sum(c[metric] and d[metric] for c, d in zip(local, hybrid)),
                               'neither_correct': sum(not c[metric] and not d[metric] for c, d in zip(local, hybrid)),
                               'D_accuracy': sum(row[metric] for row in hybrid) / len(hybrid),
                               'C_accuracy': sum(row[metric] for row in local) / len(local),
                               'delta_percentage_points': 100 * (d_only - c_only) / len(local)}
        return summary

    local_rows, hybrid_rows = reports['C']['predictions'], reports['D']['predictions']
    hybrid_vs_local = contrast(local_rows, hybrid_rows)
    hybrid_vs_local['per_family'] = {family: contrast(family_rows(local_rows, family), family_rows(hybrid_rows, family))
                                      for family in families}
    grouped = None
    if groups is not None:
        grouped = {}
        for name, group_families in groups.items():
            selected = {arm: [row for row in report['predictions'] if row['family_id'] in group_families]
                        for arm, report in reports.items()}
            grouped[name] = {'family_ids': group_families,
                             'arms': {arm: arm_summary(selected['A'], rows) for arm, rows in selected.items()},
                             'hybrid_vs_local': contrast(selected['C'], selected['D'])}
    return {'status': 'DESCRIPTIVE_PAIRED_DEV_ONLY', 'arms': result, 'promotion': False,
            'hybrid_vs_local': hybrid_vs_local,
            'evaluation_groups': grouped,
            'warning': 'Exposure is paired; trainable counts and runtime costs differ and must be reported.'}


def execute(args):
    config, binding, out, model, tokenizer = prepare(args)
    if args.command == 'qualify':
        receipt = native.qualify(model, config, args.arm, binding, args.delta)
        receipt['allocator_cache_release_events'] = list(CACHE_RELEASE_EVENTS)
    elif args.command == 'teacher':
        receipt = capture(model, tokenizer, config, binding, out)
    elif args.command in ('distill', 'recover'):
        receipt = training(model, tokenizer, config, binding, out, args.command, args.teacher, args.delta)
    else:
        receipt = evaluate(model, tokenizer, config, binding, out, args.arm, args.delta)
    write_json(out/'receipt.json', receipt)
    return receipt
