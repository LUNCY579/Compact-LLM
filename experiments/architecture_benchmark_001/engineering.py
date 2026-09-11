"""Offline teacher-feature preparation and matched 25-step engineering smokes.

No recovery campaign or DEV/TEST access. Cached final normalized teacher states,
plus the original tied embedding head, reconstruct full-vocabulary KD targets.
No top-K renormalization or online teacher backbone is hidden in the student loop.
"""
from __future__ import annotations
import gc
import shutil
import time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from .budget import presets
from .gates import (GateClosed, context, data_bindings, parent_links, protocol, read_hashed_json,
                    validate_artifacts, validate_receipt, write_once)
from .losses import recovery_loss_from_hidden
from .source import audit_source, canonical_hash, sha256_file
from .transfer import state_digest


def prepare_teacher(source: Path, pilot: Path, calibration: Path, training: Path, g0: Path,
                    out: Path, device: str) -> dict:
    from .run import _directory, _load, _text, ensure_disjoint, load_rows
    audit = audit_source(source)
    bindings = data_bindings(pilot, calibration, training)
    validate_receipt(g0, 'G0', context(audit, pilot, device), data_by_stage=bindings)
    ctx = context(audit, training, device)
    rows = load_rows(training, 'train')
    ensure_disjoint(load_rows(pilot, 'pilot'), load_rows(calibration, 'calibration'), rows)
    _directory(out, source)
    (out/'rows').mkdir()
    teacher, tokenizer, _ = _load(source, device)
    teacher.requires_grad_(False).eval()
    records, artifacts, tokens = [], {}, 0
    started = time.perf_counter()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            ids = _text(tokenizer, row, device)['input_ids']
            hidden = teacher(ids, compute_logits=False, return_hidden=True).hidden
            if hidden.dtype != torch.float32 or not torch.isfinite(hidden).all():
                raise GateClosed('teacher targets must be finite final-normalized FP32 states')
            name = f'rows/{i:06d}.safetensors'
            save_file({'input_ids': ids.cpu().contiguous(), 'hidden': hidden.cpu().contiguous()}, str(out/name))
            artifacts[name] = sha256_file(out/name)
            tokens += ids.numel()
            records.append({'id': row['id'], 'group_id': row['group_id'], 'path': name,
                            'tokens': ids.numel(), 'row_sha256': canonical_hash(row)})
    if sha256_file(training) != ctx['data_sha256'] or audit_source(source)['audit_digest'] != audit['audit_digest']:
        raise GateClosed('source/training data changed during teacher preparation')
    if data_bindings(pilot, calibration, training) != bindings:
        raise GateClosed('input split changed during teacher preparation')
    validate_receipt(g0, 'G0', context(audit, pilot, device), data_by_stage=bindings)
    links = parent_links(out, [g0])
    result = {'schema': 2, 'evidence_kind': 'teacher_targets_preparation_only', 'status': 'PASS',
              'context': ctx, 'parent_receipts': links, 'representation': 'final_normalized_hidden_fp32',
              'teacher_preset': 'B28_1024_official_shape', 'teacher_hidden_size': teacher.config.hidden_size,
              'vocab_size': teacher.config.vocab_size, 'dtype': 'float32', 'artifacts': artifacts, 'rows': records,
              'teacher_forward_tokens': tokens, 'wall_seconds': time.perf_counter()-started,
              'data_bindings': bindings, 'licenses_registered_training': False, 'licenses_efficacy_claim': False,
              'notes': 'Reconstruct full-vocabulary teacher logits with the original tied embedding. '
                       'No truncation/quantization/top-K approximation. Projection rounding can vary with chunk shape.'}
    return write_once(out/'TEACHER_CACHE.json', result)


def validate_teacher_cache(path: Path, expected_context: dict, bindings: dict[str, str]) -> dict:
    body = read_hashed_json(path)
    if (body.get('schema') != 2 or body.get('evidence_kind') != 'teacher_targets_preparation_only'
            or body.get('status') != 'PASS' or body.get('context') != expected_context
            or body.get('representation') != 'final_normalized_hidden_fp32' or body.get('dtype') != 'float32'
            or body.get('teacher_preset') != 'B28_1024_official_shape'
            or body.get('teacher_hidden_size') != presets()['B28_1024_official_shape'].hidden_size
            or body.get('vocab_size') != presets()['B28_1024_official_shape'].vocab_size
            or body.get('licenses_registered_training') is not False or body.get('licenses_efficacy_claim') is not False):
        raise GateClosed('stale, malformed, or nonmatching teacher cache')
    if body.get('data_bindings') != bindings:
        raise GateClosed('teacher cache input split bindings changed')
    validate_artifacts(path, body['artifacts'])
    links = body.get('parent_receipts', [])
    if len(links) != 1 or links[0].get('stage') != 'G0':
        raise GateClosed('teacher cache needs its actual G0 source parent')
    parent_path = (path.parent/links[0]['path']).resolve()
    parent = validate_receipt(parent_path, 'G0', {**expected_context, 'data_sha256': bindings['G0']},
                              data_by_stage=bindings)
    if parent['receipt_sha256'] != links[0].get('receipt_sha256'):
        raise GateClosed('teacher cache parent hash changed')
    records = body.get('rows', [])
    if (not records or len({r['id'] for r in records}) != len(records)
            or len({r['path'] for r in records}) != len(records)
            or {r['path'] for r in records} != set(body['artifacts'])):
        raise GateClosed('teacher cache row/shard inventory mismatch')
    return body


def load_teacher_row(cache_path: Path, manifest: dict, index: int, row: dict, ids: torch.Tensor) -> torch.Tensor:
    record = manifest['rows'][index]
    if record['id'] != row['id'] or record['group_id'] != row['group_id'] or record['row_sha256'] != canonical_hash(row):
        raise GateClosed('teacher cache row/order/group/content mismatch')
    path = cache_path.parent/record['path']
    if sha256_file(path) != manifest['artifacts'][record['path']]:
        raise GateClosed('teacher shard changed')
    tensors = load_file(str(path), device='cpu')
    if set(tensors) != {'input_ids', 'hidden'} or not torch.equal(tensors['input_ids'], ids.cpu()):
        raise GateClosed('teacher targets are not aligned to the current exact token IDs')
    hidden = tensors['hidden']
    if (hidden.shape != (*ids.shape, manifest['teacher_hidden_size']) or hidden.dtype != torch.float32
            or not torch.isfinite(hidden).all() or ids.numel() != record['tokens']):
        raise GateClosed('teacher hidden-state shape/dtype/finite check failed')
    return hidden


def smoke_cached(source: Path, pilot: Path, calibration: Path, training: Path, parent_path: Path,
                 teacher_cache: Path, control_path: Path | None, out: Path, device: str, arm: str) -> dict:
    from .run import (_directory, _load, _load_tokenizer, _record, _restore, _save, _text,
                      ensure_disjoint, load_rows)
    recipe = protocol()['G2']['cached_teacher_engineering']
    if (recipe.get('enabled') is not True or arm not in recipe['arms'] or recipe['steps'] != 25
            or recipe.get('optimizer') != 'SGD_no_momentum'
            or type(recipe.get('gradient_checkpointing')) is not bool):
        raise GateClosed('only the declared exactly-25-step engineering recipe is implemented')
    audit = audit_source(source)
    ctx = context(audit, training, device)
    bindings = data_bindings(pilot, calibration, training)
    stage = {'B28': 'G0', 'D40': 'G1_WIDTH', 'C28': 'G1_CED_C28', 'C40': 'G1_CED_C40'}[arm]
    parent_data = calibration if arm == 'D40' else pilot
    validate_receipt(parent_path, stage, context(audit, parent_data, device), data_by_stage=bindings)
    cache = validate_teacher_cache(teacher_cache, ctx, bindings)
    rows = load_rows(training, 'train')
    ensure_disjoint(load_rows(pilot, 'pilot'), load_rows(calibration, 'calibration'), rows)
    if len(rows) != len(cache['rows']):
        raise GateClosed('teacher cache row coverage differs from training data')
    fingerprint = canonical_hash({'recipe': recipe, 'teacher_cache_sha256': cache['receipt_sha256'],
                                  'teacher_head_device': 'cpu', 'data_sha256': ctx['data_sha256']})
    parents = [parent_path]
    if arm.startswith('C'):
        control_arm = {'C28': 'B28', 'C40': 'D40'}[arm]
        if control_path is None:
            raise GateClosed(f'{arm} needs a matched G2_CACHED_{control_arm} engineering control')
        control = validate_receipt(control_path, f'G2_CACHED_{control_arm}', ctx, data_by_stage=bindings)
        if control['metrics'].get('recipe_sha256') != fingerprint:
            raise GateClosed('CED/control cached targets, optimizer, seed, or data budget differ')
        parents.append(control_path)
    elif control_path is not None:
        raise GateClosed('an ordinary-arm smoke does not take a CED control argument')
    _directory(out, source)
    torch.manual_seed(recipe['seed'])
    if arm == 'B28':
        student, tokenizer, _ = _load(source, device)
    else:
        tokenizer = _load_tokenizer(source)
        student = _restore(parent_path.parent, device)
    if student.config != presets()[protocol()['primary_arms'][arm]]:
        raise GateClosed('wrong initial arm shape')
    initial_state = state_digest(student)
    student.train()
    student.gradient_checkpointing = recipe['gradient_checkpointing']
    # Load only the original tied head, not the full teacher backbone. Keep it off GPU.
    with safe_open(str(source/'model.safetensors'), framework='pt', device='cpu') as f:
        teacher_head = f.get_tensor('model.embed_tokens.weight').float().contiguous()
    teacher_head.requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=recipe['learning_rate'])
    losses, nonzero, tokens, targets = [], set(), 0, 0
    head_work = {}
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(recipe['steps']):
        index = step % len(rows)
        ids = _text(tokenizer, rows[index], device)['input_ids']
        hidden = load_teacher_row(teacher_cache, cache, index, rows[index], ids)
        optimizer.zero_grad(set_to_none=True)
        student_hidden = student(ids, compute_logits=False, return_hidden=True).hidden
        loss = recovery_loss_from_hidden(student_hidden, student.embed_tokens.weight, hidden, teacher_head, ids,
            alpha=recipe['alpha'], temperature=recipe['temperature'], chunk_tokens=recipe['head_chunk_tokens'],
            checkpoint_chunks=True, work_counter=head_work)
        if not torch.isfinite(loss):
            raise GateClosed('nonfinite cached-target engineering loss')
        loss.backward()
        for name, parameter in student.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise GateClosed(f'missing/nonfinite intended gradient: {name}')
            if parameter.grad.count_nonzero():
                nonzero.add(name)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        tokens += ids.numel()
        targets += ids.shape[0]*(ids.shape[1]-1)
    parameter_count = len(list(student.parameters()))
    teacher_frozen = teacher_head.grad is None and not teacher_head.requires_grad
    optimizer.zero_grad(set_to_none=True)
    del optimizer, teacher_head, hidden, student_hidden, loss
    gc.collect()
    student.eval()
    _save(student, out)
    after = state_digest(student)
    with torch.inference_mode():
        expected = student(ids[:, :16]).logits.cpu()
    del student; gc.collect()
    restored = _restore(out, device)
    with torch.inference_mode():
        reload_exact = after == state_digest(restored) and torch.equal(expected, restored(ids[:, :16]).logits.cpu())
    if data_bindings(pilot, calibration, training) != bindings:
        raise GateClosed('input split changed during engineering smoke')
    cache_after = validate_teacher_cache(teacher_cache, ctx, bindings)
    shutil.copyfile(teacher_cache, out/'teacher_cache_manifest.json')
    checks = {'25_steps_completed': len(losses) == 25, 'all_intended_parameters_reached': len(nonzero) == parameter_count,
              'teacher_frozen': teacher_frozen, 'save_reload_exact': reload_exact,
              'source_unchanged': audit_source(source)['audit_digest'] == audit['audit_digest'],
              'training_data_unchanged': sha256_file(training) == ctx['data_sha256'],
              'teacher_cache_unchanged': cache_after['receipt_sha256'] == cache['receipt_sha256'],
              'matched_control_verified': not arm.startswith('C') or len(parents) == 2,
              'optimizer_steps_changed_state': after != initial_state}
    metrics = {'arm': arm, 'engineering_only': True, 'recipe_sha256': fingerprint, 'recipe': recipe,
               'losses': losses, 'student_training_tokens': tokens, 'supervised_shifted_targets': targets,
               'teacher_backbone_tokens_in_student_loop': 0,
               'teacher_preparation_forward_tokens_shared': cache['teacher_forward_tokens'],
               'head_projection_work_including_recomputation': head_work, 'teacher_head_device': 'cpu',
               'teacher_cache_sha256': cache['receipt_sha256'], 'initial_state_sha256': initial_state,
               'final_state_sha256': after, 'wall_seconds': time.perf_counter()-started,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None,
               'teacher_cache_cost_policy': 'Shared preparation recorded once; allocate or amortize it explicitly in later cost comparisons.',
               'licenses_recovery': False, 'not_an_8gb_fit_guarantee': True}
    return _record(out, f'G2_CACHED_{arm}', ctx, checks, metrics,
                   ['config.json', 'model.safetensors', 'teacher_cache_manifest.json'], parents)
