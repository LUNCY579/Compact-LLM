"""Greedy soup selected on frozen, family-disjoint in-domain dev rows.

    python experiments/rab001_exploration/model_soup.py --run-id NAME [tag ...]

CPU probes only. Average directly into one model in bounded chunks; never hold
a full float64 checkpoint. Verify input hashes and source/selected-ingredient
outputs. Each run preserves its selection plan, predictions and trained weights
in create-only directories. Reports live under evidence/model-soup-runs; large
weight files stay under outputs/model-soup-checkpoints. Seed variants are soup
ingredients, not matched-order controls. Exploratory only: no gate, receipt,
efficacy claim or release.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from fractions import Fraction
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hostbudget
from experiments.architecture_benchmark_001 import retention, run
from experiments.architecture_benchmark_001.gates import implementation_digest, protocol, runtime
from experiments.architecture_benchmark_001.source import ROOT, audit_source, canonical_hash, sha256_file

CKPT = ROOT / 'outputs' / 'arm-checkpoints'
OUT = ROOT / 'evidence' / 'retention-design-v1'
ONPANEL = ROOT / 'outputs' / 'splits_onpanel'
SRC = REPO / 'Supa'
if not (SRC / 'model.safetensors').is_file():
    SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
               r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')

CHUNK_ELEMENTS = 1 << 18
_WS = re.compile(r'\s+')


def normalise(text: str) -> str:
    return _WS.sub(' ', text.strip().casefold()).rstrip('.').strip()


def json_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def write_json(path: Path, body: dict) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def tensor_header(path: Path) -> dict:
    """Validate inventory without importing tensor data into RAM."""
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f'truncated safetensors header: {path}')
        size = struct.unpack('<Q', prefix)[0]
        if not 2 <= size <= 32 * 1024 ** 2:
            raise ValueError(f'unsafe safetensors header length: {path}')
        raw = handle.read(size)
    if len(raw) != size:
        raise ValueError(f'truncated safetensors header: {path}')
    header = json.loads(raw)
    header.pop('__metadata__', None)
    if not header:
        raise ValueError(f'empty checkpoint: {path}')
    intervals = []
    for name, info in header.items():
        shape = info['shape']
        if info['dtype'] != 'F32' or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f'only nonempty fp32 parameters supported: {name}')
        start, end = info['data_offsets']
        if type(start) is not int or type(end) is not int or end - start != math.prod(shape) * 4:
            raise ValueError(f'invalid tensor byte range: {name}')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f'overlap or gap in checkpoint data: {path}')
        cursor = end
    if 8 + size + cursor != path.stat().st_size:
        raise ValueError(f'checkpoint size does not match header: {path}')
    return header


def load_manifest(directory: Path) -> dict | None:
    path = directory / 'MANIFEST.json'
    if not path.is_file() or not (directory / 'trainable.safetensors').is_file():
        return None
    body = json.loads(path.read_text(encoding='utf-8'))
    body['_dir'] = directory
    return body


def arm_spec(arm: str) -> dict:
    match = re.fullmatch(r'B28(?:last(?P<block>\d+))?|(?P<dual>[HP])(?:last(?P<read>\d+))?', arm)
    if not match:
        raise ValueError(f'unknown checkpoint arm: {arm}')
    block = int(match['block']) if match['block'] else None
    read = int(match['read']) if match['read'] else None
    if any(n is not None and not 1 <= n <= 28 for n in (block, read)):
        raise ValueError(f'invalid checkpoint tail: {arm}')
    return {'mode': match['dual'] or 'B28', 'block_tail': block, 'read_tail': read}


def validate_manifest(body: dict, source: dict, digest: str) -> dict:
    directory = body['_dir']
    weights = directory / 'trainable.safetensors'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', body['tag']):
        raise ValueError('ingredient tag must be a simple file-name component')
    if body.get('completed') is not True:
        raise ValueError(f'incomplete ingredient: {directory.name}')
    if body.get('saved_after_tokens', -1) < body.get('token_budget', 0):
        raise ValueError(f'ingredient did not spend its dose: {directory.name}')
    if weights.stat().st_size != body['bytes'] or sha256_file(weights) != body['sha256']:
        raise ValueError(f'checkpoint hash/size mismatch: {directory.name}')
    header = tensor_header(weights)
    shapes = {name: info['shape'] for name, info in header.items()}
    names_hash = hashlib.sha256('\n'.join(sorted(shapes)).encode()).hexdigest()
    if shapes != body['tensor_shapes'] or names_hash != body['soup_key']['tensor_names_sha256']:
        raise ValueError(f'checkpoint inventory mismatch: {directory.name}')
    if len(shapes) != body['tensors'] or sum(math.prod(s) for s in shapes.values()) != body['parameters']:
        raise ValueError(f'checkpoint parameter accounting mismatch: {directory.name}')
    # Historical manifests stored implementation_digest as source_digest.
    # Record that defect explicitly and bind to the verified official source plus
    # original dose artifact. Never relabel the old value as a weight hash.
    legacy = body['soup_key'].get('source_digest') == digest
    known = {source['assets']['model.safetensors'], source['audit_digest']}
    if not legacy and body['soup_key'].get('source_digest') not in known:
        raise ValueError(f'unrecognized source lineage: {directory.name}')
    for field in ('source_weights_sha256', 'source_weight_sha256'):
        value = body.get(field, body.get('reconstruct', {}).get(field))
        if value is not None and value != source['assets']['model.safetensors']:
            raise ValueError(f'source weight binding mismatch: {directory.name}')
    if body.get('source_assets_sha256', source['assets']) != source['assets']:
        raise ValueError(f'source asset binding mismatch: {directory.name}')
    if body.get('implementation_digest', digest) != digest:
        raise ValueError(f'implementation binding mismatch: {directory.name}')
    artifact_path = OUT / f"ARM_DOSE_{body['tag']}.json"
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    for field in ('arm', 'seed', 'lr', 'token_budget', 'beta2'):
        if artifact.get(field) != body[field]:
            raise ValueError(f'ingredient {field} differs from dose: {body["tag"]}')
    if artifact.get('schedule_spec') != body['schedule']:
        raise ValueError(f'ingredient schedule differs from dose: {body["tag"]}')
    data_hashes = artifact.get('data_sha256')
    if data_hashes is not None:
        inputs = {'train': ROOT / 'outputs' / 'splits_large' / 'train.jsonl',
                  'dev': ROOT / 'outputs' / 'splits_large' / 'dev.jsonl',
                  'monitor_probe': ROOT / 'data' / 'monitor_probe.jsonl',
                  'probe_label_audit': OUT / 'PROBE_LABEL_AUDIT.json'}
        if data_hashes != {name: sha256_file(path) for name, path in inputs.items()}:
            raise ValueError(f'ingredient training/evaluation data changed: {body["tag"]}')
        ids = artifact.get('train_row_ids_consumed', [])
        if (len(ids) != artifact['train_rows_consumed']
                or hashlib.sha256('\n'.join(ids).encode()).hexdigest() != artifact['train_row_ids_sha256']
                or not set(ids) <= {row['id'] for row in json_rows(inputs['train'])}):
            raise ValueError(f'ingredient consumed training-row binding mismatch: {body["tag"]}')
    if (artifact.get('completed') is not True
            or artifact.get('device_integrity_control_passed') is not True
            or artifact.get('reference_digest') != digest
            or artifact.get('checkpoint', {}).get('sha256') != body['sha256']):
        raise ValueError(f'ingredient dose/provenance mismatch: {body["tag"]}')
    endpoint = artifact['checkpoints'][-1]
    if (endpoint.get('tokens') != body['saved_after_tokens']
            or endpoint.get('steps') != body['saved_after_steps']
            or artifact.get('supervised_target_tokens') != body['saved_after_tokens']
            or artifact.get('optimizer_steps') != body['saved_after_steps']
            or 'outputs' not in endpoint):
        raise ValueError(f'ingredient lacks outputs at saved weights: {body["tag"]}')
    spec = arm_spec(body['arm'])
    for declared in (body['soup_key'], body['reconstruct']):
        if declared['mode'] not in (spec['mode'], body['arm']):
            raise ValueError(f'architecture mode differs from arm: {body["tag"]}')
        for tail in ('read_tail', 'block_tail'):
            if declared.get(tail, spec[tail]) != spec[tail]:
                raise ValueError(f'{tail} differs from arm: {body["tag"]}')
    key = {**spec, 'source_weights_sha256': source['assets']['model.safetensors'],
           'tensor_names_sha256': names_hash, 'tensor_shapes_sha256': canonical_hash(shapes)}
    body['_header'], body['_endpoint'], body['_key'] = header, endpoint['outputs'], key
    body['_provenance'] = {
        'tag': body['tag'], 'sha256': body['sha256'],
        'manifest_sha256': sha256_file(directory / 'MANIFEST.json'),
        'dose_artifact_sha256': sha256_file(artifact_path),
        'legacy_source_digest_was_implementation_digest': legacy,
        'verified_source_weights_sha256': source['assets']['model.safetensors'],
        'source_output_control_recorded': True, 'endpoint_outputs_available': True,
        'training_data_hashes_available': data_hashes is not None,
        'training_data_sha256': data_hashes,
        'consumed_train_row_ids_sha256': artifact.get('train_row_ids_sha256'),
        'seed': body['seed'], 'lr': body['lr'], 'token_budget': body['token_budget'],
        'schedule': body['schedule'], 'beta2': body['beta2'],
        'role': 'soup ingredient; seed variants have different training orders'}
    return body


def build_arm(mode: str, read_tail, device: str):
    """Legacy API; trainability does not affect reconstruction or probes."""
    if mode.startswith('B28'):
        return run._load(SRC, device)[0]
    from experiments.dual28_ced_v1.model import DualStack
    source = run._load(SRC, device)[0]
    model = DualStack.from_source(source, mode)
    del source
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model


def apply_tensors(model, tensors: dict) -> int:
    """Legacy API for small callers; soup uses apply_average."""
    by_name = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in tensors.items():
            if name not in by_name or tuple(by_name[name].shape) != tuple(value.shape):
                raise ValueError(f'checkpoint tensor missing or wrong shape: {name}')
            if not torch.isfinite(value).all():
                raise ValueError(f'nonfinite checkpoint tensor: {name}')
            by_name[name].copy_(value.to(by_name[name].device, by_name[name].dtype))
    return len(tensors)


def slices(shape: list[int]):
    rows = max(1, CHUNK_ELEMENTS // math.prod(shape[1:]))
    for start in range(0, shape[0], rows):
        yield slice(start, min(shape[0], start + rows))


def expected_trainables(model, key: dict) -> set[str]:
    names = {name for name, _ in model.named_parameters()}
    mode = key['mode']
    if mode == 'B28':
        tail = key.get('block_tail')
        if tail is None:
            return names
        if not 1 <= tail <= len(model.encoder):
            raise ValueError('invalid trainable block tail')
        first = len(model.encoder) - tail
        return {name for name in names if (match := re.match(r'^encoder\.(\d+)\.', name))
                and int(match.group(1)) >= first}
    if mode not in ('H', 'P'):
        raise ValueError(f'unsupported soup architecture: {mode}')
    tail = key.get('read_tail')
    if tail is not None and not 1 <= tail <= len(model.reads):
        raise ValueError('invalid trainable read tail')
    first = 0 if tail is None else len(model.reads) - tail
    return {name for name in names if name.startswith('memory.')
            or ((match := re.match(r'^reads\.(\d+)\.', name)) and int(match.group(1)) >= first)}


def apply_average(model, members: list[dict]) -> None:
    """Uniform average with bounded ordinary-file reads and a small fp64 accumulator."""
    from experiments.rab001_exploration.tensor_stream import MAX_ELEMENTS, TensorFile
    names = members[0]['tensor_shapes']
    if any(m['tensor_shapes'] != names for m in members):
        raise ValueError('cannot average different tensor inventories')
    expected = expected_trainables(model, members[0]['_key'])
    if set(names) != expected:
        raise ValueError(f'checkpoint omits or adds arm trainables: missing '
                         f'{sorted(expected - set(names))[:3]}, extra {sorted(set(names) - expected)[:3]}')
    params = dict(model.named_parameters())
    with ExitStack() as stack, torch.no_grad():
        files = [stack.enter_context(TensorFile(m['_dir'] / 'trainable.safetensors')) for m in members]
        for reader in files:
            if ({name: entry['shape'] for name, entry in reader.header.items()} != names
                    or any(entry['dtype'] != 'F32' for entry in reader.header.values())):
                raise ValueError('opened checkpoint inventory differs from validated manifest')
        for name, shape in names.items():
            if name not in params or list(params[name].shape) != shape or params[name].device.type != 'cpu':
                raise ValueError(f'CPU model does not match ingredient tensor: {name}')
            target = params[name].view(-1)
            chunk_elements = min(CHUNK_ELEMENTS, MAX_ELEMENTS)
            for start in range(0, target.numel(), chunk_elements):
                count = min(chunk_elements, target.numel() - start)
                total = files[0].read(name, start, count).to(torch.float64)
                for reader in files[1:]:
                    total.add_(reader.read(name, start, count))
                total.div_(len(files))
                if not bool(torch.isfinite(total).all()):
                    raise ValueError(f'nonfinite averaged weights: {name}')
                target[start:start + count].copy_(total)
                del total
            hostbudget.guard(1.2, 'model soup apply average')


def save_trainables(model, members: list[dict], destination: Path, source: dict) -> dict:
    """Stream standard safetensors bytes without a second full parameter copy."""
    destination.mkdir(parents=True, exist_ok=False)
    shapes = members[0]['tensor_shapes']
    offset, header = 0, {}
    for name, shape in sorted(shapes.items()):
        end = offset + math.prod(shape) * 4
        header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(',', ':')).encode('utf-8')
    raw += b' ' * ((-len(raw)) % 8)
    weights = destination / 'trainable.safetensors'
    params = dict(model.named_parameters())
    with weights.open('xb') as handle, torch.no_grad():
        handle.write(struct.pack('<Q', len(raw)))
        handle.write(raw)
        for name in header:
            for part in slices(shapes[name]):
                chunk = params[name][part].detach().to(device='cpu', dtype=torch.float32).contiguous()
                handle.write(memoryview(chunk.numpy()).cast('B'))
                del chunk
    if tensor_header(weights) != header:
        raise ValueError('saved soup inventory failed verification')
    result = {'schema': 2, 'evidence_kind': 'exploratory_soup_weights_only',
              'members': [m['_provenance'] for m in members], 'uniform_weight': 1 / len(members),
              'soup_key': members[0]['_key'], 'source_assets': source['assets'],
              'tensor_shapes': shapes, 'tensors': len(shapes), 'parameters': offset // 4,
              'bytes': weights.stat().st_size, 'sha256': sha256_file(weights),
              'dtype': 'float32', 'completed': True, 'licenses_release': False,
              'reconstruct': {'source': str(SRC), 'mode': members[0]['_key']['mode'],
                              'read_tail': members[0]['_key'].get('read_tail'),
                              'block_tail': members[0]['_key'].get('block_tail'),
                              'then': 'copy trainable.safetensors tensors by name'},
              'path': str(destination)}
    write_json(destination / 'MANIFEST.json', result)
    return result


def assert_disjoint(left: list[dict], right: list[dict], label: str) -> None:
    for field in ('id', 'group_id'):
        if {r[field] for r in left} & {r[field] for r in right}:
            raise ValueError(f'{label}: overlapping {field}')


def freeze_dev(count: int, panel: list[dict]) -> tuple[list[dict], dict]:
    if count <= 0:
        raise ValueError('selection row count must be positive')
    path = ONPANEL / 'dev.jsonl'
    all_dev = json_rows(path)
    if len(all_dev) < count:
        raise ValueError(f'need {count} held-out rows, found {len(all_dev)}')
    for field in ('id', 'group_id'):
        if len({r[field] for r in all_dev}) != len(all_dev):
            raise ValueError(f'duplicate dev {field}')
    assert_disjoint(all_dev, panel, 'dev/actual monitor panel')
    checked = {'actual_panel_sha256': sha256_file(ROOT / 'data' / 'monitor_probe.jsonl')}
    for training in (ROOT / 'outputs' / 'splits_large' / 'train.jsonl',
                     ROOT / 'outputs' / 'splits_large' / 'dev.jsonl', ONPANEL / 'train.jsonl',
                     ROOT / 'data' / 'calibration.jsonl', ROOT / 'data' / 'pilot.jsonl'):
        assert_disjoint(all_dev, json_rows(training), f'dev/{training.parent.name}/{training.name}')
        checked[str(training)] = sha256_file(training)
    dev = all_dev[:count]
    for row in dev:
        if not row.get('messages') or row['messages'][-1]['role'] == 'assistant' or not row.get('answer'):
            raise ValueError(f'unsafe or unscorable dev prompt: {row["id"]}')
    return dev, {'dev_file_sha256': sha256_file(path), 'selected_rows_sha256': canonical_hash(dev),
                 'selected_ids': [r['id'] for r in dev], 'selected_group_ids': [r['group_id'] for r in dev],
                 'checks': checked, 'selection': f'first {count} rows in frozen file order'}


def check_outputs(outputs: list[dict], expected: dict, label: str) -> dict:
    actual = {o['id']: {'text': o['text'], 'terminated': o['terminated']} for o in outputs}
    if len(actual) != len(outputs) or set(actual) != set(expected):
        raise ValueError(f'{label}: output row inventory differs')
    changed = [rid for rid in actual if actual[rid] != expected[rid]]
    if changed:
        raise ValueError(f'{label}: {len(changed)} text/termination mismatches; first {changed[:3]}')
    return {'passed': True, 'rows': len(actual), 'text_and_termination_mismatches': 0}


def guarded_probe(model, tokenizer, rows, eos):
    outputs = []
    for row in rows:
        hostbudget.guard(1.2, 'model soup CPU probe')
        outputs.extend(run._greedy_probe(model, tokenizer, [row], eos, 'cpu'))
    return outputs


def verify_bound_files(bindings: dict[str, str]) -> None:
    for filename, expected in bindings.items():
        if sha256_file(Path(filename)) != expected:
            raise ValueError(f'input changed during soup run: {filename}')


def paired(left: dict, right: dict) -> dict:
    if set(left) != set(right):
        raise ValueError('paired comparison requires identical row IDs')
    b = sum(left[r]['correct'] and not right[r]['correct'] for r in left)
    c = sum(right[r]['correct'] and not left[r]['correct'] for r in left)
    n = b + c
    p = min(Fraction(1), 2 * sum((Fraction(math.comb(n, k), 2 ** n)
                                  for k in range(min(b, c) + 1)), Fraction()))
    return {'soup_only_correct': b, 'single_only_correct': c, 'discordant': n,
            'p_exact': str(p), 'p_float': float(p), 'alpha': 0.05,
            'soup_improvement_demonstrated': b > c and p < Fraction(1, 20),
            'note': 'One final soup/best-dev-single test per preregistered candidate group; '
                    'training seeds need not match for pairing the same evaluation rows.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags', nargs='*')
    parser.add_argument('--eval-rows', type=int, default=60)
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        raise ValueError('run-id must be a simple directory name')
    destination = OUT / 'model-soup-runs' / args.run_id
    checkpoint_root = ROOT / 'outputs' / 'model-soup-checkpoints' / args.run_id
    if checkpoint_root.exists():
        raise ValueError(f'create-only soup checkpoint directory exists: {checkpoint_root}')
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    report = {'schema': 2, 'completed': False, 'evidence_kind': 'exploratory_weight_averaging_only',
              'licenses_release': False, 'licenses_efficacy_claim': False,
              'selection_metric': 'terminated normalised exact match on held-out reference answer',
              'tie_break': 'tag lexical ascending', 'greedy_acceptance': 'strict dev score increase',
              'selection_rows': args.eval_rows, 'probe_device': 'cpu',
              'checkpoint_root': str(checkpoint_root), 'groups': {}}
    try:
        hostbudget.require(4.0, 'model soup startup')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source = audit_source(SRC)
        digest = implementation_digest()
        report.update({'source_audit': source, 'implementation_digest': digest, 'runtime': runtime(),
                       'harness_sha256': sha256_file(Path(__file__)),
                       'initial_hostbudget': hostbudget.snapshot()})
        manifests = [m for m in (load_manifest(d) for d in sorted(CKPT.iterdir()) if d.is_dir()) if m]
        if args.tags:
            if len(set(args.tags)) != len(args.tags):
                raise ValueError('duplicate requested ingredient tag')
            found = {m['tag'] for m in manifests}
            if set(args.tags) - found:
                raise ValueError(f'missing requested checkpoints: {sorted(set(args.tags) - found)}')
            manifests = [m for m in manifests if m['tag'] in args.tags]
        if not manifests:
            raise ValueError('no usable checkpoints')
        groups = {}
        for item in manifests:
            m = validate_manifest(item, source, digest)
            groups.setdefault(canonical_hash(m['_key'])[:16], []).append(m)
        rows = run.load_probe_rows(ROOT / 'data' / 'monitor_probe.jsonl')
        dev, frozen_dev = freeze_dev(args.eval_rows, rows)
        report['frozen_selection'] = frozen_dev
        report['candidate_provenance'] = [m['_provenance'] for m in manifests]
        write_json(destination / 'SELECTION_PLAN.json', {
            'selection': frozen_dev, 'candidates': report['candidate_provenance'],
            'selection_metric': report['selection_metric'], 'tie_break': report['tie_break'],
            'greedy_acceptance': report['greedy_acceptance']})
        with (destination / 'selection_rows.jsonl').open('x', encoding='utf-8') as handle:
            for row in dev:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        policy = protocol()['G1_RETENTION']
        audit_path, strata_path = OUT / 'PROBE_LABEL_AUDIT.json', OUT / 'HEADROOM_AUDIT.json'
        overlay = retention.build_overlay(json.loads(audit_path.read_text(encoding='utf-8')), policy['bare_policy'])
        receipt_path = ROOT / 'outputs' / f'retention-{digest[:8]}' / 'B28' / 'retention.json'
        frozen = json.loads(receipt_path.read_text(encoding='utf-8'))
        references = set(frozen['reference_ids_by_policy'][policy['bare_policy']])
        expected_source = {r['id']: {'text': r['source']['raw'], 'terminated': r['source']['terminated']}
                           for r in frozen['rows']}
        strata = json.loads(strata_path.read_text(encoding='utf-8'))
        of_stratum = {rid: name for name, ids in strata['policies'][policy['bare_policy']]['ids'].items()
                      for rid in ids}
        if set(expected_source) != {r['id'] for r in rows} or set(of_stratum) & references:
            raise ValueError('panel/reference/strata inventories disagree')
        if set(of_stratum) | references != set(expected_source):
            raise ValueError('strata and references do not partition the panel')
        source_scored = {rid: retention.score_prediction(o['text'], o['terminated'], overlay['targets'][rid])
                         for rid, o in expected_source.items()}
        if {rid for rid, s in source_scored.items() if s['correct']} != references:
            raise ValueError('source rescoring does not reproduce reference set')
        report['evaluation_inputs'] = {str(p): sha256_file(p) for p in (audit_path, strata_path, receipt_path)}
        # Include external architecture/guard code as well as the hashed official
        # implementation: neither exploratory helpers nor DualStack enter that digest.
        binding_paths = {Path(__file__), Path(hostbudget.__file__),
                         Path(__file__).with_name('tensor_stream.py'),
                         REPO / 'experiments' / 'dual28_ced_v1' / 'model.py',
                         OUT / 'DISTINGUISHABILITY_PREREG.md',
                         ONPANEL / 'dev.jsonl', destination / 'selection_rows.jsonl'}
        binding_paths.update(SRC / name for name in source['assets'])
        binding_paths.update(Path(name) for name in frozen_dev['checks'] if name != 'actual_panel_sha256')
        binding_paths.add(ROOT / 'data' / 'monitor_probe.jsonl')
        binding_paths.update((audit_path, strata_path, receipt_path))
        for m in manifests:
            binding_paths.update((m['_dir'] / 'MANIFEST.json', m['_dir'] / 'trainable.safetensors',
                                  OUT / f"ARM_DOSE_{m['tag']}.json"))
        # Reuse the hashes taken when each input was consumed where available.
        bindings = {}
        bindings[str(Path(__file__))] = report['harness_sha256']
        bindings[str(ONPANEL / 'dev.jsonl')] = frozen_dev['dev_file_sha256']
        bindings[str(ROOT / 'data' / 'monitor_probe.jsonl')] = frozen_dev['checks']['actual_panel_sha256']
        bindings.update({name: value for name, value in frozen_dev['checks'].items()
                         if name != 'actual_panel_sha256'})
        bindings.update(report['evaluation_inputs'])
        bindings.update({str(SRC / name): value for name, value in source['assets'].items()})
        for m in manifests:
            bindings[str(m['_dir'] / 'trainable.safetensors')] = m['sha256']
            bindings[str(m['_dir'] / 'MANIFEST.json')] = m['_provenance']['manifest_sha256']
            bindings[str(OUT / f"ARM_DOSE_{m['tag']}.json")] = m['_provenance']['dose_artifact_sha256']
        for path in sorted(binding_paths):
            if str(path) not in bindings:
                bindings[str(path)] = sha256_file(path)
        report['bound_file_sha256'] = bindings
        write_json(destination / 'INPUT_BINDINGS.json', {'files': bindings, 'implementation_digest': digest})
        tok = run._load_tokenizer(SRC)
        eos = run._eos_ids(SRC, tok)

        def evaluate(key, ingredients, *, panel=False, control=None, save_to=None):
            mode, tail = key['mode'], key['read_tail']
            hostbudget.require(3.8 if mode == 'B28' else 7.8, f'model soup build {mode}')
            model = build_arm(mode, tail, 'cpu')
            try:
                if ingredients:
                    apply_average(model, ingredients)
                model.eval()
                result = {}
                if control is not None:
                    outputs = guarded_probe(model, tok, rows, eos)
                    result['output_control'] = check_outputs(outputs, control, 'reconstructed checkpoint')
                if ingredients:
                    outputs_dev = guarded_probe(model, tok, dev, eos)
                    answers = {r['id']: r['answer'] for r in dev}
                    per_dev = {o['id']: {**o, 'correct': bool(o['terminated'] and
                               normalise(o['text']) == normalise(answers[o['id']]))} for o in outputs_dev}
                    result.update({'dev_exact': sum(o['correct'] for o in per_dev.values()),
                                   'dev_rows': len(dev), 'dev_per_row': per_dev})
                if panel:
                    if control is None:
                        outputs = guarded_probe(model, tok, rows, eos)
                    scored = run._score_probe(outputs, overlay['targets'])
                    counts = run._retention_classes(rows, source_scored, scored, references)
                    verdict = retention.monitor_verdict(counts, policy, policy['probe_classes'])
                    gained = [rid for rid, s in scored.items() if s['correct'] and rid not in references]
                    result.update({'panel_correct': sum(s['correct'] for s in scored.values()),
                                   'panel_kept': sum(scored[r]['correct'] for r in references),
                                   'panel_lost': sum(not scored[r]['correct'] for r in references),
                                   'panel_gained': len(gained), 'reference_n': len(references),
                                   'gained_wrong_answer': sum(of_stratum[r] == 'wrong_answer' for r in gained),
                                   'gained_format_only': sum(of_stratum[r] == 'format_only' for r in gained),
                                   'gained_runaway': sum(of_stratum[r] == 'runaway' for r in gained),
                                   'panel_per_class': counts, 'retention_verdict': verdict,
                                   'panel_per_row': {o['id']: {**o, **scored[o['id']]} for o in outputs}})
                if save_to is not None:
                    result['checkpoint'] = save_trainables(model, ingredients, save_to, source)
                return result
            finally:
                del model
                gc.collect()

        for group_id, members in groups.items():
            key = members[0]['_key']
            group_dir = destination / group_id
            group_dir.mkdir()
            if len(members) < 2:
                report['groups'][group_id] = {'status': 'SKIPPED_SINGLE_INGREDIENT', 'key': key,
                                               'tags': [m['tag'] for m in members]}
                continue
            control = evaluate(key, [], control=expected_source)
            ranked = []
            for m in members:
                score = evaluate(key, [m])
                write_json(group_dir / f"candidate_{m['tag']}.json", score)
                ranked.append((score['dev_exact'], m))
                print(f"{m['tag']}: dev {score['dev_exact']}/{len(dev)}", flush=True)
            ranked.sort(key=lambda item: (-item[0], item[1]['tag']))
            kept, best, trail = [ranked[0][1]], ranked[0][0], []
            for _, candidate in ranked[1:]:
                score = evaluate(key, kept + [candidate])
                improved = score['dev_exact'] > best
                write_json(group_dir / f"trial_{candidate['tag']}.json", score)
                trail.append({'candidate': candidate['tag'], 'soup_dev': score['dev_exact'],
                              'previous': best, 'accepted': improved})
                if improved:
                    kept, best = kept + [candidate], score['dev_exact']
                print(f"+ {candidate['tag']}: {score['dev_exact']}/{len(dev)} "
                      f"{'ACCEPT' if improved else 'reject'}", flush=True)
            selected = ranked[0][1]
            # Final panel scores cannot influence ingredient selection.
            single = evaluate(key, [selected], panel=True, control=selected['_endpoint'],
                              save_to=checkpoint_root / group_id if len(kept) == 1 else None)
            final = (single if len(kept) == 1 else
                     evaluate(key, kept, panel=True, save_to=checkpoint_root / group_id))
            if single['dev_exact'] != ranked[0][0] or final['dev_exact'] != best:
                raise ValueError('final reconstruction did not reproduce frozen dev selection scores')
            comparison = paired(final['panel_per_row'], single['panel_per_row'])
            result = {'status': 'OK', 'key': key, 'source_control': control,
                      'ranked': [{'tag': m['tag'], 'dev_exact': s} for s, m in ranked],
                      'greedy_trail': trail, 'soup': [m['tag'] for m in kept],
                      'soup_result': final, 'best_single_tag': selected['tag'],
                      'best_single_result': single, 'paired_comparison': comparison}
            report['groups'][group_id] = result
            write_json(group_dir / 'RESULT.json', result)
            print(f"SOUP {final['panel_correct']}/150 kept {final['panel_kept']}/47; "
                  f"best single {single['panel_correct']}/150; p={comparison['p_float']:.6g}", flush=True)
        verify_bound_files(bindings)
        if implementation_digest() != digest:
            raise ValueError('official implementation changed during soup run')
        for result in report['groups'].values():
            if result['status'] == 'OK':
                saved = result['soup_result']['checkpoint']
                verify_bound_files({str(Path(saved['path']) / 'trainable.safetensors'): saved['sha256']})
        report['final_input_and_weight_hash_verification_passed'] = True
        report['completed'] = True
        write_json(destination / 'MODEL_SOUP.json', report)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        write_json(destination / 'FAILED.json', report)
        raise
    print(f'wrote {destination}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
