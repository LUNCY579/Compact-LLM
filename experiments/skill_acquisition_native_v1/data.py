"""Bounded atomic TRAIN selection for prospective donors; no model imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import MANIFEST, AdmissionRejected, digest, read_json, require

MAX_LINE = 131072
TRAIN_FILE = 'canonical/train.jsonl'
POLICY_FILE = 'policy/train.jsonl'
SKILLS = frozenset(('R', 'S', 'P'))


def _json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate dataset JSON key')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(AdmissionRejected('nonfinite dataset JSON')))


def _entry(manifest, relative):
    value = manifest.get('files', {}).get(relative)
    require(type(value) is dict and type(value.get('bytes')) is int
            and type(value.get('sha256')) is str, 'manifest entry missing: ' + relative)
    return value


def _bound_bytes(root, manifest, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'dataset path escaped root')
    entry = _entry(manifest, relative)
    require(entry['bytes'] <= 32 * 1024 * 1024, 'atomic manifest byte limit')
    with path.open('rb') as stream:
        raw = stream.read(entry['bytes'] + 1)
    require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'],
            'dataset manifest entry drift: ' + relative)
    value = _json(raw)
    return value, entry


def load_atomic_train(data, skills, *, expected_count=3000, expected_manifest=MANIFEST):
    """Verify the full source stream; retain only exact requested atomic identities.

    The expected-count/manifest overrides support isolated pure fixtures. A
    production caller must supply registered constants from its admitted packet.
    This loader does not authorize a native run.
    """
    root = Path(data).resolve(strict=True)
    skills = tuple(skills)
    require(skills and len(skills) == len(set(skills)) and set(skills) <= SKILLS,
            'atomic R/S/P skills only')
    require(type(expected_count) is int and expected_count > 0, 'positive stratum count')
    manifest, actual_sha = read_json(root / 'manifest.json')
    require(actual_sha == expected_manifest, 'dataset manifest drift')
    plans, wanted, role_by_id, records = {}, set(), {}, {}
    for skill in skills:
        relative = 'manifests/atomic_' + skill + '.json'
        value, entry = _bound_bytes(root, manifest, relative)
        ids = value.get('ids')
        require(value.get('name') == 'atomic_' + skill and type(ids) is list
                and value.get('rows') == len(ids) == expected_count
                and all(type(i) is str and i for i in ids) and len(set(ids)) == len(ids),
                'atomic manifest identities/count')
        require(not (wanted & set(ids)), 'row reused across atomic strata')
        plans[skill] = value
        records[relative] = dict(entry)
        wanted.update(ids)
        role_by_id.update((i, skill) for i in ids)
    train_entry = _entry(manifest, TRAIN_FILE)
    h, size, found = hashlib.sha256(), 0, {}
    train_path = (root / TRAIN_FILE).resolve()
    require(train_path.is_relative_to(root), 'TRAIN path escaped dataset root')
    with train_path.open('rb') as stream:
        while True:
            line = stream.readline(MAX_LINE + 1)
            if not line:
                break
            require(len(line) <= MAX_LINE and line.endswith(b'\n'), 'TRAIN line limit/termination')
            h.update(line)
            size += len(line)
            require(size <= train_entry['bytes'], 'TRAIN source grew')
            value = _json(line)
            row_id = value.get('id')
            if row_id not in wanted:
                continue
            require(row_id not in found, 'duplicate selected TRAIN identity')
            skill = role_by_id[row_id]
            require(value.get('role') == 'train' and value.get('tag') == 'core'
                    and value.get('kind') == 'composition' and value.get('skills') == [skill]
                    and value.get('data_source') == 'rineya_maxrl_comp_v2',
                    'non-atomic or evaluation/bridge row in training plan')
            require(value.get('prompt_hash') == digest(value.get('prompt')), 'TRAIN prompt hash drift')
            found[row_id] = value
    require(size == train_entry['bytes'] and h.hexdigest() == train_entry['sha256'], 'TRAIN payload drift')
    require(set(found) == wanted, 'selected TRAIN rows missing')
    policy_entry = _entry(manifest, POLICY_FILE)
    policy_path = (root / POLICY_FILE).resolve()
    require(policy_path.is_relative_to(root), 'policy path escaped dataset root')
    ph, psize, seen = hashlib.sha256(), 0, set()
    with policy_path.open('rb') as stream:
        while True:
            line = stream.readline(MAX_LINE + 1)
            if not line:
                break
            require(len(line) <= MAX_LINE and line.endswith(b'\n'), 'policy line limit/termination')
            ph.update(line)
            psize += len(line)
            require(psize <= policy_entry['bytes'], 'policy source grew')
            value = _json(line)
            row_id = value.get('id')
            if row_id not in wanted:
                continue
            require(row_id not in seen, 'duplicate selected policy identity')
            seen.add(row_id)
            require(set(value) == {'id', 'messages'} and value['messages'] == found[row_id]['prompt'],
                    'policy differs from canonical TRAIN prompt')
    require(psize == policy_entry['bytes'] and ph.hexdigest() == policy_entry['sha256'], 'policy payload drift')
    require(seen == wanted, 'selected policy rows missing')
    return found, plans, {'dataset_root': str(root), 'manifest_sha256': actual_sha,
                          'files': dict(records, **{TRAIN_FILE: train_entry, POLICY_FILE: policy_entry}),
                          'selected_ids_sha256': digest(sorted(wanted)), 'rows': len(found)}


def render_training_prompts(tokenizer, rows, render_prompt):
    """The admitted caller supplies the pinned renderer, never an implicit template."""
    require(callable(render_prompt), 'explicit pinned renderer required')
    result = {}
    for row_id, row in rows.items():
        text, ids = render_prompt(tokenizer, row['prompt'])
        require(type(text) is str and type(ids) is list and ids
                and all(type(i) is int and i >= 0 for i in ids), 'rendered TRAIN prompt')
        result[row_id] = {'id': row_id, 'prompt_ids': ids, 'row': row,
                          'prompt_ids_sha256': digest(ids),
                          'render_sha256': hashlib.sha256(text.encode('utf8')).hexdigest()}
    return result
