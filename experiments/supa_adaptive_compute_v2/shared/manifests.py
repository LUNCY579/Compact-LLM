"""Immutable registrations, source commitments and independent run identities."""
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import subprocess
import uuid
from .legacy import ROOT, REPO
from .hashing import canonical_sha256, sha256_file, source_files, write_json
from .guards import guard_non_sealed_path, require_track, require_within


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def git_identity():
    return {'branch': git('branch','--show-current'), 'commit_sha': git('rev-parse','HEAD'),
            'git_status': git('status','--short'), 'python': platform.python_version(),
            'os':platform.platform(),'processor':platform.processor()}


def freeze_registration(track, config_path, manifest_paths):
    require_track(track)
    config_path = require_within(config_path, ROOT / track)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifests = {}
    for path in manifest_paths:
        path = require_within(path, ROOT / 'data')
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['track'] != track:
            raise ValueError('cannot register another track dataset')
        manifests[path.relative_to(REPO).as_posix()] = sha256_file(path)
    if not manifests:
        raise ValueError('registration requires a frozen dataset manifest')
    contract = {'track':track, 'config_sha256':sha256_file(config_path),
        'config_path':config_path.relative_to(REPO).as_posix(), 'config':config,
        'sources':source_files(track), 'dataset_manifests':manifests}
    digest = canonical_sha256(contract)
    path = ROOT / track / 'evidence/00_preregistration' / digest / 'registration.json'
    write_json(path, {**contract, 'registration_sha256':digest, 'status':'FROZEN',
        'registered_at_utc':utc_now(), 'registered_against':git_identity()})
    return path


def _require_committed(path):
    relative = path.relative_to(REPO).as_posix()
    try:
        committed = subprocess.check_output(['git','-C',str(REPO),'show',f'HEAD:{relative}'], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'preregistration/source must be committed before execution: {relative}') from exc
    # Git may normalize CRLF; compare bytes after newline normalization only.
    if committed.replace(b'\r\n',b'\n') != path.read_bytes().replace(b'\r\n',b'\n'):
        raise RuntimeError(f'uncommitted registered file: {relative}')


def verify_registration(track, config_path):
    require_track(track)
    config_path = require_within(config_path, ROOT / track)
    current_sources = source_files(track)
    matches = []
    for path in (ROOT / track / 'evidence/00_preregistration').glob('*/registration.json'):
        registration = json.loads(path.read_text(encoding='utf-8'))
        if registration['config_sha256'] == sha256_file(config_path) and registration['sources'] == current_sources:
            matches.append((path,registration))
    if len(matches) != 1:
        raise RuntimeError('exactly one compatible frozen registration required; freeze source/config/data and commit before execution')
    path, value = matches[0]
    contract = {key:value[key] for key in ('track','config_sha256','config_path','config','sources','dataset_manifests')}
    if value['track'] != track or value['status'] != 'FROZEN' or canonical_sha256(contract) != value['registration_sha256']:
        raise RuntimeError('registration digest mismatch')
    for relative, expected in value['dataset_manifests'].items():
        candidate = require_within(REPO / relative, ROOT / 'data')
        if sha256_file(candidate) != expected:
            raise RuntimeError('registered dataset manifest mismatch')
        _require_committed(candidate)
    for relative in current_sources:
        candidate = REPO / relative
        # The user's preserved graft v1 is untracked; its exact hash is bound.
        if candidate.is_relative_to(ROOT):
            _require_committed(candidate)
    _require_committed(path)
    _require_committed(config_path)
    return {**value, 'registration_path':str(path)}


def new_run(track, stage, details):
    require_track(track)
    if not stage or '/' in stage or '\\' in stage or '..' in stage:
        raise ValueError('invalid evidence stage')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / track / 'evidence' / stage / (stamp + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**git_identity(), **details, 'track':track,'stage':stage,
                'start_utc':utc_now(), 'status':'STARTED','failure_receipt':None}
    write_json(output / 'started_manifest.json', manifest)
    return output, manifest
