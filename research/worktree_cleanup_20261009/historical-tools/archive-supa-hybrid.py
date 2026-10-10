"""Archive ignored historical hybrid artifacts without changing repository state."""
from __future__ import annotations
import hashlib
import io
import json
import pathlib
import re
import subprocess
import tarfile
from datetime import datetime, timezone

BASE = pathlib.Path(r'C:\Users\clclc\BusinessProject')
ROOT = BASE / 'SupaExperiment-hybrid-memory-v1'
OUT = BASE / 'Rineya-Worktree-Audit-20261009'
ARCHIVE = OUT / 'supa-hybrid-stage1-ignored-artifacts-20261009.tar.gz'
MANIFEST = OUT / 'supa-hybrid-ignored-scientific-manifest.json'
VERIFY = OUT / 'supa-hybrid-archive-verification.json'
EXPECTED_HEAD = 'b1813444f76f214fed627af4bebe39a9354b721c'

def git(*args: str) -> str:
    result = subprocess.run(['git', '-c', 'core.longpaths=true', '-C', str(ROOT), *args], text=True, capture_output=True, check=True)
    if result.stderr:
        raise RuntimeError('Git reported warnings; archive inventory must be complete: ' + result.stderr)
    return result.stdout

def digest_stream(stream):
    h = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        h.update(block)
    return h.hexdigest()

def file_digest(path):
    with path.open('rb') as stream:
        return digest_stream(stream)

if git('rev-parse', 'HEAD').strip() != EXPECTED_HEAD:
    raise RuntimeError('Hybrid source head changed')
if git('--no-optional-locks', 'status', '--porcelain=v1', '--untracked-files=all').strip():
    raise RuntimeError('Hybrid worktree has new non-ignored changes')
ignored = sorted(git('ls-files', '--others', '--ignored', '--exclude-standard', '-z').rstrip('\0').split('\0'))
excluded_pattern = re.compile(r'/(?:plot-dependencies|test_dependencies|pytest-temp|__pycache__)/|\.py[co]$')
selected = [name for name in ignored if not excluded_pattern.search(name)]
excluded = [name for name in ignored if excluded_pattern.search(name)]
secret_patterns = {
    'github_access_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,})'),
    'huggingface_token': re.compile(rb'hf_[A-Za-z0-9]{30,}'),
    'openai_key': re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{35,}'),
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'aws_access_key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
}
findings = []
records = []
for name in selected:
    path = ROOT / pathlib.PurePosixPath(name)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT.resolve()) or not path.is_file() or path.is_symlink():
        raise RuntimeError('Unexpected source path: ' + name)
    if name.endswith(('.json', '.jsonl', '.py', '.txt', '.xml', '.md', '.svg')):
        data = path.read_bytes()
        for category, pattern in secret_patterns.items():
            if pattern.search(data):
                findings.append({'path': name, 'category': category})
    records.append({'path': name, 'bytes': path.stat().st_size, 'sha256': file_digest(path)})
if findings:
    VERIFY.write_text(json.dumps({'status':'BLOCKED_SECRET_PATTERN', 'findings': findings}, indent=2), encoding='utf-8')
    raise RuntimeError('Secret-like values found; see redacted findings')
manifest = {
    'schema': 'rineya.hybrid-artifact-archive/1',
    'createdUtc': datetime.now(timezone.utc).isoformat(),
    'sourceRoot': str(ROOT),
    'branchHead': EXPECTED_HEAD,
    'sourceTag': 'archive/supa-a1.1-hybrid-memory-architecture-v1-stage1-complete-2026-09-09',
    'sourceRemote': 'https://github.com/LUNCY579/SupaExperiment.git',
    'remotePrivateVerifiedByRoot': True,
    'scientificDecision': 'Completed Stage I; no reliable hybrid advantage over matched stateless control; no native promotion',
    'preservationRequired': True,
    'scientificIgnoredFiles': records,
    'fileCount': len(records),
    'bytes': sum(record['bytes'] for record in records),
    'ignoredFileCount': len(ignored),
    'excludedRuntimeFileCount': len(excluded),
    'excludedRuntimePatterns': 'plot-dependencies/, test_dependencies/, pytest-temp/, __pycache__/, *.pyc, *.pyo',
    'included': 'Original registered/DEV/native sidecar tensors, config/manifests/receipts, training traces, raw original/recovery evaluation, immutable synthetic bank, execution ledgers, systems/analysis outputs, and software qualification transcripts.',
    'baseModelOrTokenizerIncluded': False,
    'credentialPatternMatches': [],
    'privacyAssessment': 'Private research archive; synthetic data and custom sidecar tensors only. Original local paths, user account name and hardware/environment metadata remain in byte-preserved receipts. Third-party runtime dependencies and base checkpoint/tokenizer files are excluded. Existing source tag preserves committed provenance and method. No public rights grant is inferred.',
}
manifest_bytes = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
MANIFEST.write_bytes(manifest_bytes)
readme = (
    'Historical research artifact archive for Supa hybrid-memory Stage I.\n'
    'The committed source is preserved by the existing sourceTag in ARCHIVE_MANIFEST.json.\n'
    'This archive preserves original ignored scientific artifacts that Git tags exclude.\n'
    'It contains synthetic research banks, completed run outputs and custom sidecar tensors.\n'
    'It is an archival negative-result record; no model release or native promotion.\n'
    'Base checkpoint/tokenizer and third-party plotting/testing packages are not included.\n'
    'Original absolute paths and hardware/runtime metadata are retained in immutable receipts.\n'
    'Verify each archived file against SHA256 and length in ARCHIVE_MANIFEST.json.\n'
).encode('utf-8')
with tarfile.open(ARCHIVE, 'w:gz', compresslevel=6) as tar:
    for record in records:
        tar.add(ROOT / pathlib.PurePosixPath(record['path']), arcname=record['path'], recursive=False)
    for name, data in [('ARCHIVE_MANIFEST.json', manifest_bytes), ('ARCHIVE_README.txt', readme)]:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))

expected = {record['path']: record for record in records}
verified = []
with tarfile.open(ARCHIVE, 'r:gz') as tar:
    members = tar.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)) or set(names) != set(expected) | {'ARCHIVE_MANIFEST.json', 'ARCHIVE_README.txt'}:
        raise RuntimeError('Archive membership mismatch')
    if tar.extractfile('ARCHIVE_MANIFEST.json').read() != manifest_bytes:
        raise RuntimeError('Manifest roundtrip mismatch')
    for member in members:
        if not member.isfile() or pathlib.PurePosixPath(member.name).is_absolute() or '..' in pathlib.PurePosixPath(member.name).parts:
            raise RuntimeError('Archive path/type invalid')
        if member.name not in expected:
            continue
        record = expected[member.name]
        actual_hash = digest_stream(tar.extractfile(member))
        if member.size != record['bytes'] or actual_hash != record['sha256']:
            raise RuntimeError('Archived file hash mismatch: ' + member.name)
        if file_digest(ROOT / pathlib.PurePosixPath(member.name)) != record['sha256']:
            raise RuntimeError('Source changed while archiving: ' + member.name)
        verified.append(member.name)
archive_hash = file_digest(ARCHIVE)
(OUT / (ARCHIVE.name + '.sha256')).write_text(archive_hash + '  ' + ARCHIVE.name + '\n', encoding='utf-8')
report = {
    'status': 'VERIFIED_LOCAL_ARCHIVE',
    'archive': str(ARCHIVE), 'archiveBytes': ARCHIVE.stat().st_size, 'archiveSha256': archive_hash,
    'manifest': str(MANIFEST), 'manifestSha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'fileCount': len(records), 'uncompressedScientificBytes': manifest['bytes'],
    'roundtripVerifiedFiles': len(verified), 'archiveMembershipMatchesManifest': True,
    'sourceUnchangedAfterArchive': True, 'sourceHead': EXPECTED_HEAD,
    'credentialPatternMatches': [], 'baseModelOrTokenizerIncluded': False,
    'privacyAssessment': manifest['privacyAssessment'],
    'releaseRecommendation': 'Upload archive and SHA256/manifest to private archival GitHub release on existing archive tag; independently download asset and verify archive SHA256 before worktree removal. This is preservation, not a model/product release.',
}
VERIFY.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
audit_path = OUT / 'supa-audit.json'
audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
audit['hybridArtifactArchive'] = report
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
