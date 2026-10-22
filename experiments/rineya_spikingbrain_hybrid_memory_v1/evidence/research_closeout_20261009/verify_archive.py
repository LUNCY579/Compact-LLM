"""Read-only portable byte-integrity check; no model imports or execution."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def contained(base, relative):
    path = (base / relative.replace('\\', '/')).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError('Missing or escaping published file: ' + relative)
    return path


def verify_files(base, inventory):
    for name, record in inventory.items():
        path = contained(base, name)
        if sha(path) != record['sha256']:
            raise ValueError('SHA256 mismatch: ' + name)
        if 'bytes' in record and path.stat().st_size != record['bytes']:
            raise ValueError('Byte-count mismatch: ' + name)
    return len(inventory)


def main():
    manifest = read(ROOT / 'MANIFEST.json')
    copies = read(ROOT / 'COPY_PROVENANCE.json')
    counts = {'archive_files': verify_files(ROOT, manifest['files']),
              'byte_exact_copies': verify_files(ROOT, copies['files']),
              'package_files': verify_files(PACKAGE, manifest['package_files'])}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts
              and p.name != 'MANIFEST.json'}
    if actual != set(manifest['files']):
        raise ValueError('Unlisted or missing archive file')

    review = read(ROOT / 'quality_pilot/corpus-review.json')
    quality = review['code_inventory']
    for name, expected in quality.items():
        if sha(contained(PACKAGE, name)) != expected:
            raise ValueError('Frozen quality code mismatch: ' + name)
    if digest(quality) != review['code_sha256']:
        raise ValueError('Quality inventory digest mismatch')
    # SBHM historical code_hash used Windows relative-name separators.
    sbhm = {p.relative_to(PACKAGE).as_posix().replace('/', '\\'): sha(p)
            for p in sorted((PACKAGE / 'sbhm').glob('*.py'))}
    expected = read(ROOT / 'sbhm_audit/provenance_receipt_v2.json')['binding']
    if digest(sbhm) != expected['code_sha256']:
        raise ValueError('Historical SBHM code digest mismatch')
    counts['quality_code_files'] = len(quality)
    counts['sbhm_code_files'] = len(sbhm)

    snapshots = PACKAGE / 'evidence/native_20261004/completed_receipts'
    counts['native_snapshot_files'] = verify_files(snapshots,
        read(snapshots / 'manifest.json')['files'])

    quality_corpus = ROOT / 'quality_pilot/corpus-v2'
    corpus = read(quality_corpus / 'manifest.json')
    counts['quality_corpus_files'] = verify_files(quality_corpus,
        {r['path']: r for r in corpus['files'].values()})
    if sha(quality_corpus / 'manifest.json') != review['corpus_sha256']:
        raise ValueError('Quality corpus manifest digest mismatch')
    old_corpus = PACKAGE / 'corpora/distant_evidence_v2'
    old = read(old_corpus / 'manifest.json')
    refs = list(old['splits'].values()) + [old['design'], old['review']]
    counts['sbhm_corpus_files'] = verify_files(old_corpus,
        {r['path']: r for r in refs})
    if sha(old_corpus / 'manifest.json') != expected['corpus_sha256']:
        raise ValueError('Historical corpus manifest digest mismatch')
    print(json.dumps({'status': 'PASS', 'counts': counts,
        'scope': 'Published bytes and frozen inventories only',
        'model_imported': False, 'native_execution': False}, sort_keys=True))


if __name__ == '__main__':
    main()
