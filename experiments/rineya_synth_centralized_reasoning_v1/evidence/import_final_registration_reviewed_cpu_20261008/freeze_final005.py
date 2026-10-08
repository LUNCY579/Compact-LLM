"""Preserve existing final005 CPU metadata/evidence, without copying runtime."""
from pathlib import Path
from types import ModuleType
import argparse, hashlib, json

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
R = ModuleType('CPU_preservation_helpers'); R.__file__ = str(HERE / 'refresh_final.py')
exec(compile(Path(R.__file__).read_bytes(), R.__file__, 'exec'), R.__dict__)

def preserve(target, independent):
    R.require(target.parent == HERE and target.name == 'frozen_reviewable_import_registration005', 'exact ignored final005 preservation target')
    files = {name: HERE / name for name in ('refresh_final.py', 'test_final_registration.py', 'join_actual_review.py', 'freeze_final005.py', 'AUTHORED_FINAL_RECIPE_CONTROLS005.json', 'ROOT_FINAL_RECIPE_CONTROLS005.json', 'IMPORT_SOURCE_RECIPE005.json', 'ACTUAL_SOURCE_RUNTIME_SELECTION005.json', 'EXACT_IMPORT_RUN_RECIPE005.json', 'IMPORT_OBSERVATION_CONFIG_005.json', 'IMPORT_OBSERVATION_RUNTIME_005.json', 'IMPORT_OBSERVATION_REGISTRATION_005.json', 'OWNER_REVIEW_DRAFT_005.md')}
    for dirname in ('pre_raw_leaf_name_fix001', 'pre_stale_plan_descriptor005', 'frozen_extended_closed_recipe004'):
        for path in (HERE / dirname).rglob('*'):
            if path.is_file(): files['history/' + dirname + '/' + path.relative_to(HERE / dirname).as_posix()] = path
    review = PACKAGE / 'artifacts/import_final_byte_review_work_005'
    for name in ('review_current_bytes.py', 'READER_SUCCESSOR_SELECTION.json', 'FRESH_IMPORT_CURRENT_BYTE_REVIEW005.json'):
        files['current_byte_review/' + name] = review / name
    consumer = PACKAGE / 'artifacts/import_final_review_consumer_work_005_absolute'
    for name in ('ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE.json', 'ACTUAL_BYTE_REVIEW_PROVENANCE.json', 'CPU_PROSPECTIVE_APPROVED_SCHEMA_FIXTURE.json', 'CPU_FULL_QUALIFIER_ACTUAL_WINDOW_REPLAY.json', 'ROOT_FINAL005_REVIEW_PROVENANCE.json'):
        files['actual_review_consumer/' + name] = consumer / name
    predecessor = PACKAGE / 'artifacts/import_final_review_consumer_work_005'
    for path in predecessor.rglob('*'):
        if path.is_file(): files['history/relative_review_consumer005/' + path.relative_to(predecessor).as_posix()] = path
    for path in independent:
        R.require(path.is_file() and path.is_relative_to(PACKAGE / 'artifacts'), 'existing ignored independent review source')
        key = 'independent_review/' + path.name
        R.require(key not in files, 'distinct independent evidence basename')
        files[key] = path
    R.require(all(p.is_file() for p in files.values()), 'all named final005 preservation sources exist')
    target.mkdir(exist_ok=True)
    rows = {}
    for relative, source in files.items():
        raw = source.read_bytes(); destination = target / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): R.require(destination.read_bytes() == raw, 'existing final evidence byte equality; no overwrite')
        else: destination.write_bytes(raw)
        rows[relative] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'original_path': str(source)}
    body = R.seal({'kind': 'REVIEWABLE_CLOSED_IMPORT_REGISTRATION005_PRESERVATION', 'implementation_commit': '8dccdb88870f85f98f6498d432a85c818803387b', 'files': rows, 'independent_review_included': bool(independent), 'native_launch_authorized': False, 'native_context_or_gate_called': False, 'model_training_or_transfer_result_claimed': False})
    reference = R.write(target / 'PRESERVATION.json', body)
    return {'file_count': len(rows), 'total_bytes': sum(r['bytes'] for r in rows.values()), 'reference': reference}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--independent', type=Path, action='append', default=[])
    args = parser.parse_args()
    print(R.canonical(preserve(HERE / 'frozen_reviewable_import_registration005', [p.absolute() for p in args.independent])))
