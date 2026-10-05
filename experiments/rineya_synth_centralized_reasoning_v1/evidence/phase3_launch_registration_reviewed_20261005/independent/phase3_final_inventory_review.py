"""Final source/JSON inventory review. No launch, Git, runtime or model calls."""
import hashlib, importlib.util, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STAGE = PACKAGE / 'artifacts/phase3_launch_registration_work'
FINAL = PACKAGE / 'artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH04_A1_AUDIT_INVENTORY_001.json'
FINAL_FILE_SHA = 'cc28dd3d02a8f550699892428022e0ebf61c2cdcbf2dd9286f948cd1aba4a6ea'
FINAL_BODY_SHA = '39d7731c2822d1cad3cedeaea4978d679a2c6c36619c8af91d586675a253565a'
HEAD = 'dc4e52ef3a7b6b6cde51f1ab54a68b2d63be5510'

def main():
    spec = importlib.util.spec_from_file_location('final_inventory_read_only_prepare', STAGE / 'prepare_registration.py')
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    checks = []
    def check(value, name):
        checks.append({'id': name, 'passed': bool(value)})
        if not value:
            raise ValueError(name)
    pure, auditor = prepare.pure, prepare.auditor
    raw = pure.read_bytes(FINAL)
    final = pure.loads(raw)
    template = pure.read_json(prepare.TEMPLATE)
    config = prepare.scope()
    check(hashlib.sha256(raw).hexdigest() == FINAL_FILE_SHA, 'external exact owned inventory file SHA')
    check(final['sha256'] == FINAL_BODY_SHA == pure.digest({k: v for k, v in final.items() if k != 'sha256'}), 'external exact inventory canonical body SHA')
    check(hashlib.sha256(pure.read_bytes(prepare.TEMPLATE)).hexdigest() == '9d316cabd8170576030d0f94cbb70453341a5248cfe74955670d2aa6beb89ba4', 'reviewed template bytes remain exact')
    check(set(final) == prepare.FIELDS and {k for k in template if template[k] != final[k]} == {'source_commit', 'sha256'}, 'only launch HEAD and corresponding checksum changed')
    check(final['source_commit'] == HEAD and len(HEAD) == 40 and all(c in '0123456789abcdef' for c in HEAD), 'exact full externally supplied committed launch HEAD')
    check(FINAL.resolve().parent == (PACKAGE / 'artifacts/short_m0_audit_registrations').resolve(), 'actual owned final registration directory')
    check(final['config_file_sha256'] == prepare.CONFIG_SHA == hashlib.sha256(pure.read_bytes(prepare.CONFIG)).hexdigest(), 'externally pinned selected config bytes unchanged')
    check(final['profile_body_sha256'] == prepare.PROFILE_SHA == pure.digest(config['short_phase_observation_profile']), 'externally pinned Phase3 profile unchanged')
    check(final['inventories'] == {section: config[key] for section, key in prepare.SECTIONS.items()}
        and final['counts'] == {'source': 131, 'runtime': 65, 'decoder': 17, 'data': 5, 'model': 7}, 'all exact source131/runtime65/decoder17/data5/model7 maps retained')
    before = {path: pure.file_info(path)['sha256'] for path in final['inventories']['source']}
    check(before == final['inventories']['source'], 'all131 current source bytes independently match final inventory')
    check(len(final['auditor_dependency_pins']) == 6 and final['auditor_dependency_pins'] == prepare.dependency_pins(), 'all six exact current reviewed auditor dependencies')
    check(final['qualified_runtime64_sha256'] == prepare.CORE64_SHA == template['qualified_runtime64_sha256'], 'unchanged qualified runtime64 plus exact AST extension metadata')
    auditor._source_admission(final)
    check(True, 'actual complete current source/admission/alias validator executed')
    sup = {'preflight': {'source_commit': HEAD}, 'short_RAM_monitor_diagnostic': {'path': 'SHORT_RAM_ABORT_DIAGNOSTIC.json',
        'attempted': False, 'persisted': False, 'write_error': None, 'original_abort_reason_preserved': None}}
    counts = auditor._phase3_admission(config, sup, final, prepare.PROFILE_SHA, config['short_phase_observation_profile'], {}, pure.require)
    check(counts == final['counts'], 'actual exact Phase3 inventory/core64/AST/purpose admission executed')
    after = {path: pure.file_info(path)['sha256'] for path in final['inventories']['source']}
    check(before == after and pure.read_bytes(FINAL) == raw, 'source131 and selected final inventory bytes stable throughout review')
    forbidden = [n for n in ('torch', 'transformers', 'tokenizers', 'psutil') if n in sys.modules]
    check(not forbidden, 'no framework/decoder/native probe imports')
    receipt = {'schema_version': 1, 'kind': 'FINAL_PHASE3_OWNED_INVENTORY_INDEPENDENT_SOURCE_JSON_REVIEW',
        'inventory_path': str(FINAL), 'inventory_file_sha256': FINAL_FILE_SHA, 'inventory_body_sha256': FINAL_BODY_SHA,
        'config_file_sha256': prepare.CONFIG_SHA, 'profile_body_sha256': prepare.PROFILE_SHA,
        'root_supplied_committed_launch_HEAD': HEAD, 'checks': checks, 'counts': counts,
        'source_map_sha256': pure.digest(before), 'auditor_dependency_pins': final['auditor_dependency_pins'],
        'actual_Git_HEAD_or_committed_blob_queried': False, 'config_commit_provenance_supplied_by_root': True,
        'only_template_HEAD_and_checksum_changed': True, 'forbidden_runtime_imports': forbidden,
        'runtime_decoder_model_binary_reads_or_probes': False, 'root_source_config_inventory_or_HEAD_mutated': False,
        'native_worker_started_by_reviewer': False, 'native_ready': False, 'native_PASS_or_study_result': False,
        'remaining_runtime_gates': ['authorized fresh current preflight/binary-byte validation', 'fresh unchanged resource admission', 'actual bounded supervised terminal and independent raw/model/resource audit']}
    receipt['sha256'] = pure.digest(receipt)
    target = HERE / ('phase3_final_inventory_review_%d.json' % time.time_ns())
    target.write_text(pure.canonical(receipt) + '\n', encoding='ascii')
    print(json.dumps({'receipt': str(target), 'receipt_body_sha256': receipt['sha256'], 'checks_passed': len(checks),
        'source_inventory_bytes_stable': before == after, 'only_template_HEAD_and_checksum_changed': True,
        'native_ready': False, 'forbidden_runtime_imports': forbidden}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
