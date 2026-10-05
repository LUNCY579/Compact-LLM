"""Root-authorized terminal metadata review. No copy, decoder, native or OS probes."""
import hashlib, importlib.util, json, sys, time
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STAGE = PACKAGE / 'artifacts/phase3_launch_registration_work'
SUP_SHA = '2571e27872e6206560cdedd1631b690f4f9420af6b3eb4c0dfc1cac30f619818'
DEATH_SHA = 'f5e79cfa99052ff01f73cc06902bdc1f4a331896613a9918b8a32f662ebf1158'
DEATH_BODY = '58c0681b9eab8b90cd08dd866bcc536f129e4b3a9c312308544c14614cb4b6b3'

def main():
    spec = importlib.util.spec_from_file_location('readonly_actual_terminal_preserver_review', STAGE / 'preserve_terminal.py')
    preserve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preserve)
    pure, registration = preserve.pure, preserve.registration
    source_before = hashlib.sha256((STAGE / 'preserve_terminal.py').read_bytes()).hexdigest()
    scope = registration.scope()
    final = pure.read_json(preserve.INVENTORY)
    leaf = PACKAGE / 'artifacts' / preserve.LEAF_NAME
    death = STAGE / 'ROOT_CURRENT_OWNED_DEATH_BATCH04_A1_001.json'
    sup = preserve.exact_json(leaf / 'SUPERVISOR_RECEIPT.json', SUP_SHA)
    proof = preserve.exact_json(death, DEATH_SHA)
    assert proof['sha256'] == DEATH_BODY
    source_map_before = {path: pure.file_info(path)['sha256'] for path in scope['source_pins']}
    assert source_map_before == scope['source_pins'] == final['inventories']['source']
    # Any unplanned source-only preserver attempt to reach Git or write copied
    # evidence fails. The actual terminal functions below remain unmocked.
    with (patch.object(registration.subprocess, 'run', side_effect=AssertionError('real Git/OS probe forbidden')) as probe,
            patch.object(preserve, 'copy_exact', side_effect=AssertionError('copy/write forbidden in read-only terminal review')) as copy):
        gate = preserve.terminal_gate(PACKAGE, leaf, scope, sup, SUP_SHA, preserve.HEAD,
            registration.CONFIG_SHA, death, DEATH_SHA)
        raw = preserve.raw_inventory(leaf, sup)
        assert gate['Root_current_death_observation']['all_observed_owned_births'] == 7
        assert len(raw) == 591 and len(sup['partial_outputs']) == 590
        assert sup['status'] == 'SUPERVISED_WORKER_EXITED_ZERO' and sup['returncode'] == 0
        assert probe.call_count == copy.call_count == 0
    assert pure.file_info(leaf / 'SUPERVISOR_RECEIPT.json')['sha256'] == SUP_SHA
    assert pure.file_info(death)['sha256'] == DEATH_SHA
    source_map_after = {path: pure.file_info(path)['sha256'] for path in scope['source_pins']}
    assert source_map_before == source_map_after
    forbidden = [name for name in ('torch', 'transformers', 'tokenizers', 'psutil') if name in sys.modules]
    assert not forbidden
    source_after = hashlib.sha256((STAGE / 'preserve_terminal.py').read_bytes()).hexdigest()
    receipt = {'schema_version': 1, 'kind': 'ACTUAL_PHASE3_TERMINAL_GATE_RAW_INVENTORY_READONLY_METADATA_REVIEW',
        'supervisor_file_sha256': SUP_SHA, 'Root_death_file_sha256': DEATH_SHA, 'Root_death_body_sha256': DEATH_BODY,
        'terminal_gate': gate, 'exact_raw_files': len(raw), 'supervisor_partial_outputs': len(sup['partial_outputs']),
        'complete_raw_byte_map_sha256': pure.digest(raw), 'current_source131_byte_map_sha256': pure.digest(source_map_after),
        'source131_before_after_equal': source_map_before == source_map_after,
        'preserver_source_before': source_before, 'preserver_source_after': source_after,
        'preserver_source_stable_during_run': source_before == source_after,
        'current_owned_identity_roster_count': 7, 'live_OS_Git_GPU_model_decoder_encoder_probes': False,
        'runtime_model_binary_assets_read': False, 'original_raw_or_source_modified': False,
        'destination_or_preservation_receipt_written': False, 'forbidden_runtime_imports': forbidden,
        'native_raw_full_audit_done': False, 'grading_scoring_or_study_result': False}
    receipt['sha256'] = pure.digest(receipt)
    target = HERE / ('phase3_terminal_gate_readonly_review_%d.json' % time.time_ns())
    target.write_text(pure.canonical(receipt) + '\n', encoding='ascii')
    print(json.dumps({'receipt': str(target), 'receipt_sha256': receipt['sha256'], 'exact_raw_files': len(raw),
        'partial_outputs': len(sup['partial_outputs']), 'owned_births_dead': 7, 'sources131_stable': True,
        'preserver_source_stable': source_before == source_after, 'native_probes_and_copy_calls': 0}))
    return 0 if source_before == source_after else 1

if __name__ == '__main__':
    raise SystemExit(main())
