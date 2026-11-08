"""Offline stdlib check of the complete recorded 22-document native stdout log."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(r'C:/Users/clclc/BusinessProject/ChatBubble-rineya-spikingbrain-hybrid-memory-v1/experiments/rineya_spikingbrain_hybrid_memory_v1')
RUN = ROOT / 'runs/titan_transition_v2_20261004_memory_efficient'
LOG = ROOT / 'evidence/native_20261004/matrix-memory-efficient-execution.txt'
OUT = Path(r'C:/Users/clclc/BusinessProject/sbhm-001-audits/20261005-closeout-01a10d69/provenance_log_receipt.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    text = LOG.read_text(encoding='utf-8')
    decoder = json.JSONDecoder()
    documents, offset = [], 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        document, offset = decoder.raw_decode(text, offset)
        documents.append(document)
    stages = ['A-qualification', 'teacher', 'A-evaluation']
    for arm in 'BCD':
        stages += [arm + '-' + name for name in ['initial-qualification', 'distill', 'distilled-qualification', 'recover', 'recovered-qualification', 'evaluation']]
    files = [RUN / stage / 'receipt.json' for stage in stages] + [RUN / 'comparison.json']
    failures = []
    if len(documents) != 22:
        failures.append('log must contain exactly 22 complete JSON documents')
    checks = []
    for i, (document, path) in enumerate(zip(documents, files)):
        matched = document == json.loads(path.read_text(encoding='utf-8'))
        if not matched:
            failures.append('document %s differs from %s' % (i, path))
        checks.append({'command_index_one_based': i + 1, 'status': document['status'], 'receipt': str(path), 'receipt_sha256': sha(path), 'full_document_matches_authoritative_receipt': matched})
    handoff = ROOT / 'evidence/native_20261004/final_completed_handoff.json'
    reported_exit = json.loads(handoff.read_text(encoding='utf-8'))['matrix_exit_code']
    if reported_exit != 0:
        failures.append('historical handoff does not report exit 0')
    result = {'status': 'PASS_ALL_22_LOG_DOCUMENTS_MATCH_AUTHORITATIVE_RECORDS' if not failures else 'FAIL_NATIVE_LOG_ARTIFACT_CHECK', 'created_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(__file__), 'log': str(LOG), 'log_sha256': sha(LOG), 'parsed_documents': len(documents), 'all_log_content_consumed_as_JSON_plus_whitespace': offset == len(text), 'ordered_receipt_checks': checks, 'matrix_code_sha256': sha(ROOT / 'sbhm/matrix.py'), 'historical_handoff': str(handoff), 'historical_handoff_sha256': sha(handoff), 'historically_reported_matrix_exit_code': reported_exit, 'process_exit_independently_reexecuted_or_observed': False, 'native_execution': False, 'failures': failures, 'limitation': 'Complete matching stdout supports the stored completion record; the original shell exit was not independently observed. Bound matrix code stops immediately on any subprocess nonzero status.'}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'documents': len(documents), 'failures': failures, 'output': str(OUT)}, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
