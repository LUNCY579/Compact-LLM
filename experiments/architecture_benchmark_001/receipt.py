"""Summarize an executed pytest JUnit report without issuing a model qualification."""
from __future__ import annotations
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from .gates import GateClosed, implementation_digest, runtime, write_once
from .source import sha256_file


def summarize(junit: Path, *, require_native: bool = False) -> dict:
    cases = ET.parse(junit).getroot().findall('.//testcase')
    skipped = [c for c in cases if c.find('skipped') is not None]
    failed = [c for c in cases if c.find('failure') is not None or c.find('error') is not None]
    native = [c for c in cases if c.get('classname', '').rsplit('.', 1)[-1] == 'test_native'
              and c.get('name', '').startswith('test_native_random_qwen3_logits[')
              and c not in skipped and c not in failed]
    native_variants = {c.get('name') for c in native}
    required_native = {'test_native_random_qwen3_logits[False]', 'test_native_random_qwen3_logits[True]'}
    if not cases or failed or (require_native and (not required_native.issubset(native_variants) or skipped)):
        raise GateClosed('mechanical report failed, empty, or missing required native random-model checks')
    return {'schema': 1, 'experiment_id': 'RAB-001', 'evidence_kind': 'mechanical_tests_only',
            'status': 'PASS', 'passed': len(cases)-len(skipped)-len(failed), 'skipped': len(skipped),
            'skip_reasons': [c.find('skipped').get('message', '') for c in skipped],
            'native_random_model_cases_passed': len(native), 'official_weights_loaded': False,
            'implementation_sha256': implementation_digest(), 'runtime': runtime(),
            'junit_sha256': sha256_file(junit), 'licenses_training': False, 'licenses_efficacy_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--junit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-native', action='store_true')
    parser.add_argument('--source-revision', default='local_overlay_no_commit_yet')
    args = parser.parse_args()
    result = summarize(args.junit, require_native=args.require_native)
    result['source_revision'] = args.source_revision
    write_once(args.out, result)


if __name__ == '__main__':
    main()
