"""Register the fresh bounded session envelope for a new DOSELORA experiment root.

Runs BETWEEN prepare.py and finalize_prelaunch.py.

prepare.py writes SESSION_ENVELOPE.json with the ORIGINAL prospective pretraining
reserve (19.5GiB). AMENDMENT_STORAGE.md supersedes only that reserve, replacing it
with the sequential-export schedule (17GiB before training, 16.3GiB before the
second export). launch.py and posttrain.py both read SESSION_ENVELOPE_V2.json BY
NAME, and finalize_prelaunch.py hashes both SESSION_ENVELOPE_V2.json and
AMENDMENT_STORAGE.md into PRELAUNCH_BINDING.json, so both files must exist in the
root before the freeze.

This script is deliberately NOT one of the five frozen executables. It only writes
create-only files and it refuses to loosen anything: every envelope key except the
session clock is compared against the REFERENCE envelope from the predecessor root
and any difference is a hard failure. A fresh session may move its own deadline
forward; it may not buy itself a larger budget.
"""
from pathlib import Path
import json
import shutil
import sys
from prepare import OUT, HERE, RECORDS, sha, read, write

REFERENCE = RECORDS / 'highdose-lora001-20260920' / 'SESSION_ENVELOPE_V2.json'

# The storage amendment supersedes ONLY the prospective pretraining reserve.
AMENDED = {
    'pretrain_free_disk_bytes': 18253611008,        # 17.0GiB before training
    'pre_second_export_free_disk_bytes': 17501991731,  # 16.3GiB before the norm control
}
# Keys whose values are allowed to differ from the reference envelope, because they
# ARE the fresh session. Everything else must match byte-for-byte.
CLOCK = ('session_start_unix', 'deadline_unix')


def main():
    reference = read(REFERENCE)
    envelope = read(OUT / 'SESSION_ENVELOPE.json')

    # AMENDMENT_STORAGE.md is hashed by finalize_prelaunch.py but copied by nothing.
    amendment = OUT / 'AMENDMENT_STORAGE.md'
    if not amendment.exists():
        with (HERE / 'AMENDMENT_STORAGE.md').open('rb') as src, amendment.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    if sha(amendment) != sha(HERE / 'AMENDMENT_STORAGE.md'):
        raise ValueError('Storage amendment copy differs')
    if sha(amendment) != reference['storage_amendment_sha256']:
        raise ValueError('Storage amendment differs from the one the reference envelope registered')

    value = {**envelope, **AMENDED, 'storage_amendment_sha256': sha(amendment)}

    # The session clock moves forward; nothing else may move at all.
    if set(value) != set(reference):
        raise ValueError(f'Envelope key set differs: {set(value) ^ set(reference)}')
    for key in sorted(value):
        if key in CLOCK:
            continue
        if value[key] != reference[key]:
            raise ValueError(f'Envelope key {key!r} differs from the reference envelope: '
                             f'{value[key]!r} vs {reference[key]!r}')
    if value['deadline_unix'] - value['session_start_unix'] != reference['total_owned_work_wall_seconds']:
        raise ValueError('Fresh session length must equal the registered owned-work envelope')
    if value['session_start_unix'] <= reference['session_start_unix']:
        raise ValueError('A fresh session must start after the session it succeeds')

    write(OUT / 'SESSION_ENVELOPE_V2.json', value)
    print(json.dumps({
        'status': 'FRESH_BOUNDED_SESSION_REGISTERED',
        'root': str(OUT),
        'reference_envelope': str(REFERENCE),
        'session_start_unix': value['session_start_unix'],
        'deadline_unix': value['deadline_unix'],
        'session_seconds': value['deadline_unix'] - value['session_start_unix'],
        'keys_differing_from_reference': list(CLOCK),
        'keys_verified_identical': sorted(k for k in value if k not in CLOCK),
        'envelope_sha256': sha(OUT / 'SESSION_ENVELOPE_V2.json'),
        'model_execution': False,
    }, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
