"""Read-only actual frozen admission gate; creates no worker/model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from .authority import PACKAGE, CAMPAIGN, sha
from .probe_supervisor import check_parent


def main(name):
    if Path(name).name!=name:raise ValueError('V3 filename only')
    path=PACKAGE/'probe_admission_v1.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    report={'models':0,'worker_created':False,'admission_sha256':sha(path),
        'captured_utc':datetime.now(timezone.utc).isoformat()}
    try:
        _,inventory=check_parent(Path(data['source']).resolve(),PACKAGE/'runs/native-prefix-probe-001',path)
        report.update(status='READY_PARENT_GATES_NATIVE_QUALIFICATION_REQUIRED',GPU=inventory,reasons=[])
    except Exception as error:
        report.update(status='NO_DISPATCH',reasons=[type(error).__name__+': '+str(error)])
    assert 'torch' not in sys.modules
    with (PACKAGE/name).open('x',encoding='utf-8') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(report));return 0 if not report['reasons'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True)
    raise SystemExit(main(parser.parse_args().name))
