"""Contained bootstrap: validate scope and clock before importing Torch."""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import time
from .authority import validate_probe_scope


def main(release, source, out, admission):
    data=json.loads(Path(admission).read_text(encoding='utf-8'))
    validate_probe_scope(data)
    value=float(os.environ.get('PASSIVE_PROBE_PARENT_STARTED_PERF_COUNTER','nan'))
    if not math.isfinite(value) or not 0 <= time.perf_counter()-value <= 900:
        raise PermissionError('finite parent-owned probe deadline required')
    end=time.monotonic()+15
    while not Path(release).is_file():
        if time.monotonic()>end:
            raise TimeoutError('contained parent release absent')
        time.sleep(.05)
    packet=json.loads(Path(release).read_text(encoding='utf-8'))
    if packet.get('contained') is not True or not packet.get('token'):
        raise PermissionError('WindowsJob containment/release not proven')
    import sys
    sys.argv=['passive-prefix-probe','--source',str(source),'--out',str(out),'--admission',str(admission)]
    runpy.run_module('experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.probe_native',run_name='__main__')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('release','source','out','admission'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    main(args.release,args.source,args.out,args.admission)
