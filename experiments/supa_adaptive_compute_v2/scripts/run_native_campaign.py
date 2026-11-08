"""Serial, resumable, per-arm native scheduling; each child owns one new shard."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

PROGRAM = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROGRAM))
from shared.datasets import load_frozen_dataset
from shared.evaluation import shard_cases, _read_complete
from shared.hashing import sha256_file
from shared.manifests import verify_registration
from native_armr.src.protocol import load_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PROGRAM/'native_armr/configs/protocol.json')
    parser.add_argument('--role',choices=['architecture_dev','confirmatory_dev','confirmatory_diagnostics'],required=True)
    parser.add_argument('--arms',nargs='+',choices=['A0','A1','A3','A16','AM'],default=['A0','A1','A3','A16','AM'])
    parser.add_argument('--shard-count',type=int,default=32)
    parser.add_argument('--max-wall-seconds',type=float,default=0,help='stop before starting another shard after this budget; zero means full campaign')
    parser.add_argument('--plan',action='store_true')
    args = parser.parse_args()
    if args.shard_count <= 0:
        parser.error('--shard-count must be positive')
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds < 0:
        parser.error('--max-wall-seconds must be finite and nonnegative')
    if len(args.arms) != len(set(args.arms)):
        parser.error('--arms must not contain duplicates')
    if 'A0' in args.arms and args.arms[0] != 'A0':
        parser.error('A0 must precede candidates when included')
    config_path = args.config
    config = load_protocol(config_path)
    registration = verify_registration('native_armr',config_path)
    rows = load_frozen_dataset(args.manifest,track='native_armr',role=args.role)
    complete = {}
    for path in (PROGRAM/'native_armr/evidence').glob('*/*/run_manifest.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('status') != 'COMPLETE' or value.get('role') != args.role or value.get('registration_sha256') != registration['registration_sha256'] or value.get('dataset_manifest_sha256') != sha256_file(args.manifest) or value.get('shard_count') != args.shard_count:
            continue
        _read_complete(path.parent,'native_armr')
        actual_ids = [r['example_id'] for r in shard_cases(rows,value['shard_index'],args.shard_count)]
        if value['example_ids'] != actual_ids:
            raise RuntimeError('saved shard does not match deterministic current case allocation')
        key = (value['arm'],value['shard_index'])
        if key in complete:
            raise RuntimeError('ambiguous duplicate completed shard; cannot resume silently')
        complete[key] = str(path.parent)
    pending = []
    for arm in args.arms:
        for index in range(args.shard_count):
            cases = shard_cases(rows,index,args.shard_count)
            if cases and (arm,index) not in complete:
                pending.append((arm,index,len(cases)))
    print(json.dumps({'role':args.role,'complete_shards':len(complete),'pending':pending,'sequential':True},indent=2),flush=True)
    if args.plan:
        return
    start = time.monotonic()
    for arm,index,count in pending:
        if args.max_wall_seconds and time.monotonic()-start >= args.max_wall_seconds:
            print('CAMPAIGN_PAUSED_BETWEEN_SHARDS; repeat the same command to resume',flush=True)
            return
        command = [sys.executable,str(PROGRAM/'native_armr/scripts/run_large_eval.py'),
                   '--config',str(config_path.resolve()),'--manifest',str(args.manifest.resolve()),'--role',args.role,'--arm',arm,
                   '--shard-index',str(index),'--shard-count',str(args.shard_count)]
        subprocess.run(command,cwd=PROGRAM.parents[1],check=True)
    print('CAMPAIGN_COMPLETE; run independent native aggregation/audit',flush=True)


if __name__ == '__main__':
    main()
