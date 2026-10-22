"""Explicit bounded commands; importing this package never loads a model."""
import argparse
from pathlib import Path
from .safety import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('fixture','native-preflight','native-dev'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path)
    args=p.parse_args()
    if args.stage=='fixture':
        from .experiment import run_fixture_experiment
        run_fixture_experiment(args.out); return 0
    from .native import preflight,run
    if args.stage=='native-dev': return run(args.out,args.source)
    args.out.mkdir(parents=True,exist_ok=False)
    receipt=preflight(args.source)
    write_json(args.out/'preflight.json',receipt)
    print(receipt['status'])
    return 2 if receipt['status']=='BLOCKED' else 0


if __name__=='__main__':
    raise SystemExit(main())
