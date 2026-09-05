"""Create immutable per-track registrations; commit them before model execution."""
import argparse
from pathlib import Path
import sys

PROGRAM = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROGRAM))
from shared.manifests import freeze_registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track',choices=['native_armr','cross_scale'],required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    args = parser.parse_args()
    config = args.config or PROGRAM / args.track / 'configs' / ('protocol.json' if args.track == 'native_armr' else 'alignment_v2.json')
    print(freeze_registration(args.track,config,[args.manifest]))


if __name__ == '__main__':
    main()
