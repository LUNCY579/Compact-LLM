"""Create an artifact-derived screen config or freeze its source registration."""
import argparse
import json
from pathlib import Path
import sys

PROGRAMME = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROGRAMME))


def main():
    from cross_scale.generation_stage.v1.src.protocol import build_config
    from cross_scale.generation_stage.v1.src.provenance import freeze_registration, STAGE
    from shared.guards import require_within
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('config')
    build.add_argument('--export-manifest', type=Path, required=True)
    build.add_argument('--calibration-artifact', type=Path, required=True)
    build.add_argument('--device', choices=['cpu', 'cuda:0'], required=True)
    build.add_argument('--output', type=Path, required=True)
    freeze = sub.add_parser('register')
    freeze.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'register':
        print(freeze_registration(args.config))
    else:
        config = build_config(export_manifest_path=args.export_manifest,
                              calibration_artifact_path=args.calibration_artifact, device=args.device)
        output = require_within(args.output, STAGE / 'configs')
        with output.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        print(output)


if __name__ == '__main__':
    main()
