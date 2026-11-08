"""Run one resumable cell/phase or finalize a committed CPU/GPU comparison."""
import argparse
from pathlib import Path
import sys

PROGRAMME = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROGRAMME))


def main():
    from cross_scale.generation_stage.v1.src.qualification import run_phase, finalize_cell, abandon_phase
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    actions = parser.add_subparsers(dest='action', required=True)
    phase = actions.add_parser('phase')
    phase.add_argument('--cell', required=True)
    phase.add_argument('--phase', required=True, choices=['cpu', 'gpu'])
    phase.add_argument('--attempt', type=int, default=0)
    final = actions.add_parser('finalize')
    final.add_argument('--cell', required=True)
    abandon = actions.add_parser('abandon')
    abandon.add_argument('--cell', required=True)
    abandon.add_argument('--phase', required=True, choices=['cpu', 'gpu'])
    abandon.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'phase':
        print(run_phase(args.config, args.cell, args.phase, attempt=args.attempt))
    elif args.action == 'finalize':
        print(finalize_cell(args.config, args.cell))
    else:
        print(abandon_phase(args.config, args.cell, args.phase, args.attempt))


if __name__ == '__main__':
    main()
