"""Run one registered B architecture shard, or explicitly abandon a dead worker."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def main():
    from cross_scale.generation_stage.v1.src.execution import run_shard, abandon_shard
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cell')
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--shard-count', type=int)
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--abandon', type=Path)
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.abandon:
        if args.cell is not None or args.shard_index is not None or not args.reason:
            parser.error('--abandon requires --reason and excludes --cell/--shard-index')
        result = abandon_shard(args.config, args.abandon, args.reason)
    else:
        if args.cell is None or args.shard_index is None or args.reason:
            parser.error('execution requires --cell and --shard-index')
        result = run_shard(args.config, args.cell, args.shard_index,
                           shard_count=args.shard_count, attempt=args.attempt, model_path=args.model_path)
    print(result, flush=True)


if __name__ == '__main__':
    main()
