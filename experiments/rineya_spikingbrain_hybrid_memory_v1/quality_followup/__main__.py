"""Offline preparation by default; native stages require an exact separate grant."""
import argparse
import json
from pathlib import Path
import sys

from .utils import Blocked, read_json, write_json, external_path, require

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--source-tokenizer', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('audit')
    p.add_argument('--packet', required=True)
    p.add_argument('--source-tokenizer')
    p.add_argument('--output')
    p = sub.add_parser('prepare')
    for name in ('packet', 'source-config', 'review', 'output', 'owner-thread-id'):
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('plan')
    p.add_argument('--config', required=True)
    p = sub.add_parser('stage')
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=('qualify', 'train', 'evaluate'), required=True)
    p.add_argument('--condition', choices=('anchor', 'control', 'candidate'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--qualification')
    p.add_argument('--delta')
    p.add_argument('--allow-model-load', action='store_true')
    p.add_argument('--allow-training', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('--config', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--allow-model-load', action='store_true')
    p.add_argument('--allow-training', action='store_true')
    p = sub.add_parser('compare')
    p.add_argument('--config', required=True)
    p.add_argument('--reports', nargs=3)
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            from .corpus import build_packet
            result = build_packet(args.output, args.source_tokenizer)
        elif args.command == 'audit':
            from .corpus import load_tokenizer
            from .audit import audit_packet
            tokenizer = load_tokenizer(args.source_tokenizer) if args.source_tokenizer else None
            result = audit_packet(args.packet, tokenizer=tokenizer)
            if args.output:
                write_json(args.output, result)
        elif args.command == 'prepare':
            from .prepare import prepare_study
            result = prepare_study(args.packet, args.source_config, args.review, args.output, args.owner_thread_id)
        elif args.command == 'plan' or (args.command == 'run' and not args.execute):
            from .runner import build_plan
            result = build_plan(args.config)
        elif args.command == 'stage':
            from .runner import execute_stage
            result = execute_stage(args.config, args.stage, args.condition, args.output,
                                   qualification=args.qualification, delta=args.delta,
                                   allow_model_load=args.allow_model_load, allow_training=args.allow_training)
        elif args.command == 'run':
            from .runner import execute_plan
            result = execute_plan(args.config, allow_model_load=args.allow_model_load,
                                  allow_training=args.allow_training)
        else:
            from .comparison import compare_reports
            from .corpus import load_packet
            from .protocol import load_config, bind, STABLE_KEYS
            conf = load_config(args.config)
            run = external_path(conf['native_run_root'])
            expected_paths = [run / f'{c}-evaluation' / 'receipt.json' for c in ('anchor', 'control', 'candidate')]
            reports = [Path(p).resolve() for p in args.reports] if args.reports else expected_paths
            require(reports == expected_paths, 'comparison accepts only the three exact proposed report paths')
            require(external_path(args.output) == run / 'comparison.json', 'comparison output must match the proposal')
            expected = bind(conf, 'anchor')  # Offline byte/hash checks; no model/GPU import.
            loaded = {c: read_json(path) for c, path in zip(('anchor', 'control', 'candidate'), reports)}
            for condition, report in loaded.items():
                require(all(report.get('binding', {}).get(k) == expected[k] for k in STABLE_KEYS),
                        'comparison report belongs to stale code/config/corpus/source/environment')
                current = bind(conf, condition, None if condition == 'anchor' else run / f'{condition}-train')
                require(report.get('binding') == current, 'comparison variant or saved training exposure changed')
            result = compare_reports(loaded, load_packet(conf['corpus_packet']))
            write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (Blocked, ValueError, OSError, KeyError) as error:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(error), 'native_authorization_implied': False}))
        return 2

if __name__ == '__main__':
    sys.exit(main())
