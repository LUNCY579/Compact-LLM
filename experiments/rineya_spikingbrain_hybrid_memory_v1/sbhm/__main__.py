"""Run with python -m sbhm from this experiment directory."""
import argparse
import json
import sys
from pathlib import Path
from .protocol import Blocked, read_json, bind, write_json, environment, code_hash, validate, resources, source_lock, digest, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'qualify', 'teacher', 'distill', 'recover', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--arm', choices=list('ABCD'), default='A')
        p.add_argument('--output')
        p.add_argument('--qualification')
        p.add_argument('--delta')
        p.add_argument('--teacher')
        p.add_argument('--allow-model-load', action='store_true')
        p.add_argument('--allow-training', action='store_true')
    p = sub.add_parser('make-fixture-corpus')
    p.add_argument('--output', required=True)
    p = sub.add_parser('compare')
    p.add_argument('receipts', nargs=4)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'make-fixture-corpus':
            from .data import make_fixture_corpus
            make_fixture_corpus(args.output)
            result = {'status': 'ENGINEERING_FIXTURES_ONLY', 'reviewed_for_sbhm': False}
        elif args.command == 'compare':
            from .pipeline import compare
            result = compare(args.receipts)
            write_json(args.output, result)
        elif args.command == 'preflight':
            c = read_json(args.config)
            validate(c)
            failures, source = [], None
            for check in (lambda: resources(c['device']), lambda: source_lock(c)):
                try:
                    value = check()
                    if isinstance(value, str):
                        source = value
                except (Blocked, OSError, ValueError) as e:
                    failures.append(str(e))
            env = environment()
            if env['versions']['transformers'] != '4.51.3':
                failures.append('transformers==4.51.3 is required')
            if not c.get('grant'):
                failures.append('dated operator grant is absent')
            result = {'status': 'BLOCKED' if failures else 'PREFLIGHT_ONLY_NOT_EXECUTION_AUTHORITY',
                      'blockers': failures, 'environment': env, 'code_sha256': code_hash(),
                      'source_sha256': source, 'config_sha256': digest(c),
                      'corpus_sha256': sha(c['corpus_manifest']) if c.get('corpus_manifest') else None, 'model_loaded': False}
        else:
            if not args.output:
                raise Blocked('--output must name a NEW directory')
            from .pipeline import execute
            result = execute(args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result.get('status') == 'BLOCKED' else 0
    except (Blocked, OSError, ValueError, TypeError, KeyError, ImportError, AssertionError, RuntimeError) as e:
        print(json.dumps({'status': 'BLOCKED_OR_FAILED', 'type': type(e).__name__, 'reason': str(e), 'efficacy_claim': False}, indent=2))
        return 2


if __name__ == '__main__':
    sys.exit(main())
