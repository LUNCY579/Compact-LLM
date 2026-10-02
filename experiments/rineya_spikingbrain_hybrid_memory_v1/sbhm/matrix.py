"""Print the paired execution plan; --execute additionally needs explicit grants.

Run: python -m sbhm.matrix --config local.json --output /new/run/root
No model loading or filesystem changes happen in plan-only mode.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def build_plan(config, root):
    root = Path(root)
    common = [sys.executable, '-m', 'sbhm']
    def command(stage, arm, directory, qualification=None, delta=None, teacher=None):
        result = common + [stage, '--config', str(config), '--arm', arm, '--output', str(root/directory), '--allow-model-load']
        if stage in ('distill', 'recover'):
            result += ['--allow-training']
        for flag, value in [('--qualification', qualification), ('--delta', delta), ('--teacher', teacher)]:
            if value:
                result += [flag, str(root/value)]
        return result
    plan = [command('qualify','A','A-qualification'),
            command('teacher','A','teacher','A-qualification/receipt.json'),
            command('evaluate','A','A-evaluation','A-qualification/receipt.json')]
    for arm in 'BCD':
        plan += [command('qualify',arm,f'{arm}-initial-qualification'),
                 command('distill',arm,f'{arm}-distill',f'{arm}-initial-qualification/receipt.json',teacher='teacher'),
                 command('qualify',arm,f'{arm}-distilled-qualification',delta=f'{arm}-distill'),
                 command('recover',arm,f'{arm}-recover',f'{arm}-distilled-qualification/receipt.json',f'{arm}-distill'),
                 command('qualify',arm,f'{arm}-recovered-qualification',delta=f'{arm}-recover'),
                 command('evaluate',arm,f'{arm}-evaluation',f'{arm}-recovered-qualification/receipt.json',f'{arm}-recover')]
    plan += [common + ['compare'] + [str(root/f'{a}-evaluation/receipt.json') for a in 'ABCD'] + ['--output',str(root/'comparison.json')]]
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--allow-model-load',action='store_true')
    p.add_argument('--allow-training',action='store_true')
    args = p.parse_args()
    plan = build_plan(Path(args.config).resolve(), Path(args.output).resolve())
    if not args.execute:
        print(json.dumps({'status':'PLAN_ONLY','commands':plan},indent=2))
        return 0
    if not args.allow_model_load or not args.allow_training:
        p.error('--execute requires --allow-model-load and --allow-training; per-stage grants still apply')
    Path(args.output).mkdir(parents=True,exist_ok=False)
    for command in plan:
        completed = subprocess.run(command,check=False)
        if completed.returncode:
            return completed.returncode  # fail closed; never skip a blocked arm/stage
    return 0


if __name__ == '__main__':
    sys.exit(main())
