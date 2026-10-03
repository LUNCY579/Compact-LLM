"""Optional JSON Schema check of every public RL and SFT view (requires jsonschema)."""
from __future__ import annotations
import argparse
import importlib.metadata
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forge_rl.common import read_jsonl, loads, dumps, write_json


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError('Receipt already exists')
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise SystemExit('Optional JSON Schema validation requires jsonschema. The standard-library semantic validator is scripts/verify.py.') from exc
    counts={}
    for kind in ('rl','sft'):
        schema=loads((args.root/'schemas'/f'{kind}.schema.json').read_text(encoding='utf-8'))
        Draft202012Validator.check_schema(schema)
        validator=Draft202012Validator(schema)
        folders=('rl','rl_reason') if kind=='rl' else ('sft_final','sft_reason')
        for folder in folders:
            for path in sorted((args.root/'data'/folder).glob('*.jsonl.gz')):
                n=0
                for row in read_jsonl(path):
                    validator.validate(row)
                    n+=1
                counts[str(path.relative_to(args.root)).replace('\\','/')]=n
    if not counts:
        raise ValueError('No public data files found')
    result={'status':'PASS','draft':'2020-12','jsonschema_version':importlib.metadata.version('jsonschema'),
            'files':counts,'total_public_view_rows':sum(counts.values()),
            'scope':'Public RL and SFT row schemas, not a substitute for semantic oracle checks','model_calls':0}
    if args.output:write_json(args.output,result)
    print(dumps(result))


if __name__=='__main__':main()
