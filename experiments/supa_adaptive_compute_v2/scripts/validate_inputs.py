"""Tokenization-only frozen-bank validation; no model generation or fitting."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

PROGRAM = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROGRAM))
from shared.datasets import load_frozen_dataset,validate_rows
from shared.hashing import sha256_file,write_json
from shared.integrity import checkpoint_files
from shared.model_loading import resolve_model
from shared.manifests import new_run,utc_now


def main():
    from transformers import AutoTokenizer
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track',choices=['native_armr','cross_scale'],required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args = parser.parse_args()
    path = resolve_model()
    identity = checkpoint_files(path)
    tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=True)
    config_path = PROGRAM/args.track/'configs'/('protocol.json' if args.track == 'native_armr' else 'alignment_v2.json')
    config = json.loads(config_path.read_text())
    output,receipt = new_run(args.track,'01_dataset_validation',{'model_loaded':False,
        'validation':'independent_semantic_oracle_and_tokenizer_only',
        'dataset_manifest_sha256':sha256_file(args.manifest),'checkpoint_files':identity,
        'config_sha256':sha256_file(config_path), 'scorer_status':'deterministic_exact_or_json'})
    start = time.perf_counter()
    try:
        report = {}
        for role in ('calibration','architecture_dev','confirmatory_dev','confirmatory_diagnostics'):
            rows = load_frozen_dataset(args.manifest,track=args.track,role=role)
            inputs,answers,too_long = [],[],[]
            for row in rows:
                rendered = tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
                tokens = len(tokenizer.encode(rendered))
                gold = len(tokenizer.encode(row['answer'],add_special_tokens=False))
                inputs.append(tokens)
                answers.append(gold)
                cap = config.get('calibration_max_tokens',1024) if role == 'calibration' and args.track == 'cross_scale' else config.get('max_input_tokens',1024)
                if tokens > cap or gold+1 > min(row['max_new_tokens'],config.get('max_new_tokens',48)):
                    too_long.append({'example_id':row['example_id'],'input_tokens':tokens,'gold_tokens':gold,'input_cap':cap})
            report[role] = {**validate_rows(rows),'max_input_tokens':max(inputs),'max_gold_tokens':max(answers),
                            'cap_violations':too_long}
        write_json(output/'validation.json',report)
        if any(row['cap_violations'] for row in report.values()):
            raise RuntimeError('frozen inputs exceed prospective token budget; resolve before registration/model outputs')
        receipt.update(status='COMPLETE',validation_sha256=sha256_file(output/'validation.json'),
                       no_cap_violations=True)
    except BaseException as exc:
        receipt.update(status='FAILED',failure_receipt={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        memory = psutil.Process().memory_info()
        receipt.update(end_utc=utc_now(),wall_seconds=time.perf_counter()-start,
                       peak_memory_bytes=getattr(memory,'peak_wset',memory.rss))
        write_json(output/'run_manifest.json',receipt)
    print(output)


if __name__ == '__main__':
    main()
