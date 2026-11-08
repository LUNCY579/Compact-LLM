"""Optional Parquet conversion and local-only tokenizer checks; never truncates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .contracts import require,filehash
from .corpus import readrows


def parquet(data:Path,out:Path):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:raise RuntimeError('BLOCKED: pyarrow is absent; complete JSONL remains usable. No Parquet file was written.') from e
    require(not out.exists(),'new output directory required');out.mkdir(parents=True)
    receipt={'engine':'pyarrow','version':pa.__version__,'files':{}}
    for path in sorted((data/'verl_jsonl').glob('*.jsonl')):
        rows=list(readrows(path));dest=out/(path.stem+'.parquet')
        pq.write_table(pa.Table.from_pylist(rows),dest,compression='zstd')
        require(pq.read_table(dest).to_pylist()==rows,'Parquet roundtrip mismatch')
        receipt['files'][dest.name]={'rows':len(rows),'sha256':filehash(dest)}
    return receipt


def tokenizer_check(data:Path,tokenizer_path:Path,context:int,response_budget:int):
    require(tokenizer_path.is_dir() and type(context) is int and type(response_budget) is int and context>response_budget>0,'tokenizer path or budget')
    try:from transformers import AutoTokenizer
    except ImportError as e:raise RuntimeError('BLOCKED: Transformers unavailable; no tokenizer was loaded.') from e
    tok=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True,trust_remote_code=False)
    require(tok.eos_token_id is not None,'missing EOS')
    over=[];max_tokens=0;rows=0
    for path in sorted((data/'canonical').glob('*.jsonl')):
        for row in readrows(path):
            ids=tok.apply_chat_template(row['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            require(type(ids) is list and ids and all(type(i) is int for i in ids),'template output')
            n=len(ids);max_tokens=max(max_tokens,n);rows+=1
            if n+response_budget>context:over.append(row['id'])
    return {'status':'BLOCKED_OVERLENGTH' if over else 'PASS_PROMPT_LENGTH_ONLY','rows':rows,'max_prompt_tokens':max_tokens,'overlength_ids':over,
            'context':context,'reserved_response_tokens':response_budget,'eos_token_id':tok.eos_token_id,
            'tokenizer_files':{p.name:filehash(p) for p in tokenizer_path.iterdir() if p.is_file() and p.suffix in ('.json','.jinja','.txt','.model')},
            'scope':'prompt length only, not generation/EOS/loss or SFT response qualification','truncation':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['parquet','tokenizer']);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tokenizer',type=Path);p.add_argument('--context',type=int,default=4096);p.add_argument('--response-budget',type=int,default=128)
    a=p.parse_args()
    try:
        if a.command=='parquet':result=parquet(a.data,a.out)
        else:
            require(not a.out.exists(),'new receipt required');require(a.tokenizer is not None,'local tokenizer required')
            result=tokenizer_check(a.data,a.tokenizer,a.context,a.response_budget)
            with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    except RuntimeError as e:p.exit(2,str(e)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
