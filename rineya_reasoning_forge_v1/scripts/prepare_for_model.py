#!/usr/bin/env python3
"""Measure actual model-token lengths and export fitting records WITHOUT truncation.
Requires transformers in the user's environment; the generator itself does not.
Uses a local tokenizer by default, never remote code, and never starts training.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
from verify import iter_jsonl

def measure_record(tokenizer: Any,row: dict[str,Any]) -> int:
    if [m.get('role') for m in row.get('messages',[])]!=['system','user','assistant']:
        raise ValueError('Expected exactly system/user/final-assistant messages')
    ids=tokenizer.apply_chat_template(row['messages'],tokenize=True,
             add_generation_prompt=False,enable_thinking=False)
    if not isinstance(ids,list) or not all(isinstance(x,int) for x in ids):
        raise ValueError('Tokenizer did not return a flat list of token IDs')
    if not ids:raise ValueError('Tokenizer produced no tokens')
    return len(ids)

def export(tokenizer: Any,source: Path,destination: Path,max_length: int,tokenizer_label: str) -> dict:
    if max_length<64:raise ValueError('max_length must be >= 64')
    if destination.exists():raise FileExistsError(f'Refusing to overwrite {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    excluded=destination.with_suffix('.excluded.jsonl');reportpath=destination.with_suffix('.length_audit.json')
    if excluded.exists() or reportpath.exists():raise FileExistsError('Audit destinations already exist')
    kept=[];rejected=[];seen=set()
    # Complete validation before creating the export; never silently drop a malformed row.
    for row in iter_jsonl(source):
        if row['id'] in seen:raise ValueError('duplicate input id')
        seen.add(row['id']);n=measure_record(tokenizer,row)
        (kept if n<=max_length else rejected).append((row,n))
    if not kept:raise ValueError('No records fit. Increase the limit only after checking model/hardware support.')
    with destination.open('w',encoding='utf-8') as f:
        for row,_ in kept:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with excluded.open('w',encoding='utf-8') as f:
        for row,n in rejected:f.write(json.dumps({'id':row['id'],'tokens':n,'reason':'exceeds_limit_no_truncation'},ensure_ascii=False)+'\n')
    report={'tokenizer':tokenizer_label,'max_sequence_length':max_length,'kept_rows':len(kept),
      'excluded_rows':len(rejected),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'output_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
      'rendering':{'add_generation_prompt':False,'enable_thinking':False},
      'token_counts':{row['id']:n for row,n in kept+rejected},
      'training_started':False,'labels_or_loss_masks_created':False,
      'required_trainer_loss_mask':'final assistant response and its end-of-turn token only',
      'warnings':['Verify the loaded tokenizer is the exact Rineya tokenizer, not merely a related Qwen tokenizer.',
        'Verify loss masks with a decoded batch. This script only filters lengths; it does not configure a trainer.',
        'Do not blindly truncate excluded records: necessary evidence may disappear.',
        'Recheck family/length/label balance after filtering.']}
    reportpath.write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenizer',required=True,help='Path to the exact local model tokenizer')
    p.add_argument('--max-length',type=int,default=2048);p.add_argument('--allow-network',action='store_true')
    a=p.parse_args()
    try:from transformers import AutoTokenizer
    except ImportError:raise SystemExit('transformers is required for this optional audit; run in your model environment.')
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=not a.allow_network,trust_remote_code=False)
    if not getattr(tok,'chat_template',None):raise SystemExit('Tokenizer has no chat template; cannot safely infer one.')
    result=export(tok,a.input,a.output,a.max_length,a.tokenizer)
    print(json.dumps({k:v for k,v in result.items() if k!='token_counts'},indent=2))
if __name__=='__main__':main()
