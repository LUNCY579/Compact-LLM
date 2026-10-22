"""Tokenizer-only readiness audit; DEV policy inputs and gold are separate files."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import sys

BASE=Path('C:/Users/clclc/BusinessProject')
DATA=BASE/'rineya-dataset-retention-records/binding-signal-prep-20260920/candidate-v1'
OUT=DATA/'compiled-v1'
PILOT=BASE/'ChatBubble-retention-pilot-20260917'
SOURCE=BASE/'Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import generate
    sys.path.insert(0,str(PILOT))
    from experiments.reasoning_posttraining_v1.serialization import compile_row
    from experiments.model_quality_gpu_v1.data import group_updates
    from transformers import AutoTokenizer
    if OUT.exists():
        raise FileExistsError('Token compilation outputs are create-only')
    data={split:[json.loads(line) for line in (DATA/(split.upper()+'.jsonl')).read_text(encoding='utf-8').splitlines()]
          for split in ('train','dev')}
    audit=generate.audit(data)
    input_hashes={name:sha(DATA/name) for name in ('TRAIN.jsonl','DEV.jsonl','BUILD.json')}
    expected=json.loads((BASE/'rineya-dataset-retention-records/doseint001-20260919/BUILD_GATE_DUAL899.json').read_text(encoding='utf-8'))['source_assets_sha256']
    names=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')
    assets={name:sha(SOURCE/name) for name in names}
    if any(assets[name]!=expected[name] for name in names):
        raise ValueError('Tokenizer/source sidecar identity differs')
    tokenizer=AutoTokenizer.from_pretrained(str(SOURCE),local_files_only=True,trust_remote_code=False)
    OUT.mkdir()
    totals={}
    for split,rows in data.items():
        lengths=[];prompt_lengths=[];supervised=[];final_lengths=[];compiled=[];categories=Counter()
        train_file=OUT/'TRAIN_COMPILED.jsonl'
        inputs_file=OUT/'DEV_INPUTS.jsonl'
        gold_file=OUT/'DEV_GOLD.jsonl'
        with (train_file if split=='train' else inputs_file).open('x',encoding='utf-8',newline='\n') as f:
            gold=gold_file.open('x',encoding='utf-8',newline='\n') if split=='dev' else None
            try:
                for row in rows:
                    if [m['role'] for m in row['messages']]!=['system','user','assistant']:
                        raise ValueError('Invalid native message roles')
                    prompt=row['messages'][:2]
                    if row['messages'][2]['content']!=row['reference']:
                        raise ValueError('Assistant target differs')
                    value=compile_row(prompt,row['reference'],tokenizer,max_sequence=2048,completion_reserve=128)
                    if value['labels'][0]!=-100 or value['truncated']:
                        raise ValueError('Truncated or wrong shifted loss mask')
                    n=sum(v!=-100 for v in value['labels'][1:])
                    if n!=value['supervised_tokens']:
                        raise ValueError('Shifted supervised accounting differs')
                    lengths.append(len(value['ids']));prompt_lengths.append(len(value['prompt_ids']))
                    supervised.append(n);final_lengths.append(value['final_content_tokens']);categories[row['category']]+=1
                    if split=='train':
                        output={'id':row['id'],'parent_id':row['parent_id'],'source_split':'train',
                                'family':row['family'],'category':row['category'],'prompt':prompt,
                                'reference':row['reference'],**value}
                        compiled.append(output)
                    else:
                        # No world, query-kind/category, reference, assistant target or labels in policy input.
                        output={'id':row['id'],'parent_id':row['parent_id'],'prompt':prompt,
                                'prompt_ids':value['prompt_ids'],'max_sequence':2048}
                        gold.write(json.dumps({'id':row['id'],'parent_id':row['parent_id'],
                            'category':row['category'],'reference':row['reference']},sort_keys=True)+'\n')
                    f.write(json.dumps(output,sort_keys=True,ensure_ascii=False)+'\n')
            finally:
                if gold is not None:gold.close()
        totals[split]={'rows':len(rows),'worlds':len({r['parent_id'] for r in rows}),
            'categories':dict(categories),'prompt_tokens':sum(prompt_lengths),
            'teacher_forced_tokens':sum(lengths),'supervised_response_plus_eos_tokens':sum(supervised),
            'max_prompt_tokens':max(prompt_lengths),'max_full_tokens':max(lengths),
            'max_answer_content_tokens':max(final_lengths),'truncated_rows':0}
        if split=='train':
            totals[split]['native_2048_token_update_groups']=len(group_updates(compiled,2048))
    if any(sha(SOURCE/name)!=value for name,value in assets.items()) or any(sha(DATA/name)!=value for name,value in input_hashes.items()):
        raise ValueError('Inputs changed during token compilation')
    report={'schema':1,'status':'TOKENIZER_COMPILED_NOT_TRAINING_AUTHORIZATION','source':str(SOURCE),
        'source_sidecars_sha256':assets,'input_sha256':input_hashes,
        'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
        'native_serialization_sha256':sha(PILOT/'experiments/reasoning_posttraining_v1/serialization.py'),
        'generator_sha256':sha(Path(generate.__file__)),'compiler_sha256':sha(Path(__file__)),
        'totals':totals,'build_audit':audit,'model_weights_loaded':False,'model_forward_calls':0,
        'gpu_visible':False,'dev_gold_separate_from_policy_inputs':True,
        'human_data_review':'NOT_PERFORMED','external_exclusion_audit':'PENDING',
        'training_permission':False,'quality_claim':False,'release_permitted':False}
    write_json(OUT/'TOKEN_AUDIT.json',report)
    print(json.dumps({'status':report['status'],'totals':totals},indent=2))

if __name__=='__main__':
    main()
