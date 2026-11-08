"""Fresh local tokenizer-only CPU contract replay; no model/weight execution.

All output/cache/temp paths are external to tracked source. Native receipts
store counts, not token sequences: token-ID/label hashes here identify the
freshly reconstructed input contract, not proof of captured runtime IDs.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import random
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=pathlib.Path,required=True)
    parser.add_argument('--audit-root',type=pathlib.Path,required=True)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args()
    audit=args.audit_root.resolve()
    if args.output.resolve().parent!=audit or '.codex' in audit.parts or audit.is_relative_to(args.package.resolve()):
        raise ValueError('Only output/cache paths under supplied external audit root are permitted')
    cache=audit/'tokenizer_cpu_cache'
    temp=audit/'tokenizer_cpu_tmp'
    cache.mkdir(exist_ok=True);temp.mkdir(exist_ok=True)
    env=dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='-1',USE_TORCH='0',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HUGGINGFACE_HUB_CACHE=str(cache/'hub'),TORCH_HOME=str(cache/'torch'),XDG_CACHE_HOME=str(cache),TEMP=str(temp),TMP=str(temp),PYTHONDONTWRITEBYTECODE='1')
    os.environ.update(env)
    from transformers import AutoTokenizer
    inputs,checks,failures={},collections.Counter(),[]

    def raw(rel):
        path=args.package/rel
        b=path.read_bytes();inputs[str(path.resolve())]=dict(sha256=sha(b),bytes=len(b));return b

    def read(rel):
        return json.loads(raw(rel))

    def check(condition,label,row=None):
        checks[label]+=1
        if not condition:failures.append(dict(check=label,row=row))

    config=read('configs/titan_transition.v2.memory_efficient.json')
    root=pathlib.Path('corpora/distant_evidence_v2')
    manifest=read(root/'manifest.json');design=read(root/'design.json')
    historical=read('evidence/qualification_20261003/corpus_v2_frozen_token_geometry.json')
    historical_rows={r['id']:r for r in historical['records']}
    source=pathlib.Path(config['source']['path'])
    check(source.resolve()==(args.package/'.native-source/titan-transition').resolve(),'locked_local_source_path')
    tokenizer_files=['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']
    tokenizer_hashes={}
    for name in tokenizer_files:
        b=(source/name).read_bytes();tokenizer_hashes[name]=sha(b)
        inputs[str((source/name).resolve())]=dict(sha256=sha(b),bytes=len(b))
        check(sha(b)==config['source']['files'][name],'tokenizer_source_inventory_hash',name)
        if name in historical['source_tokenizer_files_sha256']:
            check(sha(b)==historical['source_tokenizer_files_sha256'][name],'historical_tokenizer_identity',name)
    versions={k:importlib.metadata.version(k) for k in ['transformers','tokenizers']}
    check(versions==dict(transformers='4.51.3',tokenizers='0.21.4'),'pinned_tokenizer_versions')
    tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False,use_fast=True,cache_dir=str(cache/'hub'))
    check(tokenizer.is_fast and tokenizer.eos_token_id is not None,'fast_tokenizer_and_EOS')
    check('torch' not in sys.modules,'torch_backend_not_imported')
    rows,records,pairs={},[],collections.defaultdict(list)

    def overlap(offsets,prompt_start,chars):
        start,end=prompt_start+chars[0],prompt_start+chars[1]
        token_positions=[i for i,(left,right) in enumerate(offsets) if right>start and left<end]
        if not token_positions:raise ValueError('No tokens overlap requested raw span')
        return [token_positions[0],token_positions[-1]+1]

    for split in ['train','dev']:
        contents=raw(root/f'{split}.jsonl')
        check(sha(contents)==manifest['splits'][split]['sha256']==historical['corpus_files_sha256'][f'{split}.jsonl'],'corpus_current_and_historical_hash',split)
        check(manifest['splits'][split]['role']==split and manifest['splits'][split]['protected'] is False,'unprotected_explicit_split',split)
        rows[split]=[json.loads(line) for line in contents.decode().splitlines()]
        for row in rows[split]:
            id=row['id'];m=design['rows'][id];stored=m['token_geometry'];old=historical_rows[id]
            rendered=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            check(rendered.count(row['prompt'])==1,'single_raw_prompt_in_chat_template',id)
            content_start=rendered.index(row['prompt'])
            tokenized=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True)
            prompt_ids=tokenizer.encode(rendered,add_special_tokens=False)
            check(prompt_ids==tokenized['input_ids'],'encode_call_matches_offset_tokenization',id)
            target_ids=tokenizer.encode(row['target'],add_special_tokens=False)
            target_eos=target_ids+[tokenizer.eos_token_id]
            inputs_ids=prompt_ids+target_eos
            labels=[-100]*len(prompt_ids)+target_eos
            p,n,t=len(prompt_ids),len(target_eos),len(inputs_ids)
            check(p==stored['prompt_tokens']==old['prompt_tokens'],'fresh_prompt_length_matches_metadata_and_history',id)
            check(len(target_ids)==stored['target_tokens'] and n==old['answer_plus_eos_tokens'],'fresh_target_EOS_length_matches_metadata_history',id)
            check(t==stored['train_tokens']==old['training_input_tokens'],'fresh_training_input_length_matches_metadata_history',id)
            check(n<=32 and p<=2048 and (split!='train' or t<=512),'registered_token_caps',id)
            check(labels[-1]==tokenizer.eos_token_id and sum(x!=-100 for x in labels[1:])==n,'answer_only_shifted_label_EOS_contract',id)
            record=dict(id=id,split=split,role=m['role'],category=m['category'],prompt_tokens=p,answer_plus_eos_tokens=n,training_input_tokens=t,
                        prompt_token_ids_sha256=digest(prompt_ids),target_plus_EOS_token_ids_sha256=digest(target_eos),training_input_token_ids_sha256=digest(inputs_ids),training_labels_with_ignore_mask_sha256=digest(labels),rendered_prompt_sha256=sha(rendered.encode()))
            if m['role']!='retention':
                blocks=row['prompt'].split('\n---\n')
                check(len(blocks)==4,'raw_prompt_four_blocks',id)
                raw_block_spans={}
                for label,key in [('evidence','evidence_block_text'),('filler','filler_block_text'),('query','query_text')]:
                    text=m[key]
                    check(row['prompt'].count(text)==1,'unique_block_raw_text',id+':'+label)
                    start=row['prompt'].index(text);chars=[start,start+len(text)]
                    check(chars==m['block_spans'][label],'raw_derived_block_char_spans',id+':'+label)
                    raw_block_spans[label]=overlap(tokenized['offset_mapping'],content_start,chars)
                evidence_start=row['prompt'].index(m['evidence_block_text'])
                raw_record_spans=[];cursor=evidence_start
                for line in m['evidence_block_text'].splitlines():
                    chars=[cursor,cursor+len(line)]
                    raw_record_spans.append(overlap(tokenized['offset_mapping'],content_start,chars));cursor+=len(line)+1
                check(raw_block_spans==stored['block_token_spans'],'fresh_block_token_spans_match_metadata',id)
                check(raw_record_spans==stored['evidence_token_spans'],'fresh_record_token_spans_match_metadata',id)
                required=[raw_record_spans[s['event_index']] for s in m['evidence_spans'] if s['required']]
                gap=raw_block_spans['query'][0]-raw_block_spans['evidence'][1]
                earliest=p-min(a for a,b in required);latest=p-max(b for a,b in required)
                for key,value in [('evidence_to_query_gap_tokens',gap),('earliest_required_to_answer_tokens',earliest),('latest_required_to_answer_tokens',latest)]:
                    check(value==stored[key]==old[key],'fresh_distance_matches_metadata_history',id+':'+key)
                    record[key]=value
                check(record['rendered_prompt_sha256']==stored['rendered_prompt_sha256'],'fresh_rendered_chat_hash',id)
                check(earliest<=128 if m['role']=='near_control' else gap>128 and stored['registered_gap_band'][0]<=gap<=stored['registered_gap_band'][1],'fresh_near_distant_distance_contract',id)
                record.update(block_token_spans=raw_block_spans,evidence_record_token_spans=raw_record_spans)
                pairs[(split,m['position_pair_id'])].append(record)
            records.append(record)
    for key,pair in pairs.items():
        check(len(pair)==2 and {p['role'] for p in pair}=={'near_control','distant_heldout'},'position_pair_roles',str(key))
        a,b=pair
        check(a['prompt_tokens']==b['prompt_tokens'] and a['target_plus_EOS_token_ids_sha256']==b['target_plus_EOS_token_ids_sha256'] and a['training_input_tokens']==b['training_input_tokens'],'fresh_position_pair_lengths_and_target_ids',str(key))
    by_id={r['id']:r for r in records}
    run=pathlib.Path('runs/titan_transition_v2_20261004_memory_efficient')
    order=list(range(256));random.Random(config['seed']).shuffle(order)
    schedule=[rows['train'][order[i%256]]['id'] for i in range(config['limits']['steps'])]
    check(schedule==historical['training_exposure']['ordered_row_ids'],'independent_schedule_matches_historical')
    native={}
    for arm in 'ABCD':
        receipt=read(run/f'{arm}-evaluation/receipt.json')
        check([p['id'] for p in receipt['predictions']]==[r['id'] for r in rows['dev']],'native_DEV_row_order',arm)
        for p in receipt['predictions']:
            check(p['prompt_tokens']==by_id[p['id']]['prompt_tokens'],'native_DEV_prompt_count_matches_fresh_replay',arm+':'+p['id'])
        if arm=='A':continue
        for stage in ['distill','recover']:
            receipt=read(run/f'{arm}-{stage}/receipt.json')
            check([r['row_id'] for r in receipt['records']]==schedule and [r['step'] for r in receipt['records']]==list(range(512)),'native_TRAIN_order_step_matches_fresh_schedule',arm+stage)
            for r in receipt['records']:
                fresh=by_id[r['row_id']]
                supervised=fresh['training_input_tokens'] if stage=='distill' else fresh['answer_plus_eos_tokens']
                check(r['input_tokens']==fresh['training_input_tokens'] and r['supervised_positions']==supervised,'native_TRAIN_input_label_counts_match_fresh_tokenizer',arm+stage+':'+str(r['step']))
            native[arm+'_'+stage]=dict(steps=512,total_input_tokens=sum(by_id[id]['training_input_tokens'] for id in schedule),total_supervised_positions=sum(by_id[id]['training_input_tokens' if stage=='distill' else 'answer_plus_eos_tokens'] for id in schedule),ordered_input_contract_hash=digest([(id,by_id[id]['training_input_token_ids_sha256'],by_id[id]['training_labels_with_ignore_mask_sha256']) for id in schedule]))
    summary={}
    for split in ['train','dev']:
        subset=[r for r in records if r['split']==split]
        distant=[r for r in subset if r['role']=='distant_heldout'];near=[r for r in subset if r['role']=='near_control']
        summary[split]=dict(rows=len(subset),prompt_tokens_min=min(r['prompt_tokens'] for r in subset),prompt_tokens_max=max(r['prompt_tokens'] for r in subset),training_input_tokens_min=min(r['training_input_tokens'] for r in subset),training_input_tokens_max=max(r['training_input_tokens'] for r in subset),answer_plus_EOS_tokens_max=max(r['answer_plus_eos_tokens'] for r in subset),distant_gap_tokens_min=min(r['evidence_to_query_gap_tokens'] for r in distant),distant_gap_tokens_max=max(r['evidence_to_query_gap_tokens'] for r in distant),near_earliest_required_to_answer_tokens_max=max(r['earliest_required_to_answer_tokens'] for r in near))
    check(summary['train']['distant_gap_tokens_min']==140 and summary['train']['distant_gap_tokens_max']==157,'fresh_TRAIN_distance_range')
    check(summary['dev']['distant_gap_tokens_min']==145 and summary['dev']['distant_gap_tokens_max']==1058,'fresh_DEV_distance_range')
    check(summary['train']['near_earliest_required_to_answer_tokens_max']==summary['dev']['near_earliest_required_to_answer_tokens_max']==81,'fresh_near_required_max')
    check('torch' not in sys.modules,'no_torch_import_after_replay')
    final_pairing=audit/'pairing_audit_final.json';b=final_pairing.read_bytes();inputs[str(final_pairing)]=dict(sha256=sha(b),bytes=len(b))
    check(json.loads(b)['status']=='PASS','separate_raw_oracle_required_record_audit_pass')
    receipt=dict(schema='SBHM-001-fresh-offline-tokenizer-contract-replay/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS' if not failures else 'FAIL',assertions=sum(checks.values()),checks=dict(checks),failures=failures,
                 method='Pinned local AutoTokenizer only; raw prompt/block/record characters mapped to freshly produced offsets; independent chat-template/answer-EOS encoding and schedule; no sbhm import, model class, weight load, tensor creation or GPU execution',input_hashes=inputs,source_tokenizer_hashes=tokenizer_hashes,versions=versions,tokenizer_class=type(tokenizer).__name__,eos_token_id=tokenizer.eos_token_id,environment_controls=env,interpreter=sys.executable,script_sha256=sha(pathlib.Path(__file__).read_bytes()),
                 torch_imported='torch' in sys.modules,weights_loaded=False,native_model_executed=False,summary=summary,position_pairs_checked=len(pairs),native_trace_contract_crosschecks=native,records=records,
                 limitations=['Native receipts did not retain actual token IDs or label IDs. Fresh hashes verify the immutable input/tokenizer/encoding contract and receipt counts/order, not captured runtime sequences.','Required-evidence flags separately reconstructed from raw queries by finalized stdlib pairing audit; this tokenizer replay verifies their freshly tokenized positions.','This is offline input-contract verification, not Rineya capability evidence, new training, new native evaluation or confirmation evidence.'])
    with args.output.open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=receipt['status'],assertions=receipt['assertions'],failures=len(failures),summary=summary,output=str(args.output),torch_imported=receipt['torch_imported'])))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
