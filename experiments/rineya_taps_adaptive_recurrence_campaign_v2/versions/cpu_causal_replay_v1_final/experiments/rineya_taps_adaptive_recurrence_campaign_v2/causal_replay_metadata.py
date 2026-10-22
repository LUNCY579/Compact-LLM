"""Metadata-only DEV causal target binding; no model/Torch/GPU/network imports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil

from . import bulk_bank

PACKAGE=Path(__file__).resolve().parent
EXPORT=PACKAGE/'runs/causal-replay-DEV-export-001'
BANK_SHA='566e6d6e64f258ff4deaa28a9909841a451fb96cf939ca44533726c59b2a7ff0'
TOKEN_SHA='6529b48b26264123fafc9f81d67cb6cc5a957fe30a0ab4853440790c56bfdd53'
SCORER_SHA='3242439231064fc81991e157c0da62833f4e4223ec71d60cf9ffd2851815b5eb'
NORMALIZER_SHA='25db8ad7b8bd4166188eccab4be73c9a430f696610dea5b8c83b380883f9d42a'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def digest(values):
    return hashlib.sha256(json.dumps(values,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def run(source,cache,out):
    source,cache,out=Path(source).resolve(strict=True),Path(cache).resolve(),Path(out).resolve()
    if PACKAGE not in cache.parents or PACKAGE not in out.parents:raise PermissionError('metadata/cache must remain campaign-local')
    if out.exists() or cache.exists():raise FileExistsError('metadata output/cache must be exclusive')
    binding_path=PACKAGE/'runs/binding-001/MODEL_DATA_BINDING.json'
    binding=json.loads(binding_path.read_text(encoding='utf-8'))
    if source!=Path(binding['source_path']).resolve():raise PermissionError('exact canonical tokenizer source required')
    paths={name:EXPORT/name for name in ('DEVELOPMENT_BANK.json','DEVELOPMENT_TOKEN_BINDING.json','MANIFEST.json')}
    if sha(paths['DEVELOPMENT_BANK.json'])!=BANK_SHA or sha(paths['DEVELOPMENT_TOKEN_BINDING.json'])!=TOKEN_SHA:raise RuntimeError('frozen DEV exports differ')
    if sha(PACKAGE/'bulk_bank.py')!=SCORER_SHA or sha(PACKAGE/'open_bank.py')!=NORMALIZER_SHA:raise RuntimeError('frozen scorer/normalizer differs')
    bank=json.loads(paths['DEVELOPMENT_BANK.json'].read_text(encoding='utf-8'))
    tokens=json.loads(paths['DEVELOPMENT_TOKEN_BINDING.json'].read_text(encoding='utf-8'))
    manifest=json.loads(paths['MANIFEST.json'].read_text(encoding='utf-8'))
    if len(bank)!=192 or len(tokens)!=192 or manifest.get('validation_rows_exported')!=0 or any(x['split']!='DEVELOPMENT' for x in bank+tokens):raise PermissionError('only the complete192 DEV rows are permitted')
    token_map={x['case_id']:x for x in tokens}
    if len(token_map)!=192 or set(token_map)!={x['id'] for x in bank}:raise RuntimeError('DEV ID coverage differs')
    for name in ('tokenizer.json','tokenizer_config.json','generation_config.json'):
        if sha(source/name)!=binding['checkpoint_files'][name]:raise RuntimeError('canonical tokenizer metadata differs:'+name)
    cache.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/'tokenizer.json',cache/'tokenizer.json')
    if sha(cache/'tokenizer.json')!=binding['checkpoint_files']['tokenizer.json']:raise RuntimeError('local tokenizer cache copy differs')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(cache/'tokenizer.json'))
    eos=json.loads((source/'generation_config.json').read_text())['eos_token_id']
    eos=eos if isinstance(eos,list) else [eos]
    if eos!=[151645,151643]:raise RuntimeError('canonical EOS set differs')
    rows=[];counts=Counter();boundary_changes=[];normalization_variants=[]
    for case in bank:
        bound=token_map[case['id']];prompt_ids=bound['input_token_ids']
        observed=tokenizer.encode(bound['rendered_prompt'],add_special_tokens=False).ids
        if observed!=prompt_ids or len(prompt_ids)!=bound['input_tokens'] or bound['raw_prompt']!=case['prompt']:raise RuntimeError('verified rendered DEV tokenizer IDs differ')
        answer=case['gold'];grade=bulk_bank.score_response(answer,case)
        if not grade['correct'] or not grade['format_valid'] or bulk_bank.normalize_output(answer)!=answer:raise RuntimeError('canonical answer is not strict-normalized correct')
        target_ids=tokenizer.encode(answer,add_special_tokens=False).ids
        if not target_ids or tokenizer.decode(target_ids,skip_special_tokens=False)!=answer:raise RuntimeError('canonical standalone answer token roundtrip differs')
        if any(type(value) is not int or not 0<=value<151936 for value in prompt_ids+target_ids):raise RuntimeError('embedding row bound differs')
        joint=tokenizer.encode(bound['rendered_prompt']+answer,add_special_tokens=False).ids
        boundary_same=joint==prompt_ids+target_ids
        if not boundary_same:boundary_changes.append(case['id'])
        prefixes=[]
        for index,target in enumerate(target_ids):
            prefix=prompt_ids+target_ids[:index]
            prefixes.append({'answer_position':index,'target_token_id':target,'prefix_token_ids':prefix,
                'prefix_tokens':len(prefix),'prefix_sha256':digest(prefix),'future_target_in_prefix':False,
                'preceding_gold_tokens':index,'teacher_conditioned':index>0,
                'per_forward_feature_use':'original prompt-only first-position features frozen; later moments diagnostic-only'})
        eos_prefix=prompt_ids+target_ids
        variants=[answer.lower(),answer.capitalize(),' '+answer+' ']
        normalization_variants.append({'case_id':case['id'],'accepted_equivalent_variants':[value for value in variants if value!=answer and bulk_bank.score_response(value,case)['correct']],
            'canonical_variant':answer,'canonical_only_loss_limitation':True})
        rows.append({'case_id':case['id'],'parent_id':case['parent_id'],'category':case['category'],'difficulty_stratum':case['difficulty_stratum'],
            'answer_kind':case['answer_kind'],'answerability':case['answerability'],'source_status':case['source_status'],'input_tokens':len(prompt_ids),
            'original_prompt_ids_sha256':digest(prompt_ids),'original_prompt_feature_prefix_sha256':digest(prompt_ids),
            'canonical_answer':answer,'canonical_answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),'canonical_answer_token_ids':target_ids,
            'canonical_answer_positions':len(target_ids),'answer_targets':prefixes,'joint_raw_tokenization_same_as_generation_continuation':boundary_same,
            'EOS':{'separate_from_answer_loss':True,'prefix_token_ids':eos_prefix,'prefix_tokens':len(eos_prefix),'prefix_sha256':digest(eos_prefix),
                   'primary_chat_EOT_target':151645,'generation_accepted_EOS_set':eos,'single_EOT_loss_not_equivalent_to_EOS_set_loss':True}})
        counts[len(target_ids)]+=1
    answer_positions=sum(x['canonical_answer_positions'] for x in rows);eos_positions=192
    maximum_answer=max(item['prefix_tokens'] for row in rows for item in row['answer_targets']);maximum_eos=max(row['EOS']['prefix_tokens'] for row in rows)
    value=.9060052999993786;adjoint=1.3799382999968657;factor=1.5
    # Prospective fresh qual: existing26 H3 APIs + late teacher-prefix3-call FD.
    qualification_API=29;qualification_adjoints=3
    forecast={}
    for label,positions in (('answer_only',answer_positions),('answer_plus_separate_EOT',answer_positions+eos_positions)):
        task_seconds=positions*(adjoint+2*value)*factor;qual_seconds=qualification_API*adjoint*factor
        total=task_seconds+qual_seconds+120+180+1800
        forecast[label]={'target_positions':positions,'task_full_forward_attempts':3*positions,'task_state_adjoints':positions,
            'qualification_API_attempts':qualification_API,'qualification_state_adjoints':qualification_adjoints,
            'total_API_attempts':3*positions+qualification_API,'total_state_adjoints':positions+qualification_adjoints,
            'forecast_task_seconds':task_seconds,'forecast_qualification_seconds':qual_seconds,'setup_hash_reserve_seconds':120,
            'internal_evidence_reserve_seconds':180,'campaign_finalization_reserve_seconds':1800,
            'total_reserved_seconds':total,'remaining_native_seconds':20225.563,'fits_remaining_budget':total<=20225.563,
            'timing_source':'CPU H3 max256 only; new native fresh costs still required','factor':factor}
    result={'status':'FEASIBLE_METADATA_ONLY_PENDING_NEW_QUALIFICATION' if maximum_eos<=256 and all(x['fits_remaining_budget'] for x in forecast.values()) else 'NOT_FEASIBLE_COMPLETE_REGISTERED_SHAPE_OR_BUDGET',
        'native_model_loads':0,'native_forward_calls':0,'GPU_API_calls':0,'network_API_calls':0,'downloads':0,'environment_mutations':0,
        'created_utc':datetime.now(timezone.utc).isoformat(),'population_cases':192,'parent_worlds':96,'split':'DEVELOPMENT','validation_rows_read':0,'TEST_access':'NONE',
        'canonical_answer_positions':answer_positions,'separate_EOS_positions':eos_positions,'canonical_answer_length_distribution':dict(counts),
        'maximum_answer_target_prefix_tokens':maximum_answer,'maximum_separate_EOS_prefix_tokens':maximum_eos,'shape_limit':256,
        'maximum_canonical_answer_tokens':max(counts),'complete_answer_processed_prefix_positions_per_alpha':sum(item['prefix_tokens'] for row in rows for item in row['answer_targets']),
        'separate_EOS_processed_prefix_positions_per_alpha':sum(row['EOS']['prefix_tokens'] for row in rows),
        'forecast':forecast,'canonical_surface_limitations':normalization_variants,'joint_token_boundary_change_case_ids':boundary_changes,
        'tokenizer_class':'tokenizers.Tokenizer from exact canonical fast-tokenizer JSON;192 rendered-prefix identities verified against source-bound HF binding002',
        'tokenizers_version':importlib.metadata.version('tokenizers'),'python':platform.python_version(),'tokenizer_cache':str(cache),
        'first_prefix_feature_firewall':'original rendered prompt only; freeze before gold continuation; never use teacher-prefix features or labels to fit online alpha/A5 pool',
        'new_qualification_required':['new DEV/scorer/normalizer/canonical sequence/source binding','CPU same model/head/operator/device/guards and fresh max256 state tape/FD/RMS','exact causal prefix target exclusion including longest late-answer prefix','EOS objective explicitly separate; no single-token termination inference','fresh complete native timing before all192 cases; no narrowing or survivor selection'],
        'rows':rows,'input_sha256':{str(path):sha(path) for path in list(paths.values())+[binding_path,source/'tokenizer.json',source/'tokenizer_config.json',source/'generation_config.json',PACKAGE/'bulk_bank.py',PACKAGE/'open_bank.py',Path(__file__)]}}
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','population_cases','canonical_answer_positions','separate_EOS_positions','canonical_answer_length_distribution','maximum_answer_target_prefix_tokens','maximum_separate_EOS_prefix_tokens','forecast')},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.cache,args.out)
