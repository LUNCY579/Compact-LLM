from copy import deepcopy
import struct
import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_bank
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_analysis import (
    GRID, GIB, aggregate, expected_cells, expected_order, payload_digest, validate_registration, validate_source_manifest,
)


def fixture():
    bank = bulk_bank.generate("analysis-regression", 17); raw_by_id={raw["id"]:raw for raw in bank}
    smoke=[{"case_id":f"smoke-{i}","input_tokens":3,"scorer":{"type":"exact","answers":["YES"]},"format_scorer":{"type":"exact","answers":["YES"]}} for i in range(8)]
    tokens=[{"case_id":raw["id"],"input_tokens":3} for raw in bank]
    rank={"SMOKE":0,"DEVELOPMENT":1,"OPEN_VALIDATION":2}; rows=[]
    def ordering(cell):
        split,seed,identifier,label,extra=cell
        raw=raw_by_id.get(identifier)
        return rank[split], int(label=='A5'), seed, identifier, expected_order(raw,bank).index(label) if raw and label!='A5' else label, extra
    for split,seed,case_id,label,extra in sorted(expected_cells(bank,smoke),key=ordering):
        arm="A2" if label.startswith("A2:") else label
        alpha=next(value for value in GRID if f"{value:g}"==label.split(":")[1]) if arm=="A2" else None
        raw=raw_by_id.get(case_id); response=raw["gold"] if raw else "YES"
        row={"split":split,"seed":seed,"configured_seed":seed,"case_id":case_id,"arm":arm,"fixed_alpha":alpha,"extra_loops":extra,
            "physical_run":True,"response":response,"token_ids":[1],"score":1,"format_score":1,"safety_stopped":False,"gold_nll":None,"gold_answer":None,
            "invalid":False,"termination":"eos","wall_seconds":2.,"full_model_calls_including_gold_nll":1,"initial_block_applications":1,"processed_token_positions":3,"total_token_positions_including_gold_nll":3,"extra_layer_applications":2*extra}
        if raw:
            row.update({field:raw[field] for field in ("parent_id","category","generator_family","generator_variant","variant_id","difficulty_stratum","pair_kind")})
            if arm!="A5":
                order=expected_order(raw,bank);row.update(deterministic_policy_order=order,policy_order_index=order.index(label))
        rows.append(row)
    physical=list(rows)
    traces=[{"split":row["split"],"seed":row["seed"],"configured_seed":row["seed"],"case_id":row["case_id"],"arm":row["arm"],"extra_loops":row["extra_loops"],
        "forward_index":1,"iteration":iteration,"alpha":struct.unpack("f",struct.pack("f",row["fixed_alpha"]))[0] if row["arm"]=="A2" and iteration else 1.,
        **({"random_draw_index":iteration} if row["arm"]=="A5" else {})} for row in physical if row["arm"]!="qualification_legacy" for iteration in range(row["extra_loops"]+1)]
    freeze={"status":"FROZEN_BULK_DEVELOPMENT","alpha":.8,"pool":[1.]*192,"deterministic_seed":17,"A5_actual_seeds":[17,29],
        "selection_case_ids":[raw["id"] for raw in bank if raw["split"]=="DEVELOPMENT"],"selection_parent_ids":sorted({raw["parent_id"] for raw in bank if raw["split"]=="DEVELOPMENT"}),
        "OPEN_VALIDATION_native_exposure_at_freeze":False};freeze["freeze_sha256"]=payload_digest(freeze)
    for raw in bank:
        original=next(row for row in physical if row["case_id"]==raw["id"] and row["arm"]=="A2" and row["fixed_alpha"]==.8)
        rows.append({**original,"arm":"A4_example","physical_run":False,"wall_seconds":9999.})
        if raw["split"]=="OPEN_VALIDATION":rows.append({**original,"arm":"A2_frozen","physical_run":False,"wall_seconds":9999.})
    forwards=[{"event":event,"case_id":row["case_id"],"split":row["split"],"seed":row["seed"],"arm":row["arm"],"extra_loops":row["extra_loops"],
        "full_model_call":1,"prefix_tokens":3,"forward_seconds":.1 if event=="FORWARD_COMPLETED" else None} for row in physical for event in ("FORWARD_STARTED","FORWARD_COMPLETED")]
    for index in range(18):
        forwards.extend([{"event":"MAXIMUM_FORWARD_STARTED","attempt":index+1,"prefix_tokens":357},
            {"event":"MAXIMUM_FORWARD_FAILED" if index==6 else "MAXIMUM_FORWARD_COMPLETED","attempt":index+1,"prefix_tokens":357,"error":"EXPECTED_BULK_FORCED_EXCEPTION" if index==6 else None}])
        forwards[-1].update(completed=index!=6, partial_compute='UNKNOWN' if index==6 else None)
    for use_cache,cache_object in ((True,False),(False,True)):
        forwards.extend([{'event':'CACHE_REJECTION_STARTED','use_cache':use_cache,'cache_object_present':cache_object},
            {"event":"CACHE_REJECTED","completed":False,"model_compute_started":False,"wrapper_forward_count":0}])
    native={"status":"BULK_MATRIX_COMPLETE_NOT_CONFIRMATORY","source_unchanged":True,"checkpoint_files_unchanged":True,"trainable_parameters":0,"training":False}
    worker={"status":"WORKER_TERMINAL","exit_code":0,"native_worker_seconds":50.,"peak_rss_bytes":1000}
    qualification={"status":"PASS_BULK_NEW_CAP_SAME_OPERATOR","maximum_prefix_tokens":357,"noextra_unit_originalA3_identity":True,"forced_exception_restoration":True,
        "cache_rejections":2,"source_unchanged":True,"before_state_sha256":"state","after_state_sha256":"state",
        "resources":{"rss_bytes":GIB,"host_available_bytes":2*GIB,"gpu_peak_allocated_bytes":3*GIB,"gpu_free_bytes":2*GIB}}
    a5={"status":"PASS_FROZEN_BULK_DEV_POOL","actual_seed_binding":True}
    return [rows,traces,bank,smoke,tokens,freeze,native,worker,qualification,a5,forwards]


def test_complete3296_once_policy_costs_and_primary48_parent_holdout():
    report=aggregate(*fixture())
    assert report["complete"] and report["physical_rows"]==report["expected_physical_rows"]==3296
    assert report["oracle_nonphysical_rows"]==288 and report["frozen_alias_nonphysical_rows"]==96
    assert report["costs"]["physical_only"]["wall_seconds"]["complete_value"]==6592
    assert report["primary_validation_parent_units"]==48 and report["reports"]["OPEN_VALIDATION"]["parent_units"]==48
    assert report["reports"]["OPEN_VALIDATION"]["constant_oracle"]["capture"] is None
    assert report['reports']['OPEN_VALIDATION']['own_bank_blind_floors']['overall']['constant_UNKNOWN']['cases']==96
    assert report['practical_margin']==.05
    assert all(value['practical_margin']==.05 and value['meets_margin'] is False for value in report['reports']['OPEN_VALIDATION']['contrasts'].values())


def test_duplicate_deterministic_seed_is_not_extra_evidence_and_incomplete_is_null():
    args=fixture(); row=next(row for row in args[0] if row["physical_run"] and row["arm"]=="A3" and row["split"]=="DEVELOPMENT")
    args[0].append({**row,"seed":29,"configured_seed":29})
    result=aggregate(*args)
    assert not result["complete"] and not result["reports"]["OPEN_VALIDATION"]["contrasts"]
    assert all(row["accuracy"] is None for row in result["reports"]["OPEN_VALIDATION"]["tables"])


def test_evidence_transfer_and_same_gold_pairs_are_distinct():
    result=aggregate(*fixture()); pairs=result["reports"]["OPEN_VALIDATION"]["parent_variant_pairs"]["A3"]
    swaps=[pair for pair in pairs if pair["pair_kind"]=="EVIDENCE_TRANSFER"]
    same=[pair for pair in pairs if pair["pair_kind"]=="SAME_GOLD_PROMPT_VARIANTS"]
    assert len(swaps)==8 and len(same)==40
    assert all(pair["appropriate_evidence_response"] and pair["same_gold_response_consistency"] is None for pair in swaps)
    assert all(pair["same_gold_response_consistency"] and pair["appropriate_evidence_response"] is None for pair in same)


def test_validation_freeze_and_counterbalance_are_immutable():
    args=fixture();args[5]["alpha"]=.9
    assert not aggregate(*args)["complete"]
    args=fixture();row=next(row for row in args[0] if row["physical_run"] and row["arm"]=="A0" and row["split"]=="DEVELOPMENT")
    row["policy_order_index"]=(row["policy_order_index"]+1)%9
    assert not aggregate(*args)["complete"]


@pytest.mark.parametrize('defect', ['rounded_static','scorer','smoke_format','unit_trace','prefix','forward_identity','max_duplicate','cache_variant','qualification','RSS','VRAM','host_reserve','resources_missing'])
def test_recorded_contract_violations_refuse_primary(defect):
    args=fixture()
    row=next(row for row in args[0] if row['physical_run'] and row['split']=='DEVELOPMENT' and row['arm']=='A2' and row['fixed_alpha']==1.1473425965417514)
    if defect=='rounded_static': row['fixed_alpha']=1.14734
    elif defect=='scorer': row['response']='UNKNOWN'; row['score']=1
    elif defect=='smoke_format': next(row for row in args[0] if row['split']=='SMOKE')['format_score']=.5
    elif defect=='unit_trace': next(e for e in args[1] if e['arm']=='A1' and e['iteration']==1)['alpha']=.9
    elif defect=='prefix': args[10][1]['prefix_tokens']=4
    elif defect=='forward_identity': args[10][1]['case_id']='other'
    elif defect=='max_duplicate': next(e for e in args[10] if e['event']=='MAXIMUM_FORWARD_COMPLETED')['attempt']=2
    elif defect=='cache_variant': next(e for e in args[10] if e['event']=='CACHE_REJECTION_STARTED')['use_cache']=False
    elif defect=='qualification': args[8].clear()
    elif defect=='RSS': args[7]['peak_rss_bytes']=6*GIB+1
    elif defect=='VRAM': args[8]['resources']['gpu_peak_allocated_bytes']=5*GIB+1
    elif defect=='host_reserve': args[8]['resources']['host_available_bytes']=GIB-1
    elif defect=='resources_missing': args[8].pop('resources')
    report=aggregate(*args)
    assert not report['complete'] and report['errors']
    assert report['reports']['OPEN_VALIDATION']['frozen_fixed_primary']['accuracy'] is None
    assert not report['reports']['OPEN_VALIDATION']['contrasts']


def test_stochastic_A5_prefix_stream_is_reconstructed_not_forced_deterministic():
    import hashlib, json, random
    args=fixture()
    dev_events=[e for e in args[1] if e['split']=='DEVELOPMENT' and e['arm']=='A3' and e['iteration']==1]
    for index,e in enumerate(dev_events): e['alpha']=.9 if index%2 else 1.1
    pool=[e['alpha'] for e in dev_events]; args[5]['pool']=pool
    args[5]['freeze_sha256']=payload_digest({k:v for k,v in args[5].items() if k!='freeze_sha256'})
    for e in args[1]:
        if e['arm']=='A5' and e['iteration']==1:
            bound=json.dumps([e['seed'],'A5',e['case_id']],ensure_ascii=False,separators=(',',':'))
            rng=random.Random(int.from_bytes(hashlib.sha256(bound.encode()).digest(),'big'))
            e['alpha']=struct.unpack('f',struct.pack('f',rng.choice(pool)))[0]
    assert aggregate(*args)['complete']
    next(e for e in args[1] if e['arm']=='A5' and e['iteration']==1)['alpha']=1.
    assert not aggregate(*args)['complete']


def test_pure_registration_runtime_config_and_resource_binding():
    runtime={'python':'3.11.15','torch':'2.12.1+cu126','transformers':'5.14.1','qwen_sha256':'qwen'}
    assets={'model.safetensors':'asset'}; old={'max_new_tokens':24}; taps={'rho':.275}
    admission={'bulk_bank_sha256':'bank','bulk_token_binding_sha256':'token','bulk_source_sha256':'scorer','bulk_normalizer_source_sha256':'normalizer',
        'source':'canonical','checkpoint_files':assets,'device':'cuda:0','dtype':'float32','attn_implementation':'eager','training':False,'native_test':False,
        'max_new_tokens':8,'max_bulk_input_tokens':350,'gpu_allocated_ceiling_bytes':5*GIB,'gpu_global_free_reserve_bytes':GIB,
        'gpu_initial_free_floor_bytes':6*GIB,'worker_host_available_floor_bytes':6*GIB,'worker_rss_ceiling_bytes':6*GIB}
    token={'bank_sha256':'bank','generator_scorer_sha256':'scorer','normalizer_source_sha256':'normalizer','runtime':runtime,
        'status':'BULK_TOKENIZER_BOUND_NATIVE_NOT_MEASURED','native_model_loads':0,'canonical_source':'canonical','checkpoint_files':assets}
    binding={'runtime':runtime,'source_path':'canonical','checkpoint_files':assets}
    expected={**old,'device':'cuda:0','dtype':'float32','max_rss_gib':6}
    registration={'bank_sha256':'bank','token_binding_sha256':'token','scorer_sha256':'scorer','old_smoke_config':expected,
        'config':{**expected,'max_new_tokens':8},'taps':taps,'native_layers':[12,13],'extra_loops':[0,1],
        'gold_nll_status':'NOT_COLLECTED_CAUSAL_GUARD','runtime':{**runtime,'tf32':False,'CUBLAS_WORKSPACE_CONFIG':':4096:8'},
        'count_forecast':{'physical_evaluations':3296,'generation_call_cap':28416,'maximum_shape_forward_attempts':18,'cache_rejection_attempts':2,'model_API_attempt_cap':28436}}
    args=[registration,admission,binding,token,{'config':taps},old,taps]
    assert validate_registration(*args)==[]
    for mutate in ('runtime','count','cap','assets','reserve'):
        modified=deepcopy(args)
        if mutate=='runtime': modified[0]['runtime']['torch']='other'
        elif mutate=='count': modified[0]['count_forecast']['physical_evaluations']=3295
        elif mutate=='cap': modified[0]['config']['max_new_tokens']=7
        elif mutate=='assets': modified[3]['checkpoint_files']={'model.safetensors':'wrong'}
        else: modified[1]['gpu_initial_free_floor_bytes']=5*GIB
        assert validate_registration(*modified)


def test_actual_parent_manifest_schema_and_empty_cli_report():
    import json, shutil, tempfile
    from pathlib import Path
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_analysis as analyzer
    from experiments.rineya_taps_adaptive_recurrence_campaign_v2.run_apparatus import sources
    admission_path=analyzer.PACKAGE/'bulk_admission_v1.json'
    admission=json.loads(admission_path.read_text(encoding='utf-8'))
    manifest={'sha256':sources()}
    assert validate_source_manifest(admission,manifest)==[]
    assert 'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl' not in manifest['sha256']
    key=(analyzer.PACKAGE/'bulk_native.py').relative_to(analyzer.REPO).as_posix()
    manifest['sha256'][key]='wrong'
    assert validate_source_manifest(admission,manifest)
    manifest={'sha256':sources()}
    with tempfile.TemporaryDirectory(prefix='bulk-analysis-cpu-fixture-',dir=analyzer.PACKAGE/'runs') as temporary:
        base=Path(temporary); root=base/'run'; result=root/'result'; result.mkdir(parents=True)
        shutil.copyfile(admission_path,root/'admission.json')
        shutil.copyfile(analyzer.PACKAGE/admission['review_receipt'],root/'review.json')
        (root/'source_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        (root/'terminal_receipt.json').write_text(json.dumps({'lane':'BULK_GPU_FP32_SCREEN','run':str(root),'status':'WORKER_TERMINAL','exit_code':2,'native_worker_seconds':5.}),encoding='utf-8')
        (root/'resources.jsonl').write_text('',encoding='utf-8')
        for name in ('outcomes.jsonl','trajectories.jsonl','forwards.jsonl'): (result/name).write_text('',encoding='utf-8')
        for name in ('receipt.json','qualification.json','A5_qualification.json','frozen_development.json','registration.json','matrix_forecast.json'): (result/name).write_text('{}',encoding='utf-8')
        assert analyzer.run(root,base/'analysis')==0
        report=json.loads((base/'analysis/REPORT.json').read_text(encoding='utf-8'))
        assert not report['complete'] and report['costs']['worker_wall_seconds']==5.
        assert report['costs']['physical_only']['wall_seconds']['complete_value'] is None
        assert not any('source manifest differs' in error for error in report['errors'])
