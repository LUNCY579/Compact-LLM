import copy
import json
from pathlib import Path
import pytest
from experiments.titan_tangent_arithmetic_v1.binding import checked_file, digest, preflight, package_digest, qualification_identity


def make_manifest(tmp_path):
    base=tmp_path/'model';base.mkdir()
    (base/'config.json').write_text(json.dumps({'model_type':'qwen3','vocab_size':16}))
    for name in ('model.safetensors','tokenizer.json','tokenizer_config.json'):
        (base/name).write_text('fixture-not-model-weights')
    data=tmp_path/'data.jsonl';data.write_text('fixture-not-read-by-preflight')
    forbidden=tmp_path/'forbidden.json';forbidden.write_text('[]')
    files={p.name:digest(p) for p in base.iterdir()}
    bank={'path':str(data),'sha256':digest(data),'status':'developmental','human_reviewed':True,
          'tokenizer_sha256':files['tokenizer.json'],'eos_token_id':0}
    return {'schema':'TTA-001-HOST-1','action':'qualify','device':'cpu',
            'source_commit':'a'*40,'package_sha256':package_digest(),
            'base':{'family':'Nimbus-1.6.0','directory':str(base),'files':files},
            'scope_prefixes':['model.layers.8.'],'data':bank,'qualification_probe':dict(bank),
            'forbidden_parents':{'path':str(forbidden),'sha256':digest(forbidden)},
            'runtime':{},'limits':{'max_wall_seconds':120,'max_logical_calls':20,'max_length':64,
                                 'max_rows':2,'max_optimizer_steps':0,'min_available_ram_gib':12}}


def test_offline_preflight_is_not_native_pass(tmp_path):
    r=preflight(make_manifest(tmp_path),check_git=False,check_runtime=False)
    assert r['status']=='PREFLIGHT_PASS_NOT_EXECUTED'
    assert r['model_loads']==r['optimizer_steps']==0


@pytest.mark.parametrize('bad',['source','package','device','scope','data_status','review','tokenizer','ram','optimizer','length','extra_file','custom','family','endpoint','qualification_probe'])
def test_preflight_rejects_unqualified_inputs(tmp_path,bad):
    m=make_manifest(tmp_path)
    if bad=='source':m['source_commit']='main'
    if bad=='package':m['package_sha256']='0'*64
    if bad=='device':m['device']='cuda'
    if bad=='scope':m['scope_prefixes']=[]
    if bad=='data_status':m['data']['status']='spent'
    if bad=='review':m['data']['human_reviewed']=False
    if bad=='tokenizer':m['data']['tokenizer_sha256']='0'*64
    if bad=='ram':m['limits']['min_available_ram_gib']=7
    if bad=='optimizer':m['limits']['max_optimizer_steps']=1
    if bad=='length':m['limits']['max_length']=8192
    if bad=='extra_file':(Path(m['base']['directory'])/'unbound.py').write_text('')
    if bad=='custom':
        c=Path(m['base']['directory'])/'config.json'
        c.write_text(json.dumps({'model_type':'qwen3','vocab_size':16,'auto_map':{'x':'custom'}}))
        m['base']['files']['config.json']=digest(c)
    if bad=='family':m['base']['family']='Supa-A1'
    if bad=='endpoint':m['action']='diagnostic'
    if bad=='qualification_probe':m['qualification_probe']['status']='sealed'
    with pytest.raises(ValueError):preflight(m,check_git=False,check_runtime=False)


def test_qualified_model_can_use_new_developmental_bank_but_not_new_scope(tmp_path):
    m=make_manifest(tmp_path);identity=qualification_identity(m)
    receipt=tmp_path/'q.json';receipt.write_text(json.dumps({'status':'QUALIFICATION_PASS','qualification_identity':identity,'quality_claim':False}))
    m['action']='study';m['qualification_receipt']={'path':str(receipt),'sha256':digest(receipt)}
    bank=tmp_path/'other.jsonl';bank.write_text('other developmental bank')
    m['data']=dict(m['data'],path=str(bank),sha256=digest(bank))
    preflight(m,check_git=False,check_runtime=False)
    m['scope_prefixes']=['model.layers.9.']
    with pytest.raises(ValueError):preflight(m,check_git=False,check_runtime=False)


def test_missing_local_asset_is_not_a_remote_backup_verdict(tmp_path):
    with pytest.raises(FileNotFoundError,match='remote custody not assessed'):
        checked_file({'path':str(tmp_path/'missing'),'sha256':'a'*64})


def test_changed_bound_file_refused(tmp_path):
    p=tmp_path/'asset';p.write_text('before');ref={'path':str(p),'sha256':digest(p)}
    p.write_text('after')
    with pytest.raises(ValueError,match='hash mismatch'):checked_file(ref)
