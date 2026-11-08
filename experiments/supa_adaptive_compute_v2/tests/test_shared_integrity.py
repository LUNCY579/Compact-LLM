import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from shared.guards import guard_non_sealed_path,require_within,require_runtime
from shared.hashing import canonical_sha256,sha256_file,write_json
from shared.integrity import frozen_inventory,state_hash
from shared.evaluation import shard_cases,_read_complete,qualify_native_architecture
from shared import generation,manifests,evaluation
from shared import timing
from shared import model_loading
from shared.reporting import quality_cost_frontier


@pytest.mark.parametrize('name',['sealed/data.jsonl','fresh_sealed.reviewed.jsonl','TEST2/key.json','private/answers.json','sealed_roleplay_v2_DO_NOT_TRAIN.jsonl'])
def test_sealed_rejected_before_open(tmp_path,name):
    with pytest.raises(ValueError):
        guard_non_sealed_path(tmp_path/name)


def test_resolved_escape_rejected(tmp_path):
    with pytest.raises(ValueError):
        require_within(tmp_path/'owned/../outside.json',tmp_path/'owned')


def test_exclusive_evidence_never_replaces(tmp_path):
    path = tmp_path/'record.json'
    write_json(path,{'answer':1})
    expected = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json(path,{'answer':2})
    assert path.read_bytes() == expected


def test_frozen_inventory_rejects_gradient_and_trainable():
    layer = torch.nn.Linear(2,2).eval().requires_grad_(False)
    before = state_hash(layer)
    assert frozen_inventory(layer)['trainable_parameters'] == 0
    layer.weight.grad = torch.zeros_like(layer.weight)
    with pytest.raises(RuntimeError):
        frozen_inventory(layer)
    layer.weight.grad = None
    layer.weight.requires_grad_(True)
    with pytest.raises(RuntimeError):
        frozen_inventory(layer)
    layer.weight.requires_grad_(False)
    assert before == state_hash(layer)


def test_shards_cover_exactly_once_and_group_variants():
    rows = [{'scenario_id':f's{i}','example_id':f's{i}_{v}'} for i in range(50) for v in range(2)]
    shards = [shard_cases(rows,i,7) for i in range(7)]
    flat = [r['example_id'] for shard in shards for r in shard]
    assert len(flat) == len(set(flat)) == len(rows)
    for shard in shards:
        assert all(sum(r['scenario_id'] == item['scenario_id'] for r in shard) == 2 for item in shard)
    assert shards == [shard_cases(list(reversed(rows)),i,7)[::-1] for i in range(7)]


def test_runtime_change_fails_closed():
    config = dict(device='cpu',dtype='float32',use_cache=False,threads=4)
    require_runtime(config)
    for key,value in [('device','cuda'),('dtype','float16'),('use_cache',True),('threads',8)]:
        with pytest.raises(ValueError):
            require_runtime({**config,key:value})


def _fake_generation(monkeypatch):
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1),
            config=SimpleNamespace(num_attention_heads=1,num_key_value_heads=1,head_dim=2,
                                   hidden_size=2,intermediate_size=4,num_hidden_layers=1,vocab_size=3))
    tokenizer = SimpleNamespace(decode=lambda *a,**k:'yes',encode=lambda *a,**k:[])
    wrapper = SimpleNamespace(records=[],calls=0,capture=False,min_rms_ratio=1.,max_rms_ratio=1.,extra_layer_applications=0)
    case = {'example_id':'case','case_id':'case','scenario_id':'scenario','family':'logic',
            'prompt':'Test','scorer':{'type':'exact','answers':['yes']}}
    config = {'max_prompt_seconds':1.,'max_rss_gib':100.,'max_new_tokens':2,'top_k':2}
    monkeypatch.setattr(generation,'encode',lambda *a:torch.tensor([[0,2]]))
    return model,tokenizer,case,config,wrapper


def test_eos_cannot_bypass_expired_prefill_deadline(monkeypatch):
    args = _fake_generation(monkeypatch)
    clock = {'now':0.}
    monkeypatch.setattr(generation.time,'perf_counter',lambda:clock['now'])
    def slow_forward(*args):
        clock['now'] = 2.
        return torch.tensor([0.,9.,0.])
    monkeypatch.setattr(generation,'forward_logits',slow_forward)
    with pytest.raises(TimeoutError,match='deadline'):
        generation.evaluate_case(*args)


def test_deterministic_greedy_generation_without_cache(monkeypatch):
    args = _fake_generation(monkeypatch)
    monkeypatch.setattr(generation.time,'perf_counter',lambda:0.)
    monkeypatch.setattr(generation,'forward_logits',lambda *a:torch.tensor([0.,9.,0.]))
    first,logits = generation.evaluate_case(*args)
    second,repeated = generation.evaluate_case(*args)
    assert first['token_ids'] == second['token_ids'] == [1]
    assert first['score'] == 1 and first['termination'] == 'eos'
    assert torch.equal(logits,repeated)


def test_nll_forward_deadline_included(monkeypatch):
    model,tokenizer,case,config,wrapper = _fake_generation(monkeypatch)
    clock = {'now':0.}
    tokenizer.encode = lambda *a,**k:[2]
    class SlowNLL:
        def __init__(self):
            self.config,self.generation_config = model.config,model.generation_config
        def __call__(self,**kwargs):
            assert kwargs['use_cache'] is False
            clock['now'] = 2.
            return SimpleNamespace(logits=torch.zeros(1,3,3),past_key_values=None)
    monkeypatch.setattr(generation.time,'perf_counter',lambda:clock['now'])
    monkeypatch.setattr(generation,'forward_logits',lambda *a:torch.tensor([0.,9.,0.]))
    with pytest.raises(TimeoutError):
        generation.evaluate_case(SlowNLL(),tokenizer,case,config,wrapper)


def test_incomplete_and_cross_track_evidence_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'ROOT',tmp_path)
    path = tmp_path/'native_armr/evidence/run/one'
    write_json(path/'run_manifest.json',{'status':'STARTED','track':'native_armr'})
    with pytest.raises(RuntimeError,match='incomplete'):
        _read_complete(path,'native_armr')
    with pytest.raises(ValueError):
        _read_complete(path,'cross_scale')


def test_complete_receipt_requires_unchanged_rows(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'ROOT',tmp_path)
    path = tmp_path/'native_armr/evidence/run/one'
    rows = [{'example_id':'x','score':1}]
    write_json(path/'results.json',rows)
    write_json(path/'run_manifest.json',{'status':'COMPLETE','track':'native_armr','canonical_unchanged':True,
        'example_ids':['x'],'results_sha256':sha256_file(path/'results.json')})
    assert _read_complete(path,'native_armr')[1] == rows
    (path/'results.json').write_text('[]',encoding='utf-8')
    with pytest.raises(RuntimeError,match='changed'):
        _read_complete(path,'native_armr')


def test_empty_architecture_cannot_license_confirmatory(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'ROOT',tmp_path)
    monkeypatch.setattr(evaluation,'load_frozen_dataset',lambda *a,**k:[{'example_id':'one'}])
    with pytest.raises(RuntimeError,match='complete frozen architecture'):
        qualify_native_architecture({'arms':['A0','A3','AM']},tmp_path/'manifest',{'registration_sha256':'r'})


def test_no_frozen_registration_no_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(manifests,'ROOT',tmp_path)
    monkeypatch.setattr(manifests,'source_files',lambda *a:{})
    config = tmp_path/'native_armr/configs/protocol.json'
    write_json(config,{'seed':17})
    with pytest.raises(RuntimeError,match='exactly one'):
        manifests.verify_registration('native_armr',config)


def test_second_machine_lease_cannot_remove_owner(tmp_path,monkeypatch):
    monkeypatch.setattr(timing,'ROOT',tmp_path)
    with timing.machine_lease():
        before = (tmp_path/'.runtime.lock').read_bytes()
        with pytest.raises(FileExistsError):
            with timing.machine_lease():
                pytest.fail('second model process entered machine lease')
        assert (tmp_path/'.runtime.lock').read_bytes() == before
    assert not (tmp_path/'.runtime.lock').exists()


def test_frontier_preserves_real_quality_cost_tradeoff():
    def point(accuracy,latency):
        return dict(accuracy=accuracy,mean_wall_seconds=latency,flops=10,added_parameters=0,peak_memory_bytes=100)
    result = quality_cost_frontier({'A0':point(.75,1),'A3':point(.8,1.2),'bad':point(.7,1.3)})
    assert result['frontier'] == ['A0','A3']
    assert result['dominated_by']['bad'] == ['A0','A3']


def test_wrong_canonical_class_rejected(monkeypatch,tmp_path):
    import transformers
    from shared.legacy import ARMR
    reference = json.loads((ARMR/'evidence/02c_fixed_cpu_budget_v2/20260905T132743_802018Z/run_manifest.json').read_text())
    monkeypatch.setattr(model_loading,'require_available_memory',lambda _: {})
    monkeypatch.setattr(model_loading,'resolve_model',lambda _:tmp_path)
    monkeypatch.setattr(model_loading,'checkpoint_files',lambda _:reference['checkpoint_files_before'])
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',Mock(return_value=object()))
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',Mock(return_value=torch.nn.Linear(2,2)))
    config = dict(device='cpu',dtype='float32',use_cache=False,threads=4,seed=17)
    with pytest.raises(RuntimeError,match='wrong Supa class'):
        model_loading.load_supa(config)


def test_changed_canonical_sidecar_rejected_before_loading(monkeypatch,tmp_path):
    monkeypatch.setattr(model_loading,'require_available_memory',lambda _: {})
    monkeypatch.setattr(model_loading,'resolve_model',lambda _:tmp_path)
    monkeypatch.setattr(model_loading,'checkpoint_files',lambda _:{'model.safetensors':model_loading.CANONICAL_SHA256,'config.json':'changed'})
    config = dict(device='cpu',dtype='float32',use_cache=False,threads=4,seed=17)
    with pytest.raises(RuntimeError,match='sidecars differ'):
        model_loading.load_supa(config)


def test_track_registration_rejects_other_bank(tmp_path,monkeypatch):
    monkeypatch.setattr(manifests,'ROOT',tmp_path)
    config = tmp_path/'native_armr/configs/protocol.json'
    manifest = tmp_path/'data/cross_scale/manifest.json'
    write_json(config,{'seed':17})
    write_json(manifest,{'track':'cross_scale'})
    with pytest.raises(ValueError,match='another track'):
        manifests.freeze_registration('native_armr',config,[manifest])


@pytest.mark.parametrize('arguments',[
    ['--shard-count','0'], ['--shard-count','-1'], ['--max-wall-seconds','nan'],
    ['--max-wall-seconds','-1'], ['--arms','A0','A0'], ['--arms','A3','A0']])
def test_campaign_rejects_false_completion_inputs(monkeypatch,arguments):
    import importlib.util
    import sys
    path = Path(__file__).resolve().parents[1]/'scripts/run_native_campaign.py'
    spec = importlib.util.spec_from_file_location('test_native_campaign_cli',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys,'argv',[str(path),'--manifest','never_opened.json','--role','architecture_dev',*arguments])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
