"""Engineering fixtures only. No checkpoint downloads or protected data access."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from sbhm.core import HybridAttention, LoRALinear, Memory, rotate, cache_bytes
from sbhm import protocol, data, native, pipeline
from sbhm.quantization import adaptive_fake_quantize


def config():
    return SimpleNamespace(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=8, rms_norm_eps=1e-6)


class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True)+1e-6)).to(x.dtype)*self.weight


def rope(length, start=0, dtype=torch.float32):
    positions = torch.arange(start, start+length).float()[:, None]
    frequencies = torch.arange(1, 5).float()[None]/10
    phase = (positions*frequencies).repeat(1, 2)[None]
    return phase.cos().to(dtype), phase.sin().to(dtype)


class SourceAttention(nn.Module):
    """Independent tiny softmax reference; head_dim != hidden_size / heads."""
    def __init__(self, site=3):
        super().__init__()
        self.config, self.layer_idx = config(), site
        self.q_proj, self.k_proj = nn.Linear(16, 32, bias=False), nn.Linear(16, 16, bias=False)
        self.v_proj, self.o_proj = nn.Linear(16, 16, bias=False), nn.Linear(32, 16, bias=False)
        self.q_norm, self.k_norm = Norm(8), Norm(8)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        b, t, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(b,t,4,8)).transpose(1,2)
        k = self.k_norm(self.k_proj(hidden_states).view(b,t,2,8)).transpose(1,2)
        v = self.v_proj(hidden_states).view(b,t,2,8).transpose(1,2)
        q, k = rotate(q, *position_embeddings), rotate(k, *position_embeddings)
        out = F.scaled_dot_product_attention(q, k.repeat_interleave(2,1), v.repeat_interleave(2,1), is_causal=True)
        return self.o_proj(out.transpose(1,2).reshape(b,t,32)), None


class DynamicCache:
    def __init__(self):
        self.key_cache, self.value_cache = [], []


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(17)
    torch.set_num_threads(1)


@pytest.mark.parametrize('length', [1, 3, 11])
def test_local_matches_softmax_inside_window(length):
    source = SourceAttention()
    x = torch.randn(1,length,16)
    replacement = HybridAttention(copy.deepcopy(source), 'local', 16)
    actual, _ = replacement.sequence(x, *rope(length))
    expected = source(x, rope(length))[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('mode', ['local','hybrid'])
@pytest.mark.parametrize('chunk', [1, 2, 5, 13])
def test_full_chunk_incremental_parity(mode, chunk):
    module = HybridAttention(SourceAttention(), mode, 4)
    x, state, pieces = torch.randn(1,13,16), None, []
    full, _ = module.sequence(x, *rope(13))
    for start in range(0,13,chunk):
        piece = x[:,start:start+chunk]
        y, state = module.sequence(piece, *rope(piece.shape[1],start), state)
        pieces.append(y)
    torch.testing.assert_close(torch.cat(pieces,1), full, atol=1e-6, rtol=1e-5)
    assert state.seen == 13 and state.keys.shape == (1,2,4,8)
    assert state.matrix is None if mode == 'local' else state.matrix.shape == (1,2,8,8)


@pytest.mark.parametrize('mode', ['local','hybrid'])
def test_bounded_actual_storage(mode):
    module = HybridAttention(SourceAttention(), mode, 3)
    state = None
    expected = 2*2*3*8*4 + (2*8*8*4 if mode == 'hybrid' else 0)
    for t in range(25):
        _, state = module.sequence(torch.randn(1,1,16), *rope(1,t), state)
        if t >= 3:
            assert state.nbytes() == expected
            assert sum(x.untyped_storage().nbytes() for x in state.tensors()) == expected


def test_request_isolation_and_reset_only_memory():
    module = HybridAttention(SourceAttention(), 'hybrid', 3)
    x = torch.randn(1,7,16)
    first, state = module.sequence(x[:,:5], *rope(5))
    other, _ = module.sequence(x[:,:5], *rope(5))
    torch.testing.assert_close(first, other, atol=0, rtol=0)
    normal, _ = module.sequence(x[:,5:], *rope(2,5), state)
    saved_k, saved_v = state.keys.clone(), state.values.clone()
    state.reset_recurrent()
    reset, _ = module.sequence(x[:,5:], *rope(2,5), state)
    assert not torch.allclose(normal, reset)
    assert torch.equal(state.keys,saved_k) and torch.equal(state.values,saved_v)
    assert not any(isinstance(v, Memory) for v in vars(module).values())


def test_qwen3_geometry_is_not_inferred_from_width():
    source = SourceAttention()
    assert source.config.head_dim != source.config.hidden_size // source.config.num_attention_heads
    assert HybridAttention(source).head_dim == 8
    source = SourceAttention()
    source.config.head_dim = 4
    with pytest.raises(ValueError, match='projection mismatch'):
        HybridAttention(source)


@pytest.mark.parametrize('field', ['q_proj','k_proj','v_proj','o_proj'])
def test_all_projection_shapes_checked(field):
    source = SourceAttention()
    setattr(source, field, nn.Linear(16,17))
    with pytest.raises(ValueError, match='mismatch'):
        HybridAttention(source)


def test_zero_mix_reduces_to_local_and_gradients_are_finite():
    source = SourceAttention().requires_grad_(False)
    module = HybridAttention(source, 'hybrid', 32, mix_init=0)
    x = torch.randn(1,7,16)
    out, _ = module.sequence(x,*rope(7))
    torch.testing.assert_close(out, source(x,rope(7))[0], atol=1e-6,rtol=1e-5)
    module.mix.data.fill_(0.1)
    out, _ = module.sequence(x,*rope(7))
    out.square().mean().backward()
    for name in ('mix','memory_scale'):
        assert torch.isfinite(getattr(module,name).grad).all()
    assert torch.isfinite(module.forget.weight.grad).all()
    assert source.q_proj.weight.grad is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_memory_remains_fp32(dtype):
    module = HybridAttention(SourceAttention().to(dtype), 'hybrid',3)
    x = torch.randn(1,5,16).to(dtype)
    out,state = module.sequence(x,*rope(5,dtype=dtype))
    assert out.dtype == dtype and state.matrix.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_dynamic_cache_does_not_get_full_kv():
    module = HybridAttention(SourceAttention(),'hybrid',2)
    cache = DynamicCache()
    for i in range(6):
        module(torch.randn(1,1,16),rope(1,i),past_key_value=cache,cache_position=torch.tensor([i]))
    assert cache.key_cache == [] and cache.value_cache == []
    assert cache._sbhm[3].seen == 6
    assert cache_bytes(cache) == cache._sbhm[3].nbytes()
    with pytest.raises(ValueError, match='discontinuous'):
        module(torch.randn(1,1,16),rope(1),past_key_value=cache,cache_position=torch.tensor([0]))


def test_cache_storage_deduplicates_aliases():
    cache = DynamicCache()
    x = torch.zeros(20)
    cache.key_cache = [x, x[:5]]
    cache.value_cache = [x]
    assert cache_bytes(cache) == 80


def test_unsupported_cache_padding_batch_and_maps_rejected():
    module = HybridAttention(SourceAttention())
    with pytest.raises(ValueError, match='batch'):
        module.sequence(torch.randn(2,2,16),*rope(2))
    with pytest.raises(ValueError, match='DynamicCache'):
        module(torch.randn(1,2,16),rope(2),past_key_value=object())
    with pytest.raises(ValueError, match='padding'):
        module(torch.randn(1,2,16),rope(2),attention_mask=torch.full((1,1,2,2),-1.0))
    with pytest.raises(ValueError, match='attention-map'):
        module(torch.randn(1,2,16),rope(2),output_attentions=True)


def test_lora_zero_init_and_frozen_base():
    base = nn.Linear(8,6)
    original = base.weight.detach().clone()
    lora = LoRALinear(base,2)
    x = torch.randn(3,8)
    torch.testing.assert_close(lora(x),base(x),atol=0,rtol=0)
    lora(x).square().mean().backward()
    opt = torch.optim.SGD([lora.a,lora.b],lr=0.1)
    opt.step()
    assert torch.equal(base.weight,original) and base.weight.grad is None
    assert not torch.allclose(lora(x),base(x))


def reviewed_corpus(tmp_path):
    path = tmp_path/'corpus'
    data.make_fixture_corpus(path)
    manifest = protocol.read_json(path/'manifest.json')
    manifest['reviewed_for_sbhm'] = True  # fixture approval ONLY inside a test tempdir
    (path/'manifest.json').write_text(json.dumps(manifest))
    return path/'manifest.json'


def test_fixture_is_unreviewed_by_default(tmp_path):
    data.make_fixture_corpus(tmp_path/'data')
    with pytest.raises(protocol.Blocked,match='reviewed'):
        data.load_corpus(tmp_path/'data/manifest.json')


@pytest.mark.parametrize('field', ['id','family_id','prompt'])
def test_cross_split_leakage_rejected(tmp_path,field):
    p = reviewed_corpus(tmp_path)
    manifest = protocol.read_json(p)
    train = json.loads((p.parent/'train.jsonl').read_text().splitlines()[0])
    dev = [json.loads(x) for x in (p.parent/'dev.jsonl').read_text().splitlines()]
    dev[0][field] = train[field]
    target = p.parent/'dev.jsonl'
    target.write_text(''.join(json.dumps(x)+'\n' for x in dev))
    manifest['splits']['dev']['sha256'] = protocol.sha(target)
    p.write_text(json.dumps(manifest))
    with pytest.raises(protocol.Blocked,match='overlap'):
        data.load_corpus(p)


@pytest.mark.parametrize('mutation', ['protected','role','hash','extra_split'])
def test_manifest_fail_closed(tmp_path,mutation):
    p = reviewed_corpus(tmp_path)
    manifest = protocol.read_json(p)
    if mutation == 'extra_split':
        manifest['splits']['final'] = dict(manifest['splits']['dev'])
    else:
        manifest['splits']['dev'][{'protected':'protected','role':'role','hash':'sha256'}[mutation]] = {'protected':True,'role':'final','hash':'0'*64}[mutation]
    p.write_text(json.dumps(manifest))
    with pytest.raises(protocol.Blocked):
        data.load_corpus(p)


def test_safe_file_rejects_escape(tmp_path):
    with pytest.raises(protocol.Blocked):
        protocol.safe_file(tmp_path,'../outside')
    with pytest.raises(protocol.Blocked):
        protocol.safe_file(tmp_path,'/etc/passwd')


def test_seed_order_and_train_repetition():
    rows = [{'id':str(i)} for i in range(8)]
    a = data.ordered(rows,17,20)
    assert a == data.ordered(rows,17,20)
    assert a[:8] == a[8:16]
    assert a != data.ordered(rows,18,20)


class Tokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        return '<user>'+messages[0]['content']+'<assistant>'
    def encode(self,text,**kwargs):
        return [ord(c)%100+3 for c in text]
    def decode(self,ids,**kwargs):
        return ','.join(map(str,ids))


def test_response_only_and_eos_labels():
    row = {'prompt':'question','target':'answer'}
    prompt,_ = data.encode(Tokenizer(),row,False)
    ids,labels = data.encode(Tokenizer(),row,True)
    n = prompt.shape[1]
    assert torch.equal(ids[:,:n],prompt)
    assert torch.all(labels[:,:n] == -100)
    assert labels[0,-1] == 2 and torch.all(labels[:,n:] == ids[:,n:])


def test_operator_grant_and_expiry(tmp_path):
    binding = {k: 'a'*64 for k in ('code_sha256','source_sha256','config_sha256','corpus_sha256')}
    now = datetime.now(timezone.utc)
    g = {**binding,'approved_by':'test-fixture-not-native-approval','actions':['distill'],
         'issued_utc':now.isoformat(),'expires_utc':(now+timedelta(hours=1)).isoformat()}
    p = tmp_path/'grant.json'
    p.write_text(json.dumps(g))
    c = {'grant':str(p)}
    with pytest.raises(protocol.Blocked,match='allow-model-load'):
        protocol.authorize(c,'distill',binding)
    with pytest.raises(protocol.Blocked,match='allow-training'):
        protocol.authorize(c,'distill',binding,True)
    protocol.authorize(c,'distill',binding,True,True)
    g['expires_utc'] = (now-timedelta(seconds=1)).isoformat()
    p.write_text(json.dumps(g))
    with pytest.raises(protocol.Blocked,match='grant'):
        protocol.authorize(c,'distill',binding,True,True)


def test_receipt_binding_and_freshness(tmp_path):
    p = tmp_path/'receipt.json'
    receipt = {'status':'PASS_NATIVE_MECHANICS_ONLY','binding':{'arm':'D'},'created_utc':protocol.now()}
    p.write_text(json.dumps(receipt))
    protocol.require_qualification(p,{'arm':'D'})
    with pytest.raises(protocol.Blocked,match='another'):
        protocol.require_qualification(p,{'arm':'C'})
    receipt['created_utc'] = (datetime.now(timezone.utc)-timedelta(days=2)).isoformat()
    p.write_text(json.dumps(receipt))
    with pytest.raises(protocol.Blocked,match='expired'):
        protocol.require_qualification(p,{'arm':'D'})


def test_native_resource_guard_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(protocol.psutil,'virtual_memory',lambda:SimpleNamespace(available=16*2**30))
    monkeypatch.setattr(protocol.torch.cuda,'is_available',lambda:False)
    with pytest.raises(protocol.Blocked,match='cuda'):
        protocol.resources('cpu')


@pytest.mark.parametrize('value', [float('nan'),float('inf'),0,-1])
def test_quantization_invalid_parameter(value):
    with pytest.raises(ValueError):
        adaptive_fake_quantize(torch.ones(3),value)


def test_fake_quantization_is_not_energy_evidence():
    x = torch.tensor([[0.,0.1,-0.1,1.]])
    y,stats = adaptive_fake_quantize(x)
    assert torch.isfinite(y).all() and not stats['event_driven'] and stats['measured_energy'] is None


class TinyLM(nn.Module):
    """Random engineering host, NOT a pretrained model or native qualification."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(128,16)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = SourceAttention(i)
        self.head = nn.Linear(16,128)
    def forward(self,input_ids,labels=None,use_cache=False,logits_to_keep=0):
        assert use_cache is False
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = x + layer.self_attn(hidden_states=x,position_embeddings=rope(x.shape[1]),attention_mask=None)[0]
        if torch.is_tensor(logits_to_keep):
            x = x[:, logits_to_keep, :]
        logits = self.head(x)
        loss = None if labels is None else F.cross_entropy(logits[:,:-1].reshape(-1,128),labels[:,1:].reshape(-1),ignore_index=-100)
        selected = logits if torch.is_tensor(logits_to_keep) else logits[:,-logits_to_keep:] if logits_to_keep else logits
        return SimpleNamespace(logits=selected,loss=loss)


@pytest.mark.parametrize('arm', ['B','C','D'])
def test_staged_pipeline_fixture_train_save_reload(tmp_path,monkeypatch,arm):
    monkeypatch.setattr(pipeline,'resources',lambda *a,**k:{'rss_bytes':0})
    c = {'device':'cpu','layer':3,'window':4,'rank':2,'seed':17,'learning_rate':1e-3,
         'limits':{'steps':2,'max_train_tokens':256,'max_prompt_tokens':256},'corpus_manifest':str(reviewed_corpus(tmp_path))}
    binding = {'source_sha256':'s','config_sha256':'c','code_sha256':'x','arm':'A','delta_sha256':None}
    teacher_dir = tmp_path/'teacher';teacher_dir.mkdir()
    base = TinyLM().eval().requires_grad_(False)
    pipeline.capture(base,Tokenizer(),c,binding,teacher_dir)
    assert not base.model.layers[3].self_attn._forward_hooks
    student = copy.deepcopy(base)
    native.install(student,c,arm)
    binding = {**binding,'arm':arm}
    distill_dir = tmp_path/'distill';distill_dir.mkdir()
    r = pipeline.training(student,Tokenizer(),c,binding,distill_dir,'distill',teacher_dir)
    assert len(r['records']) == 2 and r['status'] == 'TRAINING_COMPLETED_NOT_EFFICACY'
    recovery_dir = tmp_path/'recovery';recovery_dir.mkdir()
    r = pipeline.training(student,Tokenizer(),c,binding,recovery_dir,'recover',parent_delta=distill_dir)
    assert r['parent_record']['stage'] == 'distill'
    clone = copy.deepcopy(base)
    native.install(clone,c,arm)
    native.load_delta(clone,recovery_dir,binding)
    ids = torch.tensor([[5,12,7,2]])
    torch.testing.assert_close(clone(ids).logits,student(ids).logits,atol=0,rtol=0)
    with pytest.raises(protocol.Blocked,match='incompatible'):
        native.load_delta(clone,recovery_dir,{**binding,'source_sha256':'changed'})


def test_real_transformers_random_qwen3_integration():
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != '4.51.3':
        pytest.skip('requires pinned Transformers, no version fallback')
    from transformers import Qwen3Config,Qwen3ForCausalLM
    cfg = Qwen3Config(hidden_size=16,intermediate_size=32,num_attention_heads=4,num_key_value_heads=2,
                     head_dim=8,num_hidden_layers=5,vocab_size=128,max_position_embeddings=128)
    cfg._attn_implementation = 'sdpa'  # the native loader's selected implementation
    c = {'layer':3,'window':4,'rank':2,'seed':17}
    base = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    ids = torch.tensor([[9,12,7,21,32,11,5]])
    with torch.no_grad():
        baseline = base(ids,use_cache=False,logits_to_keep=1).logits
    converted = {}
    for arm in 'ABCD':
        model = copy.deepcopy(base)
        native.install(model,c,arm)
        converted[arm] = model
        with torch.no_grad():
            expected = model(ids,use_cache=False,logits_to_keep=1).logits
            if arm in 'AB':
                torch.testing.assert_close(expected,baseline,atol=0,rtol=0)
            original = model.model.layers[c['layer']].self_attn
            model.model.layers[c['layer']].self_attn = native.IdentityAttention(original)
            wrapped = model(ids,use_cache=False,logits_to_keep=1).logits
            model.model.layers[c['layer']].self_attn = original
            torch.testing.assert_close(wrapped,expected,atol=0,rtol=0)
            for chunk in (1,3,ids.shape[1]):
                cache = None
                for part in ids.split(chunk,dim=1):
                    output = native.model_forward(model,part,cache)
                    cache = output.past_key_values
                torch.testing.assert_close(output.logits,expected,atol=1e-5,rtol=1e-4)
                assert torch.equal(output.logits.argmax(-1),expected.argmax(-1))
                native.check_cache(cache,c,arm,ids.shape[1])
            # A fresh request may not inherit the prior request's memory.
            fresh = native.model_forward(model,ids[:,:3])
            reference = model(ids[:,:3],use_cache=False,logits_to_keep=1).logits
            torch.testing.assert_close(fresh.logits,reference,atol=1e-5,rtol=1e-4)
            native.check_cache(fresh.past_key_values,c,arm,3)
            again = model(ids,use_cache=False,logits_to_keep=1).logits
            torch.testing.assert_close(again,expected,atol=0,rtol=0)
    with torch.no_grad():
        converted['D'].model.layers[c['layer']].self_attn.mix.zero_()
        zero_mix = converted['D'](ids,use_cache=False,logits_to_keep=1).logits
        local = converted['C'](ids,use_cache=False,logits_to_keep=1).logits
        torch.testing.assert_close(zero_mix,local,atol=0,rtol=0)


def test_matrix_is_complete_and_plan_only(tmp_path):
    from sbhm.matrix import build_plan
    root = tmp_path/'does-not-exist'
    plan = build_plan(tmp_path/'config.json',root)
    assert len(plan) == 22 and not root.exists()
    stages = [p[3] for p in plan]
    assert stages.count('qualify') == 10 and stages.count('distill') == 3
    assert stages.count('recover') == 3 and stages.count('evaluate') == 4
    assert stages[-1] == 'compare'
    for command in plan:
        if command[3] in ('distill','recover'):
            assert '--allow-training' in command


def test_source_inventory_and_hash_contract(tmp_path):
    cfg = json.loads((Path(__file__).parents[1]/'configs/native.example.json').read_text())
    root = tmp_path/'source';root.mkdir()
    model_cfg = {'model_type':'qwen3','head_dim':128,'num_hidden_layers':28,'max_position_embeddings':32768}
    (root/'config.json').write_text(json.dumps(model_cfg))
    (root/'tokenizer.json').write_text('{}')
    (root/'tokenizer_config.json').write_text('{}')
    (root/'model.safetensors').write_bytes(b'FIXTURE_NOT_REAL_WEIGHTS')
    cfg['source'] = {'path':str(root),'files':{p.name:protocol.sha(p) for p in root.iterdir()}}
    first = protocol.source_lock(cfg)
    assert len(first) == 64
    (root/'unlocked.txt').write_text('unexpected')
    with pytest.raises(protocol.Blocked,match='inventory'):
        protocol.source_lock(cfg)
    (root/'unlocked.txt').unlink()
    (root/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(protocol.Blocked,match='hash mismatch'):
        protocol.source_lock(cfg)


@pytest.mark.parametrize('change', ['window','rank','dtype','steps','learning_rate'])
def test_configuration_rejects_invalid_values(change):
    cfg = json.loads((Path(__file__).parents[1]/'configs/native.example.json').read_text())
    if change == 'steps':
        cfg['limits']['steps'] = 0
    else:
        cfg[change] = {'window':0,'rank':True,'dtype':'int8','learning_rate':float('nan')}[change]
    with pytest.raises(protocol.Blocked):
        protocol.validate(cfg)


def fake_report(arm):
    trace = {'stage':'distill','records':[{'row_id':'train1','input_tokens':6,'supervised_positions':6}]}
    recovery = {'stage':'recover','records':[{'row_id':'train1','input_tokens':6,'supervised_positions':2}], 'parent_record':trace}
    binding = {'arm':arm,'source_sha256':'s','code_sha256':'c','config_sha256':'k','environment':{},'delta_sha256':None if arm=='A' else arm}
    for stage in (trace,recovery):
        stage.update(status='TRAINING_COMPLETED_NOT_EFFICACY',corpus_sha256='d',binding=copy.deepcopy(binding))
        stage['records'][0].update(step=0,loss=0.0,gradient_norm=0.0)
    return {'status':'DEV_MEASUREMENT_ONLY','binding':binding,'corpus_sha256':'d',
            'adaptation_record':None if arm=='A' else recovery,'strict_exact_match':int(arm=='D'),
            'normalized_exact_match':int(arm=='D'),
            'predictions':[{'id':'dev1','family_id':'fixture-dev','target':'fixture-answer',
                            'prediction':'fixture-answer' if arm=='D' else 'wrong',
                            'strict_exact_match':arm=='D','normalized_exact_match':arm=='D',
                            'reset_recurrent_at':None}]}


def test_compare_checks_exposure_and_does_not_promote(tmp_path):
    paths = []
    for arm in 'ABCD':
        p = tmp_path/f'{arm}.json';p.write_text(json.dumps(fake_report(arm)));paths.append(p)
    result = pipeline.compare(paths)
    assert result['arms']['D']['repairs'] == 1 and result['promotion'] is False
    r = fake_report('C')
    r['adaptation_record']['records'][0]['input_tokens'] = 7
    paths[2].write_text(json.dumps(r))
    with pytest.raises(protocol.Blocked,match='exposure'):
        pipeline.compare(paths)
