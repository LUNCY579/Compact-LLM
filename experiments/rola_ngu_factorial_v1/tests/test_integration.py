"""Actor, logical-async execution, custody and exact-resume regression tests."""
import pytest
import torch
from experiments.rola_ngu_factorial_v1.actor import Actor
from experiments.rola_ngu_factorial_v1.fixtures import tiny, independent_logits
from experiments.rola_ngu_factorial_v1.runner import FixtureConfig, FixtureRunner, file_hash, weight_hash
from experiments.rola_ngu_factorial_v1.rollout import generate
from experiments.rola_ngu_factorial_v1.sampling import sampler_state, budget_state
from experiments.rola_ngu_factorial_v1.sparsity import cast_audit, prune_delta
from experiments.rola_ngu_factorial_v1.data_contract import overlap, admit, policy_input
from experiments.rola_ngu_factorial_v1.analysis import paired_retention, abstention_endpoints
from experiments.rola_ngu_factorial_v1.qualification import readiness

torch.set_num_threads(1)


def actor(active=False):
    a=Actor(tiny(dtype=torch.float64),sites=(1,2),rank=4).eval()
    if active:
        with torch.no_grad():
            for m in a.rola.values():m.alpha.fill_(.2)
    return a


@pytest.mark.parametrize('seed',[17,43,97])
@pytest.mark.parametrize('origin',[0,7,30])
def test_dense_actor_matches_independent_matrix_reference(seed,origin):
    a=Actor(tiny(seed,dtype=torch.float64),sites=(1,2),rank=4).eval()
    ids=torch.tensor([[1,2,3,4,5]])
    assert a.backbone.config.hidden_size != a.backbone.config.num_attention_heads*a.backbone.config.head_dim
    expected=independent_logits(a.backbone,ids,origin=origin)
    actual,_=a(ids,origin=origin)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('split',[1,2,4])
def test_active_actor_chunk_cache_and_causality(split):
    a=actor(True);ids=torch.tensor([[1,2,3,4,5]])
    with torch.no_grad():
        full,_=a(ids,origin=5)
        first,cache=a(ids[:,:split],origin=5,use_cache=True,sequence_id='s')
        kept=tuple(t.clone() for entry in cache.kv for t in entry)
        rest,_=a(ids[:,split:],cache=cache,use_cache=True,sequence_id='s')
        torch.testing.assert_close(torch.cat((first,rest),1),full,atol=1e-12,rtol=1e-12)
        assert all(torch.equal(x,y) for x,y in zip(kept,(t for entry in cache.kv for t in entry)))
        altered=ids.clone();altered[:,-1]=0
        changed,_=a(altered,origin=5)
        torch.testing.assert_close(changed[:,:-1],full[:,:-1],atol=0,rtol=0)
        for m in a.rola.values():m.alpha.zero_()
        zero,_=a(ids,origin=5)
        assert not torch.equal(full,zero)
        with pytest.raises(ValueError,match='stale'):
            a(ids[:,:1],cache=cache,use_cache=True,sequence_id='s')


def test_foreign_sequence_cache_and_grad_cache_rejected():
    a=actor(True);ids=torch.tensor([[1,2]])
    with pytest.raises(ValueError,match='no_grad'):
        a(ids,use_cache=True)
    with torch.no_grad():
        _,cache=a(ids,use_cache=True,sequence_id='s')
        for owner,sequence in ((a,'other'),(actor(True),'s')):
            with pytest.raises(ValueError,match='foreign'):
                owner(ids,cache=cache,use_cache=True,sequence_id=sequence)
        a.version+=1
        with pytest.raises(ValueError,match='stale'):
            a(ids,cache=cache,use_cache=True,sequence_id='s')


@pytest.mark.parametrize('active',[False,True])
def test_rollout_and_training_log_prob_paths_identical(active):
    a=actor(active)
    row=generate(a,prompt=(1,2,3),prompt_id='p',sample_id='s',
                 generator=torch.Generator().manual_seed(23),verifier=lambda t,e:0,eos=7,max_new=6)
    lp=a.completion_logps(row.prompt,row.tokens,chunk=1)
    torch.testing.assert_close(lp,lp.new_tensor(row.completion.behavior_logps),atol=1e-12,rtol=1e-12)
    lp.sum().backward()
    for m in a.rola.values():
        assert m.alpha.grad is not None and m.alpha.grad.abs()>0
        assert m.pq.grad is not None
        assert bool(m.pq.grad.abs().sum()>0)==active


@pytest.mark.parametrize('prompt,tokens',[((1,),(8,)),((True,),(2,)),((1,),(-1,)),((),(1,))])
def test_invalid_completion_inputs(prompt,tokens):
    with pytest.raises(ValueError):actor().completion_logps(prompt,tokens)


@pytest.mark.parametrize('modification',['hook','rope','sliding','dtype','projection'])
def test_unsupported_backbones_fail_closed(modification):
    m=tiny()
    if modification=='hook':m.model.layers[0].register_forward_hook(lambda *args:None)
    if modification=='rope':m.config.rope_scaling={'type':'dynamic'}
    if modification=='sliding':m.config.use_sliding_window=True
    if modification=='dtype':m.bfloat16()
    if modification=='projection':m.model.layers[0].self_attn.q_proj=torch.nn.Linear(24,24,bias=False)
    with pytest.raises(ValueError):Actor(m)


@pytest.mark.parametrize('arm',list('ABCD'))
@pytest.mark.parametrize('draw',[101,211,307])
def test_exact_resume_and_full_budget(tmp_path,arm,draw):
    config=FixtureConfig(arm,draw,tokens=128)
    uninterrupted=FixtureRunner(config)
    frozen={n:p.detach().clone() for n,p in uninterrupted.actor.named_parameters() if not p.requires_grad}
    uninterrupted.step()
    assert uninterrupted.events
    path=tmp_path/'state.pt';uninterrupted.save(path)
    resumed=FixtureRunner.load(path,config)
    assert sampler_state(resumed.sampler)==sampler_state(uninterrupted.sampler)
    expected=uninterrupted.run();actual=resumed.run()
    assert actual==expected and resumed.raw==uninterrupted.raw
    assert resumed.journal==uninterrupted.journal
    assert torch.equal(resumed.generator.get_state(),uninterrupted.generator.get_state())
    assert budget_state(resumed.budget)==budget_state(uninterrupted.budget)
    assert expected['prompt_plus_generated_tokens']==128
    assert expected['reference_hash']==expected['initial_hash']
    assert expected['inflight_groups']==0 and expected['pretrained_calls']==0
    for n,p in uninterrupted.actor.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n])
    # Compare optimizer moments and step counters, not only final model bytes.
    for old,new in zip(uninterrupted.optimizer.state.values(),resumed.optimizer.state.values()):
        for k in old:
            if isinstance(old[k],torch.Tensor):assert torch.equal(old[k],new[k])
            else:assert old[k]==new[k]


@pytest.mark.parametrize('field',['config','budget','reference_hash','optimizer','source_sha256','events'])
def test_checkpoint_contract_mutations_rejected(tmp_path,field):
    config=FixtureConfig('D',101,128);r=FixtureRunner(config);r.step()
    path=tmp_path/'state.pt';r.save(path)
    state=torch.load(path,weights_only=True)
    if field=='config':state[field]['tokens']=256
    if field=='budget':state[field]['limit']=256
    if field=='reference_hash':state[field]='0'*64
    if field=='optimizer':state[field]['param_groups'][0]['lr']=.01
    if field=='source_sha256':state[field]='0'*64
    if field=='events':state[field][0]['samples'][0]='not-a-raw-sample'
    torch.save(state,path);path.with_suffix('.pt.sha256').write_text(file_hash(path)+'\n')
    with pytest.raises(ValueError):FixtureRunner.load(path,config)


def test_checkpoint_corruption_and_overwrite_rejected(tmp_path):
    r=FixtureRunner(FixtureConfig('B',101,64));r.step();path=tmp_path/'state.pt';r.save(path)
    with pytest.raises(FileExistsError):r.save(path)
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(ValueError,match='hash'):FixtureRunner.load(path,r.config)


def test_common_backbone_and_rola_initialization_paired():
    runners=[FixtureRunner(FixtureConfig(a)) for a in 'ABCD']
    assert len({weight_hash(r.actor.backbone) for r in runners})==1
    assert weight_hash(runners[2].actor)==weight_hash(runners[3].actor)
    assert any(runners[3].step() for _ in range(1))
    assert runners[3].run()['max_observed_version_lag']>0


def row(i,prompt,use,instrument='FRESH'):
    return {'id':i,'parent_id':'parent-'+i,'prompt':prompt,'instrument':instrument,
            'allowed_uses':[use],'verifier_id':'reviewed-verifier-v1','gold':'SECRET'}


@pytest.mark.parametrize('variant',['exact','digits','parent','id'])
def test_overlap_controls(variant):
    a=row('a','There are 12 apples','TRAIN');b=row('b','There are 99 pears','SELECTION')
    if variant=='exact':b['prompt']=' There  ARE 12 APPLES '
    if variant=='digits':b['prompt']='There are 27 apples'
    if variant=='parent':b['parent_id']=a['parent_id']
    if variant=='id':b['id']=a['id']
    receipt=overlap([a],[b]);assert not receipt['clean'] and not receipt['training_authorized']
    assert policy_input(a)=='There are 12 apples' and 'SECRET' not in policy_input(a)


@pytest.mark.parametrize('instrument',['RCI-001','UNKNOWN64_SPENT','LONG_CONTEXT_DEMONSTRATION_SPENT'])
def test_protected_instruments_remain_closed(instrument):
    with pytest.raises(ValueError):admit([row('a','hello','TRAIN',instrument)],'TRAIN')


def test_retention_denominator_and_abstention_are_separate():
    source={'a':True,'b':False,'c':False};candidate={'a':False,'b':True,'c':True}
    result=paired_retention(source,candidate)
    assert result['lost']==1 and result['gained']==2 and result['source_correct']==1
    assert not result['no_forgetting_established']
    result=paired_retention({'a':False},{'a':True});assert result['loss_rate_given_source_correct'] is None
    r=abstention_endpoints({'a':True,'b':False},{'a':False,'b':True},{'a':False,'b':True})
    assert r['unanswerable']['lost']==1 and r['candidate_false_abstentions']==1


def test_fp32_tiny_delta_erasure_is_measured():
    b=torch.tensor([1.,1.0001]);c=b+1e-5
    r=cast_audit(b,c)
    assert r['original_changed']==2 and r['paired_delta_erased']==2
    assert r['baseline_elements_rounded']==1 and not r['serving_qualified']
    with pytest.raises(FloatingPointError):cast_audit(torch.tensor([1e10]),torch.tensor([1e10]),torch.float16)


def test_actual_trained_backbone_pruning_preserves_new_rola_parameters():
    r=FixtureRunner(FixtureConfig('D',101,128));r.run()
    base={'backbone.'+n:p.detach().clone() for n,p in r.reference.backbone.named_parameters()}
    trained={n:p.detach().clone() for n,p in r.actor.named_parameters()}
    names=tuple(n for n in r.trainable if n.startswith('backbone.'))
    pruned,receipt=prune_delta(base,trained,names)
    assert receipt['new_parameters'] and not receipt['new_parameters_pruned']
    for n in receipt['new_parameters']:assert torch.equal(pruned[n],trained[n])
    assert all(d['selected']==d['elements']//4 for d in receipt['tensors'].values())


def test_asset_readiness_cannot_authorize_execution(tmp_path):
    r=readiness(tmp_path)
    assert not r['native_qualified'] and r['pretrained_calls']==0
    assert r['blockers']
