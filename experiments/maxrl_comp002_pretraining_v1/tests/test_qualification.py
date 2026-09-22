"""Synthetic tensors/files only: these tests are not pretrained execution receipts."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
import torch.nn.functional as F
from experiments.maxrl_comp002_pretraining_v1.mechanics import (
    completion_labels, selected_logprobs, parameter_inventory)
from experiments.maxrl_comp002_pretraining_v1.preflight import (
    BASE, MANIFEST, HERE, inspect, local_file, read_json, write_receipt)


@pytest.mark.parametrize('left,right', [(0,0), (2,0), (0,2), (1,2)])
@pytest.mark.parametrize('eos,pad', [(0,0), (2,2), (2,0)])
def test_mask_eos_padding_causal_shift_and_gradient(left, right, eos, pad):
    tokens = [8,9,10,3,4,eos]
    ids = torch.tensor([[pad]*left + tokens + [pad]*right])
    attention = torch.tensor([[False]*left + [True]*6 + [False]*right])
    labels = completion_labels(ids, attention, [3], [3], [eos], ['eos'], 8)
    expected = torch.full_like(ids, -100)
    expected[0, left+3:left+6] = torch.tensor([3,4,eos])
    assert torch.equal(labels, expected)
    generator = torch.Generator().manual_seed(2002)
    logits = torch.randn(1, ids.shape[1], 16, generator=generator,
                         dtype=torch.float64, requires_grad=True)
    actual = selected_logprobs(logits, ids, labels)
    ref = torch.stack([F.log_softmax(logits[0, left+j-1], dim=-1)[tokens[j]]
                       for j in (3,4,5)])
    assert torch.allclose(actual, ref, atol=1e-12, rtol=0)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1,16), labels[:,1:].reshape(-1),
                         ignore_index=-100, reduction='sum')
    assert torch.allclose(-actual.sum(), ce, atol=1e-12, rtol=0)
    ce.backward()
    active = logits.grad.abs().sum(-1) > 0
    want = torch.zeros_like(active)
    want[0, left+2:left+5] = True
    assert torch.equal(active, want)
    assert torch.isfinite(logits.grad).all()


def test_length_cap_does_not_append_or_score_synthetic_eos():
    ids = torch.tensor([[8,9,3,4]])
    labels = completion_labels(ids, torch.ones_like(ids,dtype=torch.bool),
                               [2],[2],[0],['length'],2)
    assert labels.tolist() == [[-100,-100,3,4]]


@pytest.mark.parametrize('completion,reason,cap', [([3,0,4],'eos',8),
    ([3,0,0],'eos',8), ([3,4,5],'eos',8), ([3,0],'length',2),
    ([3,4],'length',8), ([3,4],'custom',2)])
def test_bad_termination_fails(completion,reason,cap):
    ids = torch.tensor([[8,9] + completion])
    with pytest.raises(ValueError):
        completion_labels(ids, torch.ones_like(ids,dtype=torch.bool),
                          [2],[len(completion)],[0],[reason],cap)


@pytest.mark.parametrize('prompt,count', [(0,3),(True,3),(2,True),(3,3),(2,0),(2,9)])
def test_invalid_span_counts(prompt,count):
    ids = torch.tensor([[8,9,3,4,0]])
    with pytest.raises(ValueError):
        completion_labels(ids, torch.ones_like(ids,dtype=torch.bool),
                          [prompt],[count],[0],['eos'],8)


def test_attention_hole_is_not_padding():
    ids = torch.tensor([[8,9,7,3,4,0]])
    with pytest.raises(ValueError):
        completion_labels(ids,torch.tensor([[True,True,False,True,True,True]]),
                          [2],[3],[0],['eos'],8)


def test_mixed_length_batch():
    ids = torch.tensor([[8,9,3,0,0],[8,3,4,5,0]])
    attention = torch.tensor([[True,True,True,True,False],[True]*5])
    labels = completion_labels(ids,attention,[2,1],[2,4],[0],['eos','eos'],8)
    assert (labels != -100).sum(1).tolist() == [2,4]


@pytest.mark.parametrize('mutation', ['nan','id','label','first','empty'])
def test_invalid_logprob_inputs(mutation):
    ids = torch.tensor([[8,9,3,0]])
    labels = torch.tensor([[-100,-100,3,0]])
    logits = torch.zeros(1,4,16)
    if mutation == 'nan': logits[0,0,0] = float('nan')
    if mutation == 'id': ids[0,0] = 16
    if mutation == 'label': labels[0,2] = 5
    if mutation == 'first': labels[0,0] = 8
    if mutation == 'empty': labels.fill_(-100)
    with pytest.raises(ValueError):
        selected_logprobs(logits,ids,labels)


def test_tied_parameters_count_once_and_inventory_is_stable():
    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(16,4)
    model.head = torch.nn.Linear(4,16,bias=False)
    model.head.weight = model.embed.weight
    before = model.embed.weight.detach().clone()
    a = parameter_inventory(model)
    b = parameter_inventory(model)
    assert a == b and a['unique_numel'] == a['trainable_numel'] == 64
    assert a['parameters'][0]['names'] == ['embed.weight','head.weight']
    assert torch.equal(before,model.embed.weight)


def test_frozen_inventory_is_not_guessed_trainable():
    model = torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Linear(4,2))
    model[0].requires_grad_(False)
    receipt = parameter_inventory(model)
    assert receipt['trainable_numel'] == 10
    assert receipt['unique_numel'] == 30
    model.requires_grad_(False)
    with pytest.raises(ValueError): parameter_inventory(model)


def test_shared_storage_separate_parameters_rejected():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(4))
    model.b = torch.nn.Parameter(model.a.detach())
    with pytest.raises(ValueError): parameter_inventory(model)


def test_meta_parameters_not_loaded():
    with pytest.raises(ValueError): parameter_inventory(torch.nn.Linear(4,2,device='meta'))


@pytest.mark.parametrize('relative', ['../escape','/etc/passwd','a\\b','missing'])
def test_path_escapes_and_missing_files(tmp_path,relative):
    with pytest.raises(ValueError): local_file(tmp_path,relative)


def test_duplicate_json_keys_and_nonfinite_rejected(tmp_path):
    p = tmp_path/'input.json'
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        p.write_text(text)
        with pytest.raises(ValueError): read_json(p)


def test_source_lock_pins():
    lock = read_json(HERE/'source_lock.json')
    assert lock['base_commit'] == BASE
    assert lock['manifest_sha256'] == MANIFEST
    assert len(lock['source_inventories']['maxrl_composable_skills_v2']) == 10


def test_missing_manifest_blocks(tmp_path):
    with pytest.raises(OSError): inspect(tmp_path,tmp_path)


def test_write_once_and_protect_original_trees(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    for target in (data/'receipt.json',
        tmp_path/'experiments/maxrl_composable_skills_v2/receipt.json',
        tmp_path/'experiments/maxrl_verifiable_data_v1/receipt.json'):
        with pytest.raises(ValueError): write_receipt({},target,tmp_path,data)
    out = tmp_path/'new.json'
    write_receipt({'training_authorized':False},out,tmp_path,data)
    with pytest.raises(FileExistsError): write_receipt({},out,tmp_path,data)


def test_no_training_cli_entrypoint():
    result = subprocess.run([sys.executable,'-m',
        'experiments.maxrl_comp002_pretraining_v1.preflight','--train'],
        capture_output=True,text=True)
    assert result.returncode == 2


def test_no_native_loader_or_optimizer_in_source():
    import ast
    for name in ('preflight.py','mechanics.py'):
        tree = ast.parse((HERE/name).read_text())
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        assert not called.intersection({'from_pretrained','generate','backward','step','load_state_dict'})
