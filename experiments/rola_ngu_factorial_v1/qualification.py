"""Model-free source readiness and explicit optional stock-Qwen RANDOM checks.

No function downloads, calls from_pretrained, or authorizes native training.
"""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
from .runner import file_hash

RUNTIME={'torch':'2.10.0','transformers':'4.51.3','safetensors':'0.7.0'}


def readiness(source: Path | None = None) -> dict:
    packages={}
    for name in RUNTIME:
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    blockers=[f'{k}=={v} unavailable' for k,v in RUNTIME.items() if (packages[k] or '').split('+')[0]!=v]
    result={'scope':'MODEL_FREE_READINESS','packages':packages,'blockers':blockers,
            'pretrained_calls':0,'native_qualified':False,'training_authorized':False}
    if source is None:
        blockers.append('no local exact-source directory supplied')
        return result
    source=Path(source)
    lock=json.loads(Path(__file__).with_name('source_lock.json').read_text())
    checks={}
    for name,expected in lock['assets'].items():
        path=source/name
        if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
            blockers.append('missing/unsafe asset: '+name)
            continue
        checks[name]=file_hash(path)
        if checks[name]!=expected:
            blockers.append('asset SHA256 mismatch: '+name)
    result['asset_sha256']=checks
    blockers.extend(['actual-source actor/cache/gradient qualification not run',
                     'native corpus, budget and asynchronous-worker receipts absent'])
    return result


def random_qualification() -> dict:
    """Fresh tiny stock Qwen3 object. Missing runtime is BLOCKED, never SKIP/PASS."""
    status=readiness()
    if any('unavailable' in b for b in status['blockers']):
        return {**status,'status':'BLOCKED_NATIVE_RANDOM_RUNTIME'}
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from .actor import Actor
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(43)
        c=Qwen3Config(vocab_size=31,hidden_size=24,intermediate_size=40,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,
            attention_dropout=0.0,tie_word_embeddings=True,use_sliding_window=False)
        c._attn_implementation='eager'
        model=Qwen3ForCausalLM(c).float().eval()
        actor=Actor(model,sites=(1,2),rank=4).eval()
        ids=torch.tensor([[1,3,5,7,9]])
        with torch.no_grad():
            ref=model(input_ids=ids,use_cache=False).logits
            actual,_=actor(ids)
            torch.testing.assert_close(actual,ref,atol=2e-5,rtol=2e-5)
            for module in actor.rola.values():
                module.alpha.fill_(.2)
            active,_=actor(ids)
            if torch.equal(active,actual):
                raise AssertionError('active branch is unreachable')
            first,cache=actor(ids[:,:3],use_cache=True)
            last,_=actor(ids[:,3:],cache=cache,use_cache=True)
            torch.testing.assert_close(torch.cat((first,last),1),active,atol=2e-5,rtol=2e-5)
        actor.select_tail(2)
        actor(ids)[0].square().mean().backward()
        for module in actor.rola.values():
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) or not bool(p.grad.abs().sum()>0) for p in module.parameters()):
                raise AssertionError('active gradient missing/nonfinite')
    return {'status':'PASS_STOCK_RANDOM_ONLY','pretrained_calls':0,'optimizer_steps':0,
            'actual_source_qualified':False,'training_authorized':False,
            'checks':['stock_zero_path','active_path','cached_chunks','active_gradients']}


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--native-random',action='store_true')
    args=parser.parse_args()
    result=random_qualification() if args.native_random else readiness(args.source)
    print(json.dumps(result,indent=2))
    return 0 if result.get('status')=='PASS_STOCK_RANDOM_ONLY' else 2


if __name__=='__main__':
    raise SystemExit(main())
