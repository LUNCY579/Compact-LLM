"""Bounded logical-async CPU fixture trainer, NOT a native NGU throughput study.

Two logical workers publish immutable sampled groups with one-event delivery lag.
A learner update can occur between sampling and ingesting a group. Computation is
serial in this simulator; it tests off-policy accounting, not async speedups.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
import torch
from .actor import Actor
from .fixtures import tiny
from .objective import policy_loss
from .rollout import Rollout, generate
from .sampling import (Completion, Sampler, TokenBudget, sampler_state, restore_sampler,
                       budget_state, restore_budget)


def canonical(value) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def weight_hash(model) -> str:
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(canonical([name,list(tensor.shape),str(tensor.dtype)]))
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def file_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class FixtureConfig:
    arm: str = 'A'
    draw: int = 101
    tokens: int = 512
    workers: int = 2
    # Toy-only optimizer; not a native authorization or proposed research optimum.
    learning_rate: float = .001

    def __post_init__(self):
        if self.arm not in 'ABCD' or len(self.arm) != 1 or self.draw not in (101,211,307):
            raise ValueError('unknown registered arm/draw')
        if type(self.tokens) is not int or not 32 <= self.tokens <= 8192 or self.tokens % 16:
            raise ValueError('fixture token cap must be a multiple of 16 in [32,8192]')
        if type(self.workers) is not int or not 1 <= self.workers <= 4 or self.learning_rate != .001:
            raise ValueError('fixture concurrency/optimizer contract changed')


class FixtureRunner:
    def __init__(self, config=FixtureConfig()):
        self.config=config
        with torch.random.fork_rng():
            torch.manual_seed(43)
            self.actor=Actor(tiny(),sites=(1,2) if config.arm in 'CD' else (),rank=4).eval()
        self.reference=copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.initial_hash=weight_hash(self.actor)
        self.trainable=self.actor.select_tail(2)
        self.optimizer=torch.optim.AdamW([p for p in self.actor.parameters() if p.requires_grad],
            lr=config.learning_rate,betas=(.9,.95),weight_decay=0)
        self.sampler=Sampler(ngu=config.arm in 'BD',seed=47,group_size=4,max_pending=32)
        self.budget=TokenBudget(config.tokens)
        self.generator=torch.Generator().manual_seed(47)
        worlds=[(a,b) for a in range(8) for b in range(8)]
        random.Random(config.draw).shuffle(worlds)
        self.worlds=tuple(worlds[:12])
        self.events=deque(); self.retry=deque(); self.raw={}
        self.journal=[]; self.dispatches=0; self.deliveries=0; self.next_world=0
        self.loss_tokens=0; self.failed=False

    def _dispatch(self) -> bool:
        if self.failed:
            raise ValueError('aborted fixture cannot continue')
        if self.budget.spent+sum(self.budget.pending.values())+16 > self.budget.limit:
            return False
        if self.retry:
            episode,prompt_id,prompt=self.retry.popleft()
        else:
            index=self.next_world % len(self.worlds); self.next_world+=1
            a,b=self.worlds[index]
            episode=f'episode-{self.next_world}'
            prompt_id=f'world-{a}-{b}'; prompt=(1,a,b)
        request=f'group-{self.dispatches}'
        self.budget.reserve(request,16)
        snapshot=copy.deepcopy(self.actor).eval().requires_grad_(False)
        rows=[]
        try:
            for i in range(4):
                sample_id=f'{request}-{i}'
                target=(prompt[1]+prompt[2])%8
                row=generate(snapshot,prompt=prompt,prompt_id=prompt_id,sample_id=sample_id,
                    generator=self.generator,verifier=lambda tokens,done:int(tokens==(target,)),
                    eos=7,max_new=1)
                self.raw[sample_id]=row; rows.append(sample_id)
        except Exception as exc:
            self.budget.settle(request)  # preserve unknown failed costs
            self.failed=True
            self.journal.append({'status':'ABORTED','request':request,'error':type(exc).__name__,'completed_samples':rows})
            raise
        actual=sum(len(self.raw[i].prompt)+len(self.raw[i].tokens) for i in rows)
        self.budget.settle(request,actual)
        self.events.append({'episode':episode,'prompt_id':prompt_id,'prompt':prompt,'samples':rows})
        self.dispatches+=1
        return True

    def prime(self) -> None:
        while len(self.events)<self.config.workers and self._dispatch():
            pass

    def _learn(self, decision) -> float:
        rows=[self.raw[c.sample_id] for c in decision.samples]
        if any(r.completion != c for r,c in zip(rows,decision.samples)):
            raise ValueError('raw rollout/decision mismatch')
        for row in rows:
            row.validate(self.actor.version)
        # Toy uses exactly one token; the shared objective itself supports masks.
        current=torch.stack([self.actor.completion_logps(r.prompt,r.tokens) for r in rows])
        old=current.new_tensor([r.completion.behavior_logps for r in rows])
        with torch.no_grad():
            ref=torch.stack([self.reference.completion_logps(r.prompt,r.tokens) for r in rows])
        value=policy_loss(current,old,ref,current.new_tensor(decision.advantages),torch.ones_like(current,dtype=torch.bool))
        self.optimizer.zero_grad(set_to_none=True); value.backward()
        parameters=[p for p in self.actor.parameters() if p.requires_grad]
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError('missing/nonfinite actor gradient')
        torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
        self.optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in parameters):
            raise FloatingPointError('nonfinite optimizer result')
        self.actor.version+=1; self.loss_tokens+=sum(len(r.tokens) for r in rows)
        return float(value.detach())

    def step(self) -> bool:
        if self.failed:
            raise ValueError('aborted fixture cannot continue')
        self.prime()
        if not self.events:
            return False
        event=self.events.popleft()
        batch=tuple(self.raw[i].completion for i in event['samples'])
        before=self.actor.version
        try:
            decision=self.sampler.ingest(event['episode'],batch,before)
            loss=None
            if decision.status=='UPDATE':
                loss=self._learn(decision)
            elif decision.status=='RETRY':
                self.retry.append((event['episode'],event['prompt_id'],event['prompt']))
        except Exception as exc:
            self.failed=True
            self.journal.append({'status':'ABORTED','event':event,'error':type(exc).__name__})
            raise
        self.deliveries+=1
        self.journal.append({'status':decision.status,'episode':event['episode'],'learner_before':before,
            'behavior_version':batch[0].policy_version,'loss':loss,'stale':decision.stale_count,
            'group_count':decision.total_count,'advantages':list(decision.advantages),
            'spent_tokens':self.budget.spent})
        self.prime()
        return True

    def run(self) -> dict:
        while self.step():
            pass
        return self.report()

    def report(self) -> dict:
        return {'scope':'RANDOM_CPU_LOGICAL_ASYNC_FIXTURE_NOT_EFFICACY',
                'config':asdict(self.config),'initial_hash':self.initial_hash,
                'final_hash':weight_hash(self.actor),'reference_hash':weight_hash(self.reference),
                'optimizer_steps':self.actor.version,'dispatches':self.dispatches,'deliveries':self.deliveries,
                'prompt_plus_generated_tokens':self.budget.spent,'loss_tokens':self.loss_tokens,
                'unresolved_retries_at_budget':len(self.retry),'inflight_groups':len(self.events),
                'max_observed_version_lag':max((r['learner_before']-r['behavior_version'] for r in self.journal if 'learner_before' in r),default=0),
                'statuses':{s:sum(r['status']==s for r in self.journal) for s in sorted({r['status'] for r in self.journal})},
                'pretrained_calls':0,'native_training_authorized':False,'efficiency_claim':False}

    def state(self) -> dict:
        from .preflight import verify_manifest
        return {'schema':1,'source_sha256':verify_manifest(),'config':asdict(self.config),
                'initial_hash':self.initial_hash,'weights':self.actor.state_dict(),
                'weight_hash':weight_hash(self.actor),'reference':self.reference.state_dict(),
                'reference_hash':weight_hash(self.reference),'version':self.actor.version,
                'trainable':list(self.trainable),'optimizer':self.optimizer.state_dict(),
                'rng':self.generator.get_state(),'torch_rng':torch.get_rng_state(),
                'sampler':sampler_state(self.sampler),'budget':budget_state(self.budget),
                'raw':{k:asdict(v) for k,v in self.raw.items()},'events':list(self.events),
                'retry':list(self.retry),'journal':copy.deepcopy(self.journal),
                'dispatches':self.dispatches,'deliveries':self.deliveries,
                'next_world':self.next_world,'loss_tokens':self.loss_tokens,'failed':self.failed}

    def save(self, path: Path) -> str:
        """Exclusive-create, hash-bound checkpoint; never persist cache objects."""
        import os
        from .preflight import ROOT
        path=Path(path)
        if path.resolve().is_relative_to(ROOT.parent.parent.resolve()) or any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError('checkpoint must be outside checkout with no symlinks')
        if path.exists() or path.with_suffix(path.suffix+'.sha256').exists():
            raise FileExistsError(path)
        with path.open('xb') as f:
            torch.save(self.state(),f); f.flush(); os.fsync(f.fileno())
        digest=file_hash(path)
        with path.with_suffix(path.suffix+'.sha256').open('x') as f:
            f.write(digest+'\n')
        return digest

    @classmethod
    def load(cls, path: Path, expected: FixtureConfig):
        """Construct a fresh runner, validate everything, then expose it to caller."""
        from .preflight import verify_manifest
        path=Path(path); side=path.with_suffix(path.suffix+'.sha256')
        if any(p.is_symlink() for p in (path,side,*path.parents)) or path.stat().st_size>128*2**20:
            raise ValueError('unsafe fixture checkpoint')
        if file_hash(path)!=side.read_text().strip():
            raise ValueError('checkpoint hash mismatch')
        s=torch.load(path,map_location='cpu',weights_only=True)
        if s.get('schema')!=1 or s['source_sha256']!=verify_manifest() or s['config']!=asdict(expected):
            raise ValueError('checkpoint source/config binding mismatch')
        result=cls(expected)
        if s['initial_hash']!=result.initial_hash or s['trainable']!=list(result.trainable):
            raise ValueError('checkpoint source/trainable mismatch')
        for field,model in (('weights',result.actor),('reference',result.reference)):
            current=model.state_dict()
            if current.keys()!=s[field].keys():
                raise ValueError('checkpoint parameter inventory mismatch')
            for name,value in s[field].items():
                if value.shape!=current[name].shape or value.dtype!=current[name].dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError('checkpoint tensor mismatch')
            model.load_state_dict(s[field])
        if weight_hash(result.actor)!=s['weight_hash'] or weight_hash(result.reference)!=s['reference_hash'] or s['reference_hash']!=result.initial_hash:
            raise ValueError('checkpoint weights/reference hash mismatch')
        sampler,budget=restore_sampler(s['sampler']),restore_budget(s['budget'])
        if sampler_state(sampler)['config']!=sampler_state(result.sampler)['config'] or budget.limit!=expected.tokens:
            raise ValueError('checkpoint sampler/budget configuration differs')
        counters=('version','dispatches','deliveries','next_world','loss_tokens')
        if any(type(s[k]) is not int or s[k]<0 for k in counters) or s['deliveries']>s['dispatches'] or sampler.learner_version>s['version']:
            raise ValueError('checkpoint chronology invalid')
        raw={}
        for key,item in s['raw'].items():
            c=dict(item['completion']); c['behavior_logps']=tuple(c['behavior_logps'])
            row=Rollout(Completion(**c),tuple(item['prompt']),tuple(item['tokens']),item['terminated'])
            row.validate(s['version'])
            if row.completion.sample_id!=key:
                raise ValueError('checkpoint raw identity differs')
            raw[key]=row
        if not sampler.seen <= raw.keys() or len(s['events'])!=s['dispatches']-s['deliveries'] or len(s['events'])>expected.workers:
            raise ValueError('checkpoint raw/event coverage differs')
        event_ids=[]
        for event in s['events']:
            if len(event['samples'])!=4 or any(i not in raw or raw[i].completion.prompt_id!=event['prompt_id'] or raw[i].prompt!=tuple(event['prompt']) for i in event['samples']):
                raise ValueError('checkpoint event differs from raw samples')
            event_ids.extend(event['samples'])
        if len(set(event_ids))!=len(event_ids) or set(event_ids)&sampler.seen:
            raise ValueError('checkpoint event replay')
        if type(s['failed']) is not bool:
            raise ValueError('invalid failure status')
        wanted=result.optimizer.state_dict()['param_groups']
        groups=s['optimizer']['param_groups']
        if groups!=wanted:
            raise ValueError('checkpoint optimizer hyperparameters/inventory differ')
        retry_ids=[e for e,_,_ in s['retry']]
        inflight_episodes={e['episode'] for e in s['events']}
        if len(set(retry_ids))!=len(retry_ids) or set(retry_ids)&inflight_episodes:
            raise ValueError('checkpoint retry event replay')
        for episode,prompt_id,prompt in s['retry']:
            if episode not in sampler.pending or sampler.pending[episode].prompt_id!=prompt_id or not any(r.prompt==tuple(prompt) and r.completion.prompt_id==prompt_id for r in raw.values()):
                raise ValueError('checkpoint retry does not match pending episode')
        if not s['failed'] and (budget.pending or budget.spent!=s['dispatches']*16 or len(raw)!=s['dispatches']*4 or set(raw)!=sampler.seen|set(event_ids)):
            raise ValueError('checkpoint successful-run accounting mismatch')
        result.optimizer.load_state_dict(s['optimizer'])
        for state in result.optimizer.state.values():
            for value in state.values():
                if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):
                    raise ValueError('nonfinite optimizer checkpoint')
        result.actor.version=s['version']; result.sampler=sampler; result.budget=budget
        result.generator.set_state(s['rng']); result.raw=raw
        result.events=deque(s['events']); result.retry=deque((e,p,tuple(t)) for e,p,t in s['retry'])
        result.journal=s['journal']; result.failed=s['failed']
        for k in counters[1:]:
            setattr(result,k,s[k])
        torch.set_rng_state(s['torch_rng'])
        return result


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=list('ABCD'),default='A')
    parser.add_argument('--draw',type=int,choices=[101,211,307],default=101)
    parser.add_argument('--tokens',type=int,default=512)
    args=parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(FixtureRunner(FixtureConfig(args.arm,args.draw,args.tokens)).run(),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
