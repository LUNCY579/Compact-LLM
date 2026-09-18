"""Execute the twelve RANDOM fixture cells and twelve paired delta transforms.

Print a reproducible mechanics receipt. No pretrained source, native corpus,
selection endpoint, confirmation instrument or quality claim is consumed.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import torch
from .runner import FixtureRunner, FixtureConfig, weight_hash, canonical
from .preflight import verify_manifest
from .sparsity import prune_delta


def run() -> dict:
    torch.set_num_threads(1)
    source=verify_manifest()
    cells=[];initial_backbones=set()
    for draw in (101,211,307):
        for arm in 'ABCD':
            runner=FixtureRunner(FixtureConfig(arm,draw,tokens=512))
            initial_backbones.add(weight_hash(runner.actor.backbone))
            report=runner.run()
            if report['prompt_plus_generated_tokens']!=512 or report['inflight_groups'] or report['reference_hash']!=report['initial_hash']:
                raise AssertionError('fixture budget, completion or frozen-reference invariant failed')
            base={'backbone.'+n:p.detach().clone() for n,p in runner.reference.backbone.named_parameters()}
            trained={n:p.detach().clone() for n,p in runner.actor.named_parameters()}
            names=tuple(n for n in runner.trainable if n.startswith('backbone.'))
            pruned,receipt=prune_delta(base,trained,names)
            for n in receipt['new_parameters']:
                if not torch.equal(pruned[n],trained[n]):
                    raise AssertionError('new RoLA parameter was pruned')
            raw={'rollouts':{k:asdict(v) for k,v in runner.raw.items()},'journal':runner.journal}
            report['rollout_journal_sha256']=hashlib.sha256(canonical(raw)).hexdigest()
            report['sparse_pair']={'backbone_tensors':len(names),
                'selected_slots':sum(v['selected'] for v in receipt['tensors'].values()),
                'eligible_elements':sum(v['elements'] for v in receipt['tensors'].values()),
                'new_parameters':receipt['new_parameters'],'new_parameters_unchanged':True,
                'mask_receipt_sha256':hashlib.sha256(canonical(receipt)).hexdigest(),
                'extra_training_runs':0}
            cells.append(report)
    if len(initial_backbones)!=1:
        raise AssertionError('factorial initial backbones differ')
    return {'schema':1,'scope':'CPU_RANDOM_MECHANICS_ONLY','source_sha256':source,
            'initial_backbone_sha256':next(iter(initial_backbones)),
            'cells':cells,'trained_fixture_cells':12,'paired_pruned_derivatives':12,
            'total_accounted_prompt_plus_generated_tokens':sum(x['prompt_plus_generated_tokens'] for x in cells),
            'total_fixture_optimizer_steps':sum(x['optimizer_steps'] for x in cells),
            'pretrained_calls':0,'native_qualified':False,'efficacy_established':False,
            'release_permitted':False}


if __name__=='__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
