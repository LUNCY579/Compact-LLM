"""Prospective native CPU/GPU qualification on eight own-track calibration cases."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

PROGRAM = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROGRAM))
from shared.datasets import load_frozen_dataset
from shared.hashing import canonical_sha256,sha256_file,source_files,write_json
from shared.manifests import new_run,verify_registration,utc_now
from shared.model_loading import load_supa
from shared.integrity import state_hash,frozen_inventory,assert_unchanged
from shared.generation import evaluate_case
from shared.runtime import configure_runtime,runtime_identity,cuda_memory,synchronize
from shared.timing import machine_lease
from native_armr.src.operators import operator_factory,REGISTERED_ARMS
from native_armr.src.protocol import load_protocol

CAPTURE_SITES = (8,12,13,16,19)


def select_fixtures(rows):
    families = sorted({row['family'] for row in rows})
    if len(families) != 8:
        raise ValueError('qualification requires exactly eight calibration families')
    selected = [min((r for r in rows if r['family']==family),key=lambda r:r['example_id']) for family in families]
    if len({r['scenario_id'] for r in selected}) != 8 or any(r['track'] != 'native_armr' or r['split'] != 'calibration' for r in selected):
        raise ValueError('qualification fixture role/identity mismatch')
    return selected


def numeric_agreement(reference,actual,tokenwise=False):
    import torch
    from torch.nn import functional as F
    if reference.shape != actual.shape or not reference.numel():
        raise RuntimeError('qualification tensor shape mismatch')
    a,b = reference.detach().cpu().double(),actual.detach().cpu().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('nonfinite qualification tensor')
    difference = b-a
    if tokenwise:
        aa,bb = a.reshape(-1,a.shape[-1]),b.reshape(-1,b.shape[-1])
        relative = float(((bb-aa).norm(dim=-1)/aa.norm(dim=-1).clamp_min(1e-12)).max())
        cosine = float(F.cosine_similarity(aa,bb,dim=-1).min())
    else:
        relative = float(difference.norm()/a.norm().clamp_min(1e-12))
        cosine = float(F.cosine_similarity(a.reshape(-1),b.reshape(-1),dim=0))
    return {'max_abs':float(difference.abs().max()),'relative_l2':relative,'cosine':cosine}


def _capture_one(model,tokenizer,case,config,wrapper):
    import torch
    captured = {}
    def hook(index,output):
        if index not in captured:
            if not isinstance(output,torch.Tensor):
                raise RuntimeError('unqualified layer output type')
            captured[index] = output.detach().cpu().clone()
    handles = [model.model.layers[index].register_forward_hook(
        lambda module,args,out,index=index:hook(index,out)) for index in CAPTURE_SITES]
    try:
        row,logits = evaluate_case(model,tokenizer,case,config,wrapper)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(CAPTURE_SITES):
        raise RuntimeError('qualification hidden-site capture incomplete')
    return row,logits,captured


def run_phase(config,cases,output,repeat):
    import torch
    from armr.hooks import frozen_inference
    from armr.recurrence import FixedBlock
    model,tokenizer,identity = load_supa(config)
    values,operator_hashes,artifacts = {},{},{}
    try:
        for arm in REGISTERED_ARMS:
            wrapper = operator_factory(model,arm,config)
            extra = getattr(wrapper,'transform',None)
            if extra is not None:
                frozen_inventory(extra)
                operator_hashes[arm] = state_hash(extra)
            with wrapper,frozen_inference(model):
                for case in cases:
                    row,logits,hidden = _capture_one(model,tokenizer,case,config,wrapper)
                    repeated_identical = True
                    if repeat:
                        again,second,second_hidden = _capture_one(model,tokenizer,case,config,wrapper)
                        repeated_identical = (torch.equal(logits,second) and row['token_ids']==again['token_ids']
                            and row['gold_nll']==again['gold_nll'] and all(torch.equal(hidden[k],second_hidden[k]) for k in hidden))
                    if not repeated_identical:
                        raise RuntimeError('same-GPU generation/hidden/NLL repeat changed')
                    key = arm+'_'+canonical_sha256(case['example_id'])
                    target = output/'logits'/(config['device'].replace(':','_')+'_'+key+'.pt')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('xb') as stream:
                        torch.save(logits.cpu(),stream)
                    artifacts[target.relative_to(output).as_posix()] = sha256_file(target)
                    values[(arm,case['example_id'])] = {'row':row,'logits':logits,'hidden':hidden,
                                                       'repeat_identical':repeated_identical}
                    print(f"QUAL {config['device']} {arm} {case['example_id']} {row['wall_seconds']:.3f}s",flush=True)
            if extra is not None:
                frozen_inventory(extra)
                if state_hash(extra) != operator_hashes[arm]:
                    raise RuntimeError('qualification compute control changed')
            del wrapper,extra
        # Real same-device disabled and restoration identity, including greedy/NLL.
        probe = cases[0]
        original = values[('A0',probe['example_id'])]
        for mode in ('restored','disabled'):
            wrapper = operator_factory(model,'A0',config) if mode == 'restored' else FixedBlock(model,[12,13],2,enabled=False)
            with wrapper,frozen_inference(model):
                row,logits,hidden = _capture_one(model,tokenizer,probe,config,wrapper)
            if row['token_ids'] != original['row']['token_ids'] or row['gold_nll'] != original['row']['gold_nll'] or not torch.equal(logits,original['logits']) or any(not torch.equal(hidden[k],original['hidden'][k]) for k in hidden):
                raise RuntimeError(f'{mode} same-device identity failed')
        integrity = assert_unchanged(model,identity['state_sha256_before'],Path(identity['model_path']),identity['checkpoint_files_before'])
        identity.update(integrity)
        identity['operator_hashes'] = operator_hashes
        return values,identity,artifacts
    finally:
        if 'wrapper' in locals():
            del wrapper
        del model,tokenizer
        gc.collect()
        if config['device'].startswith('cuda'):
            synchronize(config['device'])
            torch.cuda.empty_cache()


def compare_phases(cpu,gpu,cases,config):
    q = config['qualification']
    records = []
    for arm in REGISTERED_ARMS:
        for case in cases:
            left,right = cpu[(arm,case['example_id'])],gpu[(arm,case['example_id'])]
            logits = numeric_agreement(left['logits'],right['logits'])
            hidden = {str(site):numeric_agreement(left['hidden'][site],right['hidden'][site],True) for site in CAPTURE_SITES}
            nll = abs(left['row']['gold_nll']-right['row']['gold_nll'])
            gates = {'logit_max_abs':logits['max_abs'] <= q['max_abs_logit_error'],
                     'logit_relative_l2':logits['relative_l2'] <= q['max_relative_logit_l2'],
                     'logit_cosine':logits['cosine'] >= q['minimum_logit_cosine'],
                     'nll':nll <= q['max_abs_nll_error'],
                     'greedy_identity':left['row']['token_ids']==right['row']['token_ids'],
                     'hidden':all(v['relative_l2'] <= q['max_relative_hidden_l2'] and v['cosine'] >= q['minimum_hidden_cosine'] for v in hidden.values()),
                     'gpu_repeat_identity':right['repeat_identical']}
            records.append({'arm':arm,'example_id':case['example_id'],'family':case['family'],
                            'logits':logits,'hidden':hidden,'absolute_nll_error':nll,
                            'cpu_response':left['row']['response'],'gpu_response':right['row']['response'],
                            'cpu_token_ids':left['row']['token_ids'],'gpu_token_ids':right['row']['token_ids'],
                            'gpu_repeat_identity':right['repeat_identical'],'gates':gates,'pass':all(gates.values())})
    return records


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROGRAM/'native_armr/configs/protocol_gpu_v2_2.json')
    parser.add_argument('--manifest',type=Path,required=True)
    args = parser.parse_args()
    config = load_protocol(args.config)
    if config['device'] != 'cuda:0':
        raise ValueError('native GPU qualifier requires GPU configuration')
    registration = verify_registration('native_armr',args.config)
    cases = select_fixtures(load_frozen_dataset(args.manifest,track='native_armr',role='calibration'))
    if registration['dataset_manifests'].get(args.manifest.resolve().relative_to(PROGRAM.parents[1]).as_posix()) != sha256_file(args.manifest):
        raise RuntimeError('unregistered qualification calibration manifest')
    configure_runtime(config)
    details = {'runtime_qualification':config['runtime_qualification'],'config_sha256':canonical_sha256(config),
        'registration_sha256':registration['registration_sha256'],'dataset_manifest_sha256':sha256_file(args.manifest),
        'sources':source_files('native_armr'),'runtime_identity':runtime_identity(config),'qualification_pass':False,
        'device':config['device'],'dtype':config['dtype'],'threads':config['threads'],'seed':config['seed'],
        'cache_mode':'full_prefix_no_cache','pid':__import__('os').getpid()}
    output,receipt = new_run('native_armr','00_runtime_qualification',details)
    print(f'QUALIFICATION_DIR={output}',flush=True)
    start = time.perf_counter()
    try:
        with machine_lease():
            reference_config = {**config,'device':'cpu','min_available_gib':3.0}
            cpu,cpu_identity,cpu_artifacts = run_phase(reference_config,cases,output,False)
            gpu,gpu_identity,gpu_artifacts = run_phase(config,cases,output,True)
            if cpu_identity['state_sha256_before'] != gpu_identity['state_sha256_before'] or cpu_identity['checkpoint_files_before'] != gpu_identity['checkpoint_files_before'] or cpu_identity['operator_hashes'] != gpu_identity['operator_hashes']:
                raise RuntimeError('CPU/GPU backbone or seeded compute-control bytes differ')
            comparisons = compare_phases(cpu,gpu,cases,config)
            report = {'qualification_pass':all(r['pass'] for r in comparisons),'comparisons':comparisons,
                      'fixture_ids':[case['example_id'] for case in cases],
                      'fixture_prompt_sha256':{case['example_id']:case['prompt_sha256'] for case in cases},
                      'thresholds':config['qualification'],'cpu_identity':cpu_identity,'gpu_identity':gpu_identity,
                      'artifacts':{**cpu_artifacts,**gpu_artifacts},'same_device_disabled_restored_identity':True,
                      'scope':'numerical/runtime qualification only; calibration outputs are not efficacy evidence'}
            write_json(output/'report.json',report)
            receipt.update(report_sha256=sha256_file(output/'report.json'),qualification_pass=report['qualification_pass'],
                           canonical_unchanged=True,checkpoint_files=gpu_identity['checkpoint_files_before'])
            if source_files('native_armr') != details['sources']:
                raise RuntimeError('qualification source changed during execution')
            if not report['qualification_pass']:
                raise RuntimeError('prospective CPU/GPU native qualification failed')
            receipt['status'] = 'COMPLETE'
    except BaseException as exc:
        receipt.update(status='FAILED',failure_receipt={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        memory = psutil.Process().memory_info()
        receipt.update(end_utc=utc_now(),wall_seconds=time.perf_counter()-start,peak_memory_bytes=getattr(memory,'peak_wset',memory.rss))
        try:
            receipt.update(cuda_memory(config))
        except BaseException as exc:
            receipt.update(status='FAILED',cuda_memory_final_error=str(exc))
        write_json(output/'run_manifest.json',receipt)
    if receipt['status'] != 'COMPLETE':
        raise RuntimeError('GPU qualification terminal integrity/resource check failed')
    print(f'QUALIFICATION_PASS={output}',flush=True)


if __name__ == '__main__':
    main()
