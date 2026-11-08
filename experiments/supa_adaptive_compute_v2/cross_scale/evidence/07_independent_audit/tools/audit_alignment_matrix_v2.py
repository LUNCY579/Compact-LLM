"""Read-only, stdlib-only audit of the original CPU Track B v2 full matrix.

Run only after all 16 pairs and one successful aggregate exist. This file lives
outside the frozen scientific source map. It never imports programme helpers,
torch, or numerical libraries. Dense activation/bridge files are SHA-256 streamed
as opaque bytes, never deserialized. Failure receipts and prior attempts remain
untouched. See --help for the two required arguments.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import itertools
import json
import math
import pathlib
import re
import sys

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
AUDIT_ROOT = TOOLS_DIR.parent
TRACK = AUDIT_ROOT.parents[1]
PROGRAMME = TRACK.parent
REPO = PROGRAMME.parents[1]
OBJECTIVES = ('B0', 'B1', 'B2', 'B3', 'BS', 'BR')
AXES = ((8, 12, 16, 19), (12, 16, 20, 24))
PAIRS = tuple(itertools.product(*AXES))
STRATA = ('all_tokens', 'ordinary_tokens', 'special_tokens', 'final_prompt_token')
METRIC_KEYS = ('mean_token_cosine', 'mean_token_relative_mse', 'global_relative_mse',
               'predicted_rms', 'target_rms', 'residual_magnitude_ratio')
EXPECTED_CONFIG_SHA256 = '68b5de95cf4eca48d8a93fb26ff7cae9148afa1636d05fc834d5a628fd9655be'
GATES = {'minimum_delta_cosine': .60, 'minimum_cosine_improvement': .10,
         'maximum_mse_fraction_of_B0': .80, 'minimum_null_margin': .10,
         'minimum_all_token_cosine': .50, 'residual_ratio_bounds': [.25, 4.]}
LIMITATIONS = [
    'This checks saved metadata, scalar bounds, file hashes and ranking arithmetic. It does not rerun hidden-state capture, bridge fitting, donor transformations, losses or metric tensor arithmetic.',
    'Activation and bridge files are streamed only for SHA-256; they are not loaded or deserialized. The activation token IDs and special-ID set are not exposed in capture JSON, so ordinary/special counts can be checked for consistency across all cells but not independently reconstructed from token IDs.',
    'Native donor replay and frozen/zero-gradient statements are receipt claims checked for complete consistent coverage; no live model/state/gradient inspection occurs. Pair JSON does not retain an independent rotary-buffer inventory.',
    'Canonical checkpoint and donor identity claims are reconciled against registered reference metadata. Checkpoint weight files are never opened. Git commit status is not checked.',
    'Mechanism GO is a prerequisite for a separately frozen small generation screen. It does not establish capability, license large generation, or open the 8B/routing/attention/distillation gates.',
]

class AuditError(RuntimeError):
    def __init__(self, message, code='INTEGRITY_FAILURE'):
        super().__init__(message)
        self.code = code

def require(condition, message, code='INTEGRITY_FAILURE'):
    if not condition:
        raise AuditError(message, code)

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def digest_ok(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

def within(path, root, *, strict=False):
    path, root = pathlib.Path(path).resolve(strict=strict), pathlib.Path(root).resolve()
    require(path.is_relative_to(root), f'path escapes allowed subtree: {path}', 'PATH_REJECTED')
    return path

def exclusive_output(path, audit_root=AUDIT_ROOT):
    root = pathlib.Path(audit_root).resolve()
    path = within(path, root)
    require(path != root and not path.is_relative_to(root/'tools'),
            'output must be a fresh result directory below the audit root, outside tools', 'PATH_REJECTED')
    require(not path.exists(), f'output already exists: {path}', 'OUTPUT_EXISTS')
    path.mkdir(parents=True, exist_ok=False)
    return path

def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')

def assert_full_coverage(records):
    # Deliberately inspect no candidate metrics before exact full pair coverage.
    keys = [(row.get('supa_layer'), row.get('donor_layer')) for row in records]
    require(len(keys) == len(set(keys)) == 16 and set(keys) == set(PAIRS),
            'all 16 unique registered pairs are required before selection', 'INCOMPLETE_MATRIX')

def reconstruct_selection(records, gates=GATES):
    """Pure, independent exact gate/ranking arithmetic; refuses partial input."""
    assert_full_coverage(records)
    require(gates == GATES, 'selection gates differ from the original preregistration')
    indexed = {(row['supa_layer'],row['donor_layer']):row for row in records}
    decisions, eligible = [], []
    for i,j in PAIRS:
        row = indexed[i,j]
        require(set(row['objectives']) == set(OBJECTIVES), 'all six controls are required')
        baseline = row['objectives']['B0']['heldout']['delta']['ordinary_tokens']
        null = max(row['objectives'][key]['heldout']['delta']['ordinary_tokens']['mean_token_cosine'] for key in ('BS','BR'))
        for objective in ('B1','B2','B3'):
            delta = row['objectives'][objective]['heldout']['delta']
            ordinary = delta['ordinary_tokens']
            cosine, mse = ordinary['mean_token_cosine'], ordinary['mean_token_relative_mse']
            tests = {
                'absolute_delta_cosine': cosine >= gates['minimum_delta_cosine'],
                'improvement_over_B0': cosine >= baseline['mean_token_cosine'] + gates['minimum_cosine_improvement'],
                'relative_error_improvement': mse <= baseline['mean_token_relative_mse'] * gates['maximum_mse_fraction_of_B0'],
                'above_null_controls': cosine >= null + gates['minimum_null_margin'],
                'all_token_floor': delta['all_tokens']['mean_token_cosine'] >= gates['minimum_all_token_cosine'],
                'scale_stability': gates['residual_ratio_bounds'][0] <= ordinary['residual_magnitude_ratio'] <= gates['residual_ratio_bounds'][1],
            }
            candidate = {'supa_layer':i, 'donor_layer':j, 'objective':objective,
                'checks':tests, 'go':all(tests.values()), 'ordinary_delta_cosine':cosine,
                'ordinary_delta_relative_mse':mse}
            decisions.append(candidate)
            if candidate['go']:
                eligible.append(candidate)
    eligible.sort(key=lambda row:(-row['ordinary_delta_cosine'], row['ordinary_delta_relative_mse'],
        row['supa_layer'],row['donor_layer'],row['objective']))
    return {'status':'MECHANISM_GO' if eligible else 'MECHANISM_NO_GO', 'selected':eligible[:1],
        'selection_data':'cross_scale/calibration scenario-held-out activations only',
        'comparisons':decisions, 'large_generation_licensed':False,
        'small_generation_eligible_for_separate_freeze':bool(eligible)}

class MatrixAuditor:
    def __init__(self, run_id):
        require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id) is not None, 'invalid run ID','PATH_REJECTED')
        self.run_id = run_id
        self.capture_dir = within(TRACK/'evidence/01_alignment_matrix'/run_id,TRACK/'evidence')
        self.pair_dir = within(TRACK/'evidence/02_bridge_objectives'/run_id,TRACK/'evidence')
        self.local = within(TRACK/'local'/run_id,TRACK/'local')
        self.inventory = {'capture':None, 'pairs':[], 'aggregates':[]}
        self.files = {}
        self.checks = 0
        self.full_matrix_ready = False
        self.selection = None

    def check(self, condition, message, code='INTEGRITY_FAILURE'):
        self.checks += 1
        require(condition,message,code)

    def file_hash(self, path, root=REPO):
        path = within(path, root, strict=True)
        require(path.is_file(),f'expected file: {path}')
        value = sha(path)
        self.files[str(path)] = value
        return value

    def load(self, path, root=REPO):
        path = within(path, root, strict=True)
        payload = path.read_bytes()
        self.files[str(path)] = hashlib.sha256(payload).hexdigest()
        def reject_constant(value):
            raise AuditError('nonfinite JSON number: '+value)
        def unique_pairs(items):
            result={}
            for key,value in items:
                require(key not in result,'duplicate JSON key '+key)
                result[key]=value
            return result
        return json.loads(payload.decode('utf-8'), parse_constant=reject_constant, object_pairs_hook=unique_pairs)

    def attempt_inventory(self, directory, arm):
        directory = within(directory,TRACK/'evidence',strict=True)
        started_path=directory/'started.json'
        final_path, abandoned_path=directory/'run_manifest.json',directory/'abandonment.json'
        self.check(not (final_path.exists() and abandoned_path.exists()),f'ambiguous terminal receipts: {directory}')
        final=final_path if final_path.exists() else abandoned_path if abandoned_path.exists() else None
        item={'path':str(directory),'status':'UNFINISHED','manifest_sha256':None,'files':{}}
        # Keep discovered attempts in the rejection receipt even if validating
        # their metadata subsequently fails.
        self.inventory['pairs' if arm=='pair' else 'aggregates'].append(item)
        # Hash all small evidence artifacts, including failed/abandoned leftovers,
        # without interpreting any partial metric values for selection.
        for child in sorted(directory.rglob('*')):
            if child.is_file():
                item['files'][child.relative_to(directory).as_posix()]=self.file_hash(child, directory)
        if not started_path.is_file():
            item['status']='MISSING_STARTED_RECEIPT'
            return item,None
        started=self.load(started_path)
        if final is None:
            return item,None
        terminal=self.load(final)
        item['status']=terminal.get('status','INVALID')
        item['manifest_sha256']=self.files[str(final)]
        item['terminal_file']=final.name
        if (directory/'failure.json').exists():
            failure=self.load(directory/'failure.json')
            item['failure']={key:failure.get(key) for key in ('type','error')}
        self.validate_receipt(started,terminal,arm,directory)
        return item,terminal

    def validate_receipt(self, started, terminal, arm, directory):
        self.check(started.get('status')=='STARTED' and terminal.get('status') in ('COMPLETE','FAILED','ABANDONED'),f'invalid terminal status: {directory}')
        self.check(terminal.get('track')=='cross_scale' and terminal.get('arm')==arm,f'wrong receipt role/track: {directory}')
        self.check(all(terminal.get(key)==value for key,value in started.items() if key not in ('status','peak_rss_bytes')),f'started contract was rewritten: {directory}')
        self.check(terminal.get('registration')==self.registration and terminal.get('registration_sha256')==canonical(self.registration),f'embedded registration binding mismatch: {directory}')
        self.check(terminal.get('sources')==self.registration['sources'] and terminal.get('source_sha256')==canonical(self.registration['sources']) and terminal.get('config_sha256')==canonical(self.config),f'source/config binding mismatch: {directory}')
        self.check(all(terminal.get(key)==self.config[key] for key in ('device','dtype','seed','threads')) and terminal.get('cache_mode')=='none' and terminal.get('torch')=='2.12.1+cu126' and terminal.get('transformers')=='5.14.1',f'nonreference execution profile: {directory}')
        self.check(terminal.get('generation_count')==0 and terminal.get('model_execution')==(arm!='aggregate'),f'unsupported generation/execution role: {directory}')
        if terminal['status']=='FAILED':
            self.check((directory/'failure.json').is_file(),f'failed attempt lacks failure receipt: {directory}')
        if terminal['status']=='ABANDONED':
            self.check(terminal.get('original_process_absent') is True,f'abandonment lacks recorded process absence: {directory}')

    def load_registration_and_capture(self):
        capture=self.load(self.capture_dir/'capture.json')
        self.registration=capture['registration']
        self.config=self.registration['config']
        self.check(canonical(self.config)==EXPECTED_CONFIG_SHA256 and self.config['mechanism_gate']==GATES and self.config['delta_steps']==96 and self.config['objectives']==list(OBJECTIVES),'only the original frozen CPU v2 config is supported')
        diskpath=within(self.registration['registration_path'],TRACK/'evidence/00_preregistration',strict=True)
        disk=self.load(diskpath)
        self.check({key:value for key,value in self.registration.items() if key!='registration_path'}==disk,'embedded registration differs from frozen file')
        contract={key:disk[key] for key in ('track','config_sha256','config_path','config','sources','dataset_manifests')}
        self.check(disk['status']=='FROZEN' and disk['track']=='cross_scale' and canonical(contract)==disk['registration_sha256']==diskpath.parent.name,'frozen registration contract digest mismatch')
        configpath=within(REPO/disk['config_path'],TRACK/'configs',strict=True)
        self.check(self.file_hash(configpath)==disk['config_sha256'] and self.load(configpath)==self.config,'config file changed')
        for relative,expected in disk['sources'].items():
            path=within(REPO/relative,REPO,strict=True)
            self.check(self.file_hash(path)==expected,'registered source changed: '+relative)
        started=self.load(self.capture_dir/'started.json')
        terminal=self.load(self.capture_dir/'run_manifest.json')
        self.validate_receipt(started,capture,'capture',self.capture_dir)
        self.check(capture==terminal and capture['status']=='COMPLETE','capture receipts disagree or incomplete')
        self.inventory['capture']={'path':str(self.capture_dir),'status':capture['status'],'capture_sha256':self.files[str(self.capture_dir/'capture.json')],'run_manifest_sha256':self.files[str(self.capture_dir/'run_manifest.json')]}
        self.capture=capture

    def collect_full_inventory(self):
        expected_names={f'S{i}_D{j}' for i,j in PAIRS}
        if self.pair_dir.exists():
            self.check(all(x.is_dir() and x.name in expected_names for x in self.pair_dir.iterdir()),'unregistered depth directory or unexpected pair artifact')
        completed={}
        blockers=[]
        for i,j in PAIRS:
            root=self.pair_dir/f'S{i}_D{j}'
            selected=[]
            if root.exists():
                self.check(all(x.is_dir() and re.fullmatch('attempt_[0-9]{3,}',x.name) for x in root.iterdir()),f'invalid attempt directory under {root}')
            for directory in sorted(root.glob('attempt_*')):
                item,terminal=self.attempt_inventory(directory,'pair')
                item.update(supa_layer=i,donor_layer=j)
                if terminal is None:
                    blockers.append('unfinished '+str(directory))
                elif terminal['status']=='COMPLETE':
                    selected.append((directory,terminal))
            if len(selected)!=1:
                blockers.append(f'S{i}_D{j} has {len(selected)} complete attempts; expected exactly one')
            else:
                completed[i,j]=selected[0]
        aggregates=[]
        aggregate_root=self.capture_dir/'aggregate'
        if aggregate_root.exists():
            self.check(all(x.is_dir() and re.fullmatch('attempt_[0-9]{3,}',x.name) for x in aggregate_root.iterdir()),'invalid aggregate attempt directory')
        for directory in sorted(aggregate_root.glob('attempt_*')):
            item,terminal=self.attempt_inventory(directory,'aggregate')
            if terminal is None:
                blockers.append('unfinished aggregate '+str(directory))
            elif terminal['status']=='COMPLETE':
                aggregates.append((directory,terminal))
        if len(aggregates)!=1:
            blockers.append(f'{len(aggregates)} complete aggregates; expected exactly one')
        # Completeness is decided before opening any pair metrics or selection.
        self.check(not blockers,'; '.join(blockers),'INCOMPLETE_MATRIX')
        self.completed=completed
        self.aggregate=aggregates[0]
        self.full_matrix_ready=True

    def verify_capture_data(self):
        capture=self.capture
        dataset=within(capture['dataset_manifest'],PROGRAMME/'data/generated/v2/cross_scale',strict=True)
        self.check(dataset.name=='manifest.json' and self.file_hash(dataset)==capture['dataset_manifest_sha256']==self.registration['dataset_manifests'].get(dataset.relative_to(REPO).as_posix()),'capture dataset hash/registration mismatch')
        bank=self.load(dataset)
        body={key:value for key,value in bank.items() if key!='manifest_sha256'}
        self.check(canonical(body)==bank['manifest_sha256'] and bank['track']=='cross_scale' and bank['status']=='FROZEN_UNSEALED' and bank['sealed_access'] is False,'unfrozen or wrong-track dataset')
        info=bank['datasets']['calibration']
        datafile=within(dataset.parent/info['path'],dataset.parent,strict=True)
        self.check(datafile.name=='calibration.jsonl' and self.file_hash(datafile)==info['sha256'],'calibration bytes mismatch')
        rows=[]
        with datafile.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():continue
                raw=json.loads(line)
                self.check((raw['track'],raw['split'],raw['variant'])==('cross_scale','calibration','original'),'noncalibration scenario')
                row={key:raw[key] for key in ('example_id','scenario_id','family','template_id','prompt_sha256','prompt','alignment_partition')}
                del raw
                self.check(canonical(row['prompt'])==row['prompt_sha256'] and row['alignment_partition']==('holdout' if int(row['scenario_id'].rsplit('_',1)[1])%4==3 else 'fit'),'prompt hash or frozen scenario partition mismatch')
                rows.append(row)
        self.check(len(rows)==len({x['example_id'] for x in rows})==len({x['scenario_id'] for x in rows})==128,'capture requires exactly 128 distinct scenarios')
        fit=[x for x in rows if x['alignment_partition']=='fit']
        held=[x for x in rows if x['alignment_partition']=='holdout']
        self.check(len(fit)==capture['fit_prompts']==96 and len(held)==capture['holdout_prompts']==32 and sorted(x['scenario_id'] for x in held)==capture['heldout_scenarios'],'capture partition coverage mismatch')
        lengths=capture['truncation']
        by_id={x['example_id']:x for x in lengths}
        self.check(len(lengths)==len(by_id)==128 and set(by_id)=={x['example_id'] for x in rows},'capture length identity coverage mismatch')
        for item in lengths:
            self.check(type(item['full_tokens']) is int and item['full_tokens']==item['retained_tokens'] and 0<item['full_tokens']<=256,'truncated or invalid capture prompt')
        self.held_tokens=sum(by_id[x['example_id']]['full_tokens'] for x in held)
        self.check(capture.get('canonical_fixture_bit_identical') is True,'missing direct/streamed canonical fixture identity')
        supa=capture['supa_identity']
        self.check(supa['model_class']=='Qwen3AbstainForCausalLM' and supa['trainable_parameters']==supa['backbone_gradients']==0 and digest_ok(supa['state_sha256_before']) and supa['checkpoint_files_before']==capture['supa_files'] and capture['supa_files']['model.safetensors']==self.config['checkpoint_sha256'],'Supa frozen/checkpoint identity invalid')
        self.check(supa['runtime_identity']=={'attention':'eager','device':'cpu','dtype':'float32','seed':1702,'threads':4,'torch':'2.12.1+cu126','transformers':'5.14.1','use_cache':False},'Supa capture profile mismatch')
        recordpath=REPO/'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/cross_scale_evidence/02_architecture/qwen4b.json'
        record=self.load(recordpath)
        weights=record['weights'] if isinstance(record['weights'],list) else record['weights']['files']
        expected={x['name']:x['sha256'] for x in weights}
        expected['config.json']=record['config']['sha256']
        donor=capture['donor_identity']
        self.check(donor['revision']==self.config['donor_revision']==record['revision'] and donor['files']==expected and donor['record_sha256']==self.files[str(recordpath)] and donor['qwen_implementation_sha256']==supa['qwen_implementation_sha256'],'donor registered identity mismatch')
        activation=within(self.local/'activations.pt',self.local,strict=True)
        print('Hashing the opaque activation artifact (no deserialization).',flush=True)
        self.check(self.file_hash(activation,self.local)==capture['activation_sha256'],'activation artifact changed')
        self.data_summary={'calibration_scenarios':128,'fit_scenarios':96,'heldout_scenarios':32,'heldout_tokens':self.held_tokens,'dataset_sha256':capture['dataset_manifest_sha256']}

    def validate_metrics(self, report, pair):
        i,j=pair
        self.check((report['supa_layer'],report['donor_layer'])==pair and report['config_sha256']==canonical(self.config),'metric pair/config identity mismatch')
        self.check(digest_ok(report['donor_state_before']) and report['donor_state_before']==report['donor_state_after'] and report['native_replay_bit_identical'] is True and report['trainable_backbone_parameters']==report['backbone_gradient_tensors']==0,'donor replay/frozen-state claims incomplete')
        self.check(set(report['objectives'])==set(OBJECTIVES),'missing objective cell')
        baseline_counts=None
        for objective in OBJECTIVES:
            cell=report['objectives'][objective]
            expected_steps=96 if objective in ('B2','B3') else 0
            history=cell['training_trace']
            self.check(type(cell['optimizer_steps']) is int and cell['optimizer_steps']==len(history)==expected_steps,'wrong optimizer step budget: '+objective)
            self.check(cell['bridge_coefficients']==2*1024*2560+1024+2560,'wrong bridge coefficient inventory')
            if expected_steps:
                self.check([x['step'] for x in history]==list(range(96)) and sorted(x['training_prompt_index'] for x in history)==list(range(96)),'optimizer trace must cover 96 steps and every fit prompt once')
                self.check(all(type(x['step']) is int and type(x['training_prompt_index']) is int and finite(x['loss']) and finite(x['delta_loss']) and x['delta_loss']>=0 for x in history),'invalid optimizer trace entries')
                order=[x['training_prompt_index'] for x in history]
                if self.training_order is None:self.training_order=order
                self.check(order==self.training_order,'optimizer prompt order differs across B2/B3 or pairs')
            held=cell['heldout']
            self.check(set(held)=={'up','down','delta','mapped_delta_to_supa_rms'} and finite(held['mapped_delta_to_supa_rms']) and held['mapped_delta_to_supa_rms']>=0,'invalid heldout metric groups/mapped RMS')
            for group in ('up','down','delta'):
                strata=held[group]
                self.check(set(strata)==set(STRATA),'missing required token strata')
                counts={key:strata[key]['tokens'] for key in STRATA}
                self.check(all(type(x) is int and x>=0 for x in counts.values()) and counts['all_tokens']==self.held_tokens and counts['final_prompt_token']==32 and counts['ordinary_tokens']>0 and counts['ordinary_tokens']+counts['special_tokens']==counts['all_tokens'],'heldout token counts do not match frozen capture')
                if baseline_counts is None:baseline_counts=counts
                self.check(counts==baseline_counts,'token strata differ within pair')
                if self.stratum_counts is None:self.stratum_counts=counts
                self.check(counts==self.stratum_counts,'token strata differ across the 96 cells')
                for name,values in strata.items():
                    if not values['tokens']:
                        self.check(name=='special_tokens' and values=={'tokens':0},'fabricated empty stratum metrics')
                        continue
                    self.check(set(values)=={'tokens',*METRIC_KEYS} and all(finite(values[key]) for key in METRIC_KEYS),'missing or nonfinite scalar metric')
                    self.check(-1.000001<=values['mean_token_cosine']<=1.000001 and all(values[key]>=0 for key in METRIC_KEYS[1:]),'metric violates preregistered finite bounds')

    def audit(self):
        self.load_registration_and_capture()
        self.collect_full_inventory()
        self.verify_capture_data()
        self.training_order=None
        self.stratum_counts=None
        reports=[]
        donor_states={}
        for pair in PAIRS:
            directory,terminal=self.completed[pair]
            self.check(terminal['capture_sha256']==self.inventory['capture']['capture_sha256'] and terminal['activation_sha256']==self.capture['activation_sha256'] and terminal['dataset_manifest_sha256']==self.capture['dataset_manifest_sha256'],'pair capture/activation/data binding differs')
            self.check(terminal['donor_identity']==self.capture['donor_identity'] and terminal['supa_identity']==self.capture['supa_identity'],'pair checkpoint/model identity differs from capture')
            metricpath=directory/'metrics.json'
            self.check(self.file_hash(metricpath)==terminal['metrics_sha256'],'metrics file hash mismatch')
            bridgepath=within(terminal['bridge_path'],self.local,strict=True)
            expected=self.local/'pairs'/f'S{pair[0]}_D{pair[1]}'/directory.name/'bridges.pt'
            self.check(bridgepath==expected.resolve(),'bridge path does not belong to its pair/attempt')
            self.check(self.file_hash(bridgepath,self.local)==terminal['bridge_sha256'],'bridge file hash mismatch')
            report=self.load(metricpath)
            self.validate_metrics(report,pair)
            if pair[1] not in donor_states:donor_states[pair[1]]=report['donor_state_before']
            self.check(donor_states[pair[1]]==report['donor_state_before'],'same donor depth has different state across Supa pairs')
            reports.append(report)
        assert_full_coverage(reports)
        self.check(sum(len(row['objectives']) for row in reports)==96,'matrix is not exactly 96 cells')
        aggregate_dir,aggregate=self.aggregate
        self.check(aggregate['dataset_manifest_sha256']==self.capture['dataset_manifest_sha256'],'aggregate dataset binding mismatch')
        matrixpath=aggregate_dir/'matrix.json'
        self.check(self.file_hash(matrixpath)==aggregate['matrix_sha256'],'aggregate matrix hash mismatch')
        matrix=self.load(matrixpath)
        self.check(set(matrix)=={'pairs','sources'} and matrix['pairs']==reports,'aggregate matrix differs from the verified 16 pair reports')
        sources=[{key:item[key] for key in ('path','status','manifest_sha256')} for item in self.inventory['pairs']]
        self.check(matrix['sources']==sources,'aggregate omitted/changed a failed, abandoned or completed attempt')
        saved=self.load(aggregate_dir/'selection.json')
        self.check(saved==aggregate['selection'],'saved selection differs from aggregate receipt')
        reconstructed=reconstruct_selection(reports,self.config['mechanism_gate'])
        self.selection=reconstructed
        self.check(saved==reconstructed,'saved selection differs from independent exact gates/ranking','SELECTION_MISMATCH')
        # Recheck every byte-hashed input at completion, detecting concurrent edits.
        for path,expected in tuple(self.files.items()):
            self.check(sha(path)==expected,'input changed during audit: '+path)
        return {'status':'PASS','run_id':self.run_id,'matrix_pairs':16,'objective_cells':96,
            'candidate_gate_records':48,'selection_recomputed':True,'selection':reconstructed,
            'registration_contract_sha256':self.registration['registration_sha256'],
            'embedded_registration_sha256':canonical(self.registration),
            'config_canonical_sha256':canonical(self.config),'capture_sha256':self.inventory['capture']['capture_sha256'],
            'data':self.data_summary,'stratum_counts':self.stratum_counts,
            'optimizer_steps_per_B2_B3':96,'optimizer_steps_other_objectives':0,
            'donor_state_by_layer':donor_states,'aggregate_directory':str(aggregate_dir)}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True,help='Original CPU alignment run ID, e.g. alignment-cpu-v2-002')
    parser.add_argument('--output-dir',required=True,type=pathlib.Path,help='New absolute or repository-relative directory strictly under cross_scale/evidence/07_independent_audit; never reused')
    args=parser.parse_args(argv)
    output_arg=args.output_dir if args.output_dir.is_absolute() else REPO/args.output_dir
    try:
        output=exclusive_output(output_arg)
    except (AuditError,OSError) as error:
        print(str(error),file=sys.stderr)
        return 2
    auditor=None
    began=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        auditor=MatrixAuditor(args.run_id)
        result=auditor.audit()
        exit_code=0
    except (AuditError,OSError,KeyError,TypeError,ValueError) as error:
        result={'status':'REJECTED','run_id':args.run_id,'error_type':type(error).__name__,
            'error_code':getattr(error,'code','MALFORMED_EVIDENCE'),'message':str(error),
            'full_matrix_and_aggregate_present':bool(auditor and auditor.full_matrix_ready),
            'selection_recomputed':bool(auditor and auditor.selection is not None)}
        if auditor and auditor.selection is not None:
            result['recomputed_selection']=auditor.selection
        exit_code=1
    result.update(schema='independent_cross_scale_alignment_v2_audit_v1',
        start_utc=began,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        script_sha256=sha(__file__),audit_checks=auditor.checks if auditor else 0,
        models_loaded=0,torch_imported='torch' in sys.modules,programme_helpers_imported=False,
        checkpoint_weight_files_opened=False,bridge_activation_deserialization=False,
        generation_performed=False,git_commands_run=False,limitations=LIMITATIONS)
    write_json(output/'attempt_inventory.json',auditor.inventory if auditor else {})
    write_json(output/'input_hashes.json',auditor.files if auditor else {})
    result['attempt_inventory_sha256']=sha(output/'attempt_inventory.json')
    result['input_hashes_sha256']=sha(output/'input_hashes.json')
    write_json(output/'audit_receipt.json',result)
    print(json.dumps({'status':result['status'],'audit_receipt':str(output/'audit_receipt.json'),
        'selection_recomputed':result.get('selection_recomputed',False),'error_code':result.get('error_code')},indent=2))
    return exit_code

if __name__=='__main__':
    raise SystemExit(main())
