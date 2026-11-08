"""Publisher controls with synthetic rows; actual bank/scorer integration is separate."""
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from experiments.post_training_mimo_centralized_v1.scripts import prepare_dev_binding as p


def binding():
    policy=[]; evaluator=[]
    for panel,(count,parents) in p.inputs.COUNTS.items():
        for index in range(count):
            if panel=='deterministic_retention':parent=index if index<128 else 128+(index-128)//2
            elif panel=='long_context':parent=index//8
            else:parent=index
            key=panel+str(index);ids=[11,12,13]
            policy.append({'row_id':key,'messages':[{'role':'user','content':'Synthetic prompt'}],
                'input_ids':ids,'attention_mask':[1,1,1],'prompt_tokens':3,'prompt_token_hash':p.inputs.digest(ids)})
            evaluator.append({'row_id':key,'parent_id':panel+str(parent),'family':'fixture','stratum':'fixture',
                'variant':str(index),'role':'DEV','panel':panel,'prompt_token_hash':p.inputs.digest(ids),
                'scoring_payload':{'schema_version':1,'panel':panel,'source_row_sha256':'a'*64,
                                   'source_row':{'target':'EVALUATOR_ONLY_SENTINEL'}}})
    result={'schema_version':1,'status':'BOUND_DEV_INPUTS_NOT_NATIVE_ADMISSION','policy_rows':policy,
        'evaluator_rows':evaluator,'bindings':{},'counts':{k:{'rows':v[0],'parents':v[1]} for k,v in p.inputs.COUNTS.items()},
        **dict.fromkeys(p.UNQUALIFIED,False)}
    result['data_sha256']=p.inputs.digest(result)
    return result


class PublisherControls(unittest.TestCase):
    @contextmanager
    def sandbox(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root=Path(temporary)/'repo';package=root/'experiments/post_training_mimo_centralized_v1'
            script=package/'scripts/prepare_dev_binding.py';script.parent.mkdir(parents=True)
            script.write_text('# Synthetic publisher source\n')
            cfg=package/p.CONFIG;cfg.parent.mkdir();cfg.write_bytes(b'{"fixture":true}\n')
            amendment=package/p.AMENDMENT;amendment.write_text('Synthetic amendment\n')
            datafile=root/'raw/DEV.jsonl';datafile.parent.mkdir();datafile.write_bytes(b'fixture\n')
            value=binding();value['bindings']={'raw/DEV.jsonl':p.inputs.identity(datafile.read_bytes())}
            value['data_sha256']=p.inputs.digest({k:v for k,v in value.items() if k!='data_sha256'})
            def git(*args):
                if args[1:]==('rev-parse','HEAD'):return 'c'*40+'\n'
                return ''
            stack.enter_context(patch.object(p,'__file__',str(script)))
            stack.enter_context(patch.object(p,'critical_paths',return_value=[script,cfg,amendment]))
            stack.enter_context(patch.object(p,'clean'))
            git_mock=stack.enter_context(patch.object(p,'git',side_effect=git))
            bind_mock=stack.enter_context(patch.object(p.inputs,'bind',return_value=value))
            stack.enter_context(patch.object(p.inputs,'_read',side_effect=lambda root,label:(root/label).read_bytes()))
            validator=stack.enter_context(patch.object(p.scorer,'validate_evaluator_row'))
            stack.enter_context(patch.object(p.scorer,'dependency_binding',return_value={'scope':'SYNTHETIC'}))
            stack.enter_context(patch.object(p.scorer,'validate_dependency_binding'))
            score=stack.enter_context(patch.object(p.scorer,'score_response',side_effect=AssertionError('must never score')))
            stack.enter_context(patch.object(p.enrollment,'scorer_binding',return_value={'scope':'SYNTHETIC'}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield {'root':root,'package':package,'script':script,'config':cfg,'data':datafile,'output':root/'out',
                   'binding':value,'bind':bind_mock,'validator':validator,'score':score,'git':git_mock}

    def test_write_once_receipts_and_separated_targets(self):
        with self.sandbox() as s:
            r=p.prepare(s['config'],s['output'])
            self.assertEqual((r['rows'],r['parent_groups'],r['prompt_tokens_total']),(832,544,2496))
            self.assertEqual(s['validator'].call_count,832);s['score'].assert_not_called()
            policy=(s['output']/'policy/DEV_INPUTS.jsonl').read_bytes()
            sidecar=(s['output']/'evaluation/DEV_ROWS.jsonl').read_bytes()
            self.assertNotIn(b'EVALUATOR_ONLY_SENTINEL',policy);self.assertIn(b'EVALUATOR_ONLY_SENTINEL',sidecar)
            for label,identity in r['files'].items():self.assertEqual(identity,p.inputs.identity((s['output']/label).read_bytes()))
            self.assertFalse(r['native_launch_authorized']);self.assertEqual(r['actual_response_records'],0)
            self.assertTrue(all(value is None for value in r['unresolved_native_registration'].values()))
            self.assertNotEqual(r['config_file_sha256'],r['config_canonical_sha256'])
            with self.assertRaisesRegex(ValueError,'write-once'):p.prepare(s['config'],s['output'])

    def test_host_validation_failure_never_publishes(self):
        with self.sandbox() as s:
            s['validator'].side_effect=ValueError('source invalid')
            with self.assertRaisesRegex(ValueError,'source invalid'):p.prepare(s['config'],s['output'])
            self.assertFalse(s['output'].exists())

    def test_authority_flag_and_policy_extra_field_rejected(self):
        for kind in ('grant','leak'):
            with self.sandbox() as s:
                b=s['binding']
                if kind=='grant':b['native_launch_authorized']=True
                else:b['policy_rows'][0]['target']='REFERENCE'
                b['data_sha256']=p.inputs.digest({k:v for k,v in b.items() if k!='data_sha256'})
                with self.assertRaises(ValueError):p.prepare(s['config'],s['output'])
                self.assertFalse(s['output'].exists())

    def test_source_config_or_data_edit_during_binding_rejected(self):
        for target in ('script','config','data'):
            with self.sandbox() as s:
                def mutate(*args):
                    s[target].write_bytes(s[target].read_bytes()+b' ')
                    return s['binding']
                s['bind'].side_effect=mutate
                with self.assertRaises(ValueError):p.prepare(s['config'],s['output'])
                self.assertFalse(s['output'].exists())

    def test_head_change_blocks_publication(self):
        with self.sandbox() as s:
            calls=0
            def changed(root,*args):
                nonlocal calls
                if args==('rev-parse','HEAD'):
                    calls+=1;return ('c' if calls==1 else 'd')*40+'\n'
                return ''
            s['git'].side_effect=changed
            with self.assertRaisesRegex(ValueError,'HEAD changed'):p.prepare(s['config'],s['output'])
            self.assertFalse(s['output'].exists())

    def test_fresh_import_preserves_environment_and_no_backends(self):
        source="import os,sys; before=dict(os.environ); from experiments.post_training_mimo_centralized_v1.scripts import prepare_dev_binding; assert before==dict(os.environ); assert not set(sys.modules).intersection({'torch','transformers','tokenizers','tensorflow','jax'})"
        subprocess.run([__import__('sys').executable,'-B','-c',source],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
