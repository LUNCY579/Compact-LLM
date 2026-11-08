import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.post_training_mimo_centralized_v1.scripts import e1_data as d


def row():
    ids=[151645,17,18,151645];p=2;c=2
    return dict(row_id='r',parent_id='p',family='f',role='TRAIN',training_role_eligible=True,
        input_ids=ids,labels=[-100,-100,18,151645],attention_mask=[1]*4,completion_mask=[0,0,1,1],
        prompt_tokens=p,completion_tokens=c,nonpadding_tokens=4,padding_tokens=0,completion_eos_id=151645,
        input_token_ids_sha256=d.digest(ids),tokenized_prompt_sha256=d.digest(ids[:p]),
        completion_token_ids_sha256=d.digest(ids[p:]),source_row_sha256='a'*64)


class BindingTests(unittest.TestCase):
    def test_prompt_eos_not_masked_by_value_and_terminal_eos_supervised(self):
        r=d.validate_array(row(),'acquisition');self.assertEqual(r['completion_tokens'],2)
        self.assertEqual(set(r),{'key','component','role','parent_id','nonpadding_tokens','completion_tokens'})

    def test_masks_counts_padding_and_roles_rejected(self):
        mutations={'role':'DEV','labels':[-100,17,18,151645],'attention_mask':[1,1,1,0],
                   'completion_mask':[0,1,1,1],'padding_tokens':1,'prompt_tokens':True,'completion_tokens':3}
        for key,value in mutations.items():
            with self.subTest(key=key),self.assertRaises(ValueError):d.validate_array({**row(),key:value},'acquisition')

    def test_rehashed_internal_eos_still_rejected(self):
        r=row();r['input_ids'][2]=151643;r['labels'][2]=151643
        r['input_token_ids_sha256']=d.digest(r['input_ids']);r['completion_token_ids_sha256']=d.digest(r['input_ids'][2:])
        with self.assertRaises(ValueError):d.validate_array(r,'acquisition')

    def test_token_boolean_and_hash_drift_rejected(self):
        r=row();r['input_ids'][0]=True
        with self.assertRaises(ValueError):d.validate_array(r,'acquisition')
        with self.assertRaises(ValueError):d.validate_array({**row(),'input_token_ids_sha256':'0'*64},'acquisition')

    def test_duplicate_json_and_closed_path_rejected(self):
        with self.assertRaises(ValueError):d.parse('{"x":1,"x":2}')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for path in ('../escape','bank/CHECK.jsonl','bank/fresh_sealed/TRAIN.jsonl','bank/FINAL/TRAIN.tokenized.jsonl'):
                with self.assertRaises(ValueError):d.safe_path(root,path)

    def test_resolved_closed_bank_rejected_before_content_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();relative='bank/TRAIN.tokenized.jsonl'
            alias=root/relative;closed=root/'FINAL'/'TRAIN.tokenized.jsonl'
            resolve=Path.resolve
            def target(path,*args,**kwargs):
                return closed if path==alias else resolve(path,*args,**kwargs)
            with patch.object(Path,'resolve',target):
                with self.assertRaisesRegex(ValueError,'closed bank'):d.safe_path(root,relative)

    def test_read_json_identity_binds_exact_parsed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json';raw=b'{ "key": 2 }\r\n';path.write_bytes(raw)
            value,identity=d.read_json(path);path.write_bytes(b'{}')
            self.assertEqual(value,{'key':2})
            self.assertEqual(identity,{'sha256':d.hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})

    def test_quota_cannot_be_satisfied_by_duplicate_rows(self):
        r={'row_id':'r','parent_id':'p','family':'f','nonpadding_tokens':1,'completion_tokens':1}
        with self.assertRaises(ValueError):d.validate_population([r]*4096,'acquisition')


if __name__=='__main__':unittest.main()
