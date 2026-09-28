"""Synthetic tokenizer-boundary controls; no pretrained tokenizer or model."""
from pathlib import Path
from unittest.mock import patch
import unittest
from experiments.post_training_mimo_centralized_v1.scripts import tokenize_language_eval as t


class FakeTokenizer:
    eos_token_id = 151645
    eos_token = "<|im_end|>"
    def encode(self, text, add_special_tokens=False):
        ids=[]
        while text:
            if text.startswith(self.eos_token):
                ids.append(self.eos_token_id);text=text[len(self.eos_token):]
            else:
                ids.append(ord(text[0]));text=text[1:]
        return ids
    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.eos_token if x==self.eos_token_id else chr(x) for x in ids)
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        if add_generation_prompt:
            result=''.join(m['role']+':'+m['content']+'\n' for m in messages)
            result+='<|im_start|>assistant\n<think>\n\n</think>\n\n'
        else:
            result=self.apply_chat_template(messages[:-1])+messages[-1]['content']+self.eos_token+'\n'
        return self.encode(result) if tokenize else result


def world():
    return {'parent_id':'p','role':'DEV','stratum':'clarification',
            'messages':[{'role':'user','content':'Red or blue?'},{'role':'assistant','content':'What matters?'},
                        {'role':'user','content':'Use blue; just confirm.'}],
            'dev_exemplar':'Blue it is.','cal_controls':[]}


class LanguageEvalTokens(unittest.TestCase):
    def test_wrong_interpreter_stops_before_any_bank_or_tokenizer_read(self):
        with patch.object(t.sys,'version_info',(3,13,2)),patch.object(t.data,'read_json') as read,patch.object(t.common,'load_tokenizer') as load:
            with self.assertRaisesRegex(ValueError,'pinned Python 3.11.15'):
                t.profile(Path('missing-bank'),Path('missing-pin'),Path('nonexistent-new-token-profile'))
            read.assert_not_called();load.assert_not_called()

    def test_complete_context_and_exact_final_eos(self):
        row=world();out=t.tokenize_world(row,FakeTokenizer())
        self.assertEqual(out['prior_authored_assistant_turns'],1)
        self.assertIn('What matters?',FakeTokenizer().decode(out['prompt_token_ids']))
        self.assertEqual(FakeTokenizer().decode(out['references'][0]['completion_token_ids']),'Blue it is.<|im_end|>')
        self.assertFalse(out['training_eligibility']);self.assertFalse(out['native_qualified'])

    def test_caps_reject_without_truncation(self):
        for field in ('prompt','target'):
            row=world()
            if field=='prompt':row['messages'][0]['content']='x'*897
            else:row['dev_exemplar']='x'*128
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'over cap'):
                t.tokenize_world(row,FakeTokenizer())

    def test_closed_roles_system_messages_and_embedded_eos_refused(self):
        row=world();row['role']='CHECK'
        with self.assertRaises(ValueError):t.tokenize_world(row,FakeTokenizer())
        row=world();row['messages'][0]['role']='system'
        with self.assertRaises(ValueError):t.tokenize_world(row,FakeTokenizer())
        row=world();row['dev_exemplar']='Yes.<|im_end|>Again'
        with self.assertRaises(ValueError):t.tokenize_world(row,FakeTokenizer())

    def test_cal_requires_all_six_unqualified_controls(self):
        row=world();row.update(role='CAL',dev_exemplar=None)
        row['cal_controls']=[{'control_id':key,'response':'Candidate '+key+'.','qualified_reference_scores':None}
            for key in ('positive_a','positive_b','adequacy','relevance','fluency','role_consistency')]
        out=t.tokenize_world(row,FakeTokenizer());self.assertEqual(len(out['references']),6)
        row['cal_controls'][0]['qualified_reference_scores']={'adequacy':2}
        with self.assertRaises(ValueError):t.tokenize_world(row,FakeTokenizer())

    def test_template_boundary_drift_refused(self):
        class Broken(FakeTokenizer):
            def apply_chat_template(self,*args,**kwargs):
                value=super().apply_chat_template(*args,**kwargs)
                if not kwargs.get('add_generation_prompt',True) and not kwargs.get('tokenize',False):value+='extra'
                return value
        with self.assertRaisesRegex(ValueError,'boundary drift'):t.tokenize_world(world(),Broken())


if __name__=='__main__':unittest.main()
