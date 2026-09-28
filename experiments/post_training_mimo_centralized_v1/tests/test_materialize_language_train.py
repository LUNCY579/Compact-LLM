import copy
import unittest
from experiments.post_training_mimo_centralized_v1.scripts import materialize_language_train as m
from experiments.post_training_mimo_centralized_v1.tests.test_tokenize_language_candidates import ConversationTokenizer


class LanguageArrayTests(unittest.TestCase):
    def setUp(self):
        self.row={"id":"train1","metadata":{"split":"train","base_template_family":"f","reply_class":"c"},
                  "messages":[{"role":"user","content":"Old request"},{"role":"assistant","content":"Old reply"},
                              {"role":"user","content":"New request"},{"role":"assistant","content":"New reply"}]}
        self.tok=ConversationTokenizer()
        self.profile=m.measured.tokenize_row(self.row,self.tok)

    def test_arrays_match_prior_measurement_and_only_last_reply(self):
        r=m.encode_row(self.row,self.profile,self.tok)
        self.assertEqual(r['completion_tokens'],len('New reply')+1)
        self.assertEqual(r['labels'][:r['prompt_tokens']],[-100]*r['prompt_tokens'])
        self.assertEqual(r['labels'][r['prompt_tokens']:],r['input_ids'][r['prompt_tokens']:])
        self.assertEqual(r['role'],'TRAIN');self.assertFalse(r['training_eligibility'])

    def test_changed_profile_hash_or_count_refused(self):
        for field,value in [('prompt_token_ids_sha256','0'*64),('completion_tokens',99)]:
            changed={**self.profile,field:value}
            with self.assertRaises(ValueError):m.encode_row(self.row,changed,self.tok)

    def test_changed_source_or_role_refused(self):
        for mode in ['target','role']:
            row=copy.deepcopy(self.row)
            if mode=='target':row['messages'][-1]['content']='Altered'
            else:row['metadata']['split']='DEV'
            with self.assertRaises(ValueError):m.encode_row(row,self.profile,self.tok)


if __name__=='__main__':unittest.main()
