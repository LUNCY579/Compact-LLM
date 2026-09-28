import unittest
from experiments.post_training_mimo_centralized_v1.scripts import audit_language_eval as a


class LanguageExclusionControls(unittest.TestCase):
    def test_normalized_matches_do_not_hide_exact_text_difference(self):
        first=[{'role':'user','content':'  Use BLUE. '}]
        second=[{'role':'user','content':'use blue.'}]
        x,y=a.prompt_keys(first),a.prompt_keys(second)
        self.assertNotEqual(x['exact_conversation'],y['exact_conversation'])
        self.assertEqual(x['normalized_conversation'],y['normalized_conversation'])

    def test_system_stripped_match_kept_separate(self):
        prompt=[{'role':'user','content':'Which page?'}]
        system=[{'role':'system','content':'Be helpful.'}]+prompt
        x,y=a.prompt_keys(prompt),a.prompt_keys(system)
        self.assertNotEqual(x['exact_conversation'],y['exact_conversation'])
        self.assertEqual(x['without_system'],y['without_system'])

    def test_prior_assistant_context_not_silently_dropped(self):
        prompt=[{'role':'user','content':'Help.'},{'role':'assistant','content':'Page A.'},
                {'role':'user','content':'Why?'}]
        changed=[dict(x) for x in prompt];changed[1]['content']='Page B.'
        x,y=a.prompt_keys(prompt),a.prompt_keys(changed)
        self.assertNotEqual(x['normalized_conversation'],y['normalized_conversation'])
        self.assertEqual(x['normalized_user_sequence'],y['normalized_user_sequence'])
        with self.assertRaises(ValueError):a.prompt_keys(prompt[:-1])

    def test_lexical_neighbours_bounded_and_reproducible(self):
        prompt=[{'role':'user','content':'Blue lantern cover.'}]
        training=[]
        for index,text in enumerate(('Blue lantern cover.','Blue lantern.','Cover board.','No overlapping nouns.')):
            messages=[{'role':'user','content':text}]
            training.append({'source':'TEST','id':str(index),'messages':messages,'words':a.wordset(messages),
                'source_row_sha256':'a'*64,'template_family':'synthetic'})
        result=a.neighbours(prompt,training)
        self.assertEqual(len(result),3);self.assertEqual(result[0]['id'],'0')
        self.assertEqual(result[0]['jaccard_wordset'],1)
        self.assertEqual(result,a.neighbours(prompt,list(reversed(training))))


if __name__=='__main__':unittest.main()
