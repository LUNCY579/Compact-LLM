import unittest
from experiments.post_training_mimo_centralized_v1.scripts import audit_language_fingerprints as a


class HistoricalLanguageControls(unittest.TestCase):
    def test_complete_multiturn_context_is_fingerprinted(self):
        row={'parent_id':'p','role':'DEV','messages':[{'role':'user','content':'Which?'},
            {'role':'assistant','content':'Blue.'},{'role':'user','content':'Why?'}]}
        self.assertEqual(len(a.compare([row],{a.history.digest(row['messages'])})),1)
        self.assertEqual(a.compare([row],{a.history.digest(row['messages'][-1:])}),[])

    def test_closed_duplicate_or_target_containing_context_refused(self):
        row={'parent_id':'p','role':'CAL','messages':[{'role':'user','content':'Hello.'}]}
        for rows in ([row,row],[{**row,'role':'FINAL'}],[{**row,'messages':row['messages']+[{'role':'assistant','content':'Hi.'}]}]):
            with self.assertRaises(ValueError):a.compare(rows,set())


if __name__=='__main__':unittest.main()
