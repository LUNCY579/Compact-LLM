import unittest
from experiments.cheap_quality_shortcuts_v1.study import kind,predict,controls
from experiments.post_training_mimo_centralized_v1.scripts import bridge_data as B

class ControlsTest(unittest.TestCase):
    def test_public_contract_unknown_rejected(self):
        with self.assertRaises(ValueError):kind('unrelated input')
    def test_predictor_cannot_receive_gold_or_family(self):
        self.assertEqual(predict('first_number','Compute 17 + 22. Answer with an integer.',{}),17)
    def test_initial_registers_ignore_update(self):
        self.assertEqual(predict('initial_state','Registers: A=1, B=-2, C=3. Apply 1. A = B. Answer with the final integer list in order [A,B,C].',{}),[1,-2,3])
    def test_duplicate_and_torn_are_not_valid(self):
        for label,text,expected in controls('{"answer":12}'):
            if label in ('duplicate_key','torn','fenced','extra_key'):
                with self.assertRaises((ValueError,TypeError)):B.parse_answer(text)
    def test_equivalent_controls_preserve_semantics(self):
        for label,text,expected in controls('{"answer":[]}'):
            if expected:self.assertEqual(B.parse_answer(text),[])
    def test_wrong_answer_differs_for_legal_empty(self):
        for target in ('{"answer":[]}','{"answer":""}','{"answer":0}'):
            wrong=next(text for label,text,_ in controls(target) if label=='wrong_answer')
            self.assertNotEqual(B.parse_answer(wrong),B.parse_answer(target))
if __name__=='__main__':unittest.main()
