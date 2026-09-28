import copy
import unittest
from experiments.post_training_mimo_centralized_v1.scripts import tokenize_language_candidates as t
from experiments.post_training_mimo_centralized_v1.tests.test_tokenize_bridge import FakeTokenizer


class ConversationTokenizer(FakeTokenizer):
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,enable_thinking):
        text=""
        for i,m in enumerate(messages):
            text+="<|im_start|>"+m["role"]+"\n"
            if m["role"]=="assistant" and i==len(messages)-1:
                text+="<think>\n\n</think>\n\n"
            text+=m["content"]+self.eos_token+"\n"
        if add_generation_prompt:text+="<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.encode(text) if tokenize else text


class LanguageMaskControls(unittest.TestCase):
    def setUp(self):
        self.row={"id":"train-1","metadata":{"split":"train","base_template_family":"f","reply_class":"c"},
                  "messages":[{"role":"system","content":"Be helpful."},{"role":"user","content":"Old request"},
                              {"role":"assistant","content":"Old answer"},{"role":"user","content":"New request"},
                              {"role":"assistant","content":"New answer"}]}

    def test_only_last_target_is_supervised(self):
        out=t.tokenize_row(self.row,ConversationTokenizer())
        self.assertEqual(out["prior_assistant_turns"],1)
        self.assertTrue(out["all_prior_turns_masked"])
        self.assertEqual(out["supervised_positions"],len("New answer")+1)
        self.assertFalse(out["training_eligibility"])

    def test_closed_split_and_malformed_role_refused(self):
        for field in ("split","role"):
            row=copy.deepcopy(self.row)
            if field=="split":row["metadata"]["split"]="DEV"
            else:row["messages"][-2]["role"]="assistant"
            with self.assertRaises(ValueError):t.tokenize_row(row,ConversationTokenizer())

    def test_target_eos_and_oversize_refused(self):
        for text in ("bad<|im_end|>target","a"*129):
            row=copy.deepcopy(self.row);row["messages"][-1]["content"]=text
            with self.assertRaises(ValueError):t.tokenize_row(row,ConversationTokenizer())

    def test_prompt_tail_boundary_mutation_refused(self):
        class Bad(ConversationTokenizer):
            def apply_chat_template(self,messages,**kwargs):
                result=super().apply_chat_template(messages,**kwargs)
                if not kwargs["add_generation_prompt"] and not kwargs["tokenize"]:result+="oops"
                return result
        with self.assertRaises(ValueError):t.tokenize_row(self.row,Bad())


if __name__=="__main__":unittest.main()
