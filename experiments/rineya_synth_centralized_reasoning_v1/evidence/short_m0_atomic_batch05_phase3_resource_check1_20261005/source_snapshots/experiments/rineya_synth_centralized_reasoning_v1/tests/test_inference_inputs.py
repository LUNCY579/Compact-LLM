"""Prompt-only boundaries and exact token envelope; no model or GPU."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference_inputs as inputs
from rs_contract import digest


class Tokenizer:
    def __init__(self):
        self._template = SimpleNamespace(render=lambda **kw: kw["messages"][-1]["content"] + inputs.ASSISTANT_PREFIX)
        self._engine = SimpleNamespace(encode=lambda text, **kw: SimpleNamespace(ids=list(range(int(text.split(":")[1].split("<")[0])))))
    def verify_binding(self): return {"fixture": "CPU_TEST_ONLY", "eos": 151645}


def packet(lengths):
    rows = []
    for index, length in enumerate(lengths):
        messages = [{"role": "system", "content": "Return a final JSON answer."},
                    {"role": "user", "content": f"case{index}:{length}"}]
        rows.append({"row_id": digest(messages), "messages": messages, "prompt_sha256": digest(messages)})
    return {"rows": rows, "native_launch_authorized": False, "confirmation_eligible": False}


class InputTests(unittest.TestCase):
    def test_exact480_prompt_plus32_fits_without_padding_or_oracle(self):
        source = packet([10, 480])
        result = inputs.prepare(source, Tokenizer())
        self.assertEqual("CPU_INPUTS_PREPARED_NATIVE_UNADMITTED", result["status"])
        self.assertEqual(480, inputs.validate(result, source, Tokenizer())["maximum_prompt_tokens"])
        self.assertFalse(result["native_launch_authorized"])
        self.assertFalse(result["oracle_inputs_read"])
        self.assertEqual([1] * 480, result["rows"][1]["attention_mask"])

    def test_one_overlong_row_blocks_whole_packet_and_preserves_it(self):
        source = packet([479, 481, 572])
        result = inputs.prepare(source, Tokenizer())
        self.assertEqual("BLOCKED_PROMPT_ENVELOPE", result["status"])
        self.assertEqual(3, len(result["rows"]))
        self.assertEqual(572, len(result["rows"][2]["prompt_ids"]))
        self.assertEqual(2, len(result["overlong_row_ids"]))
        self.assertEqual(0, result["rows_dropped"])
        self.assertEqual(0, result["rows_truncated"])
        with self.assertRaisesRegex(ValueError, "envelope blocked"):
            inputs.validate(result, source, Tokenizer())

    def test_teacher_answers_private_metadata_assistant_turns_and_ids_rejected(self):
        source = packet([10])
        for change in (lambda x: x["rows"][0].update(answer=[1]),
                       lambda x: x.update(oracle={"answer": [1]}),
                       lambda x: x["rows"][0].update(stratum="solved_to_UNKNOWN"),
                       lambda x: x["rows"][0]["messages"].append({"role": "assistant", "content": '{"answer":[1]}'}),
                       lambda x: x["rows"].append(copy.deepcopy(x["rows"][0])),
                       lambda x: x["rows"][0].update(row_id="not-a-hash")):
            altered = copy.deepcopy(source)
            change(altered)
            with self.assertRaises(ValueError): inputs.prepare(altered, Tokenizer())

    def test_prompt_or_re_signed_token_order_forgery_cannot_pass(self):
        source = packet([10, 12])
        result = inputs.prepare(source, Tokenizer())
        altered_source = copy.deepcopy(source)
        altered_source["rows"][0]["messages"][1]["content"] = "changed:10"
        with self.assertRaisesRegex(ValueError, "identity"):
            inputs.prepare(altered_source, Tokenizer())
        for mutate in (lambda x: x["rows"].reverse(),
                       lambda x: x["rows"][0]["prompt_ids"].pop(),
                       lambda x: x.update(native_launch_authorized=True)):
            altered = copy.deepcopy(result)
            mutate(altered)
            altered["sha256"] = digest({k: v for k, v in altered.items() if k != "sha256"})
            with self.assertRaisesRegex(ValueError, "binding drift"):
                inputs.validate(altered, source, Tokenizer())

    def test_padding_boolean_out_of_vocabulary_and_bad_template_rejected(self):
        for values in ([151643], [True], [151936], [], [-1]):
            tokenizer = Tokenizer()
            tokenizer._engine.encode = lambda *args, **kw: SimpleNamespace(ids=values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                inputs.prepare(packet([10]), tokenizer)
        tokenizer = Tokenizer()
        tokenizer._template.render = lambda **kw: "wrong prefix"
        with self.assertRaisesRegex(ValueError, "prefix"):
            inputs.prepare(packet([10]), tokenizer)

    def test_tokenizer_drift_mid_packet_is_rejected(self):
        tokenizer = Tokenizer()
        counter = [0]
        def binding():
            counter[0] += 1
            return {"revision": counter[0]}
        tokenizer.verify_binding = binding
        with self.assertRaisesRegex(ValueError, "binding drift"):
            inputs.prepare(packet([10]), tokenizer)


if __name__ == "__main__": unittest.main()
