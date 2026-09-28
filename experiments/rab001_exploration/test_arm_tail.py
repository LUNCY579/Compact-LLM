"""Small, model-free checks of the actual arm builder's freeze boundary.

Extract the function AST to avoid importing torch or starting a model alongside a run.
These are mechanical checks, not source qualification or capability evidence.
"""
import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name('arm_dose.py')


class Parameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class Model:
    def __init__(self):
        names = ['embed.weight', 'norm.weight']
        names += [f'encoder.{i}.{part}.weight' for i in range(28)
                  for part in ('attn.q_proj', 'attn.k_proj', 'ffn.down_proj', 'input_norm')]
        self.named = {name: Parameter() for name in names}

    def parameters(self):
        return iter(self.named.values())

    def named_parameters(self):
        return iter(self.named.items())


def extracted_builder(tail):
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
    model = Model()
    namespace = {'MODE': 'B28', 'BLOCK_TAIL': tail, 'SRC': 'fixture', 're': re,
                 'run': SimpleNamespace(_load=lambda source, device: (model, None))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), 'exec'), namespace)
    return namespace['build']('fixture')[0]


class TailBoundaryTests(unittest.TestCase):
    def test_tail_boundaries_and_nonblock_freezing(self):
        for tail in (1, 8, 12, 16, 20, 28):
            with self.subTest(tail=tail):
                model = extracted_builder(tail)
                selected = {int(n.split('.')[1]) for n, p in model.named_parameters()
                            if p.requires_grad and n.startswith('encoder.')}
                self.assertEqual(selected, set(range(28 - tail, 28)))
                self.assertFalse(model.named['embed.weight'].requires_grad)
                self.assertFalse(model.named['norm.weight'].requires_grad)

    def test_full_model_includes_embedding_and_final_norm(self):
        self.assertTrue(all(p.requires_grad for p in extracted_builder(None).parameters()))

    def test_complete_model_metadata_distinguishes_tail28_from_full(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8-sig'))
        expressions = [v for node in ast.walk(tree) if isinstance(node, ast.Dict)
                       for k, v in zip(node.keys, node.values)
                       if isinstance(k, ast.Constant) and k.value == 'complete_model']
        self.assertEqual(len(expressions), 1)
        code = compile(ast.Expression(expressions[0]), str(SCRIPT), 'eval')
        for mode, tail, expected in [('B28', None, True), ('B28', 28, False),
                                     ('B28', 12, False), ('P', None, False)]:
            self.assertEqual(eval(code, {'MODE': mode, 'BLOCK_TAIL': tail}), expected)


if __name__ == '__main__':
    unittest.main()
