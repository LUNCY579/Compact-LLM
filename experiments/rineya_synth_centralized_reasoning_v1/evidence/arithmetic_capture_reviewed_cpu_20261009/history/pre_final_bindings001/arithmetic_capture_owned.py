"""Own arithmetic purpose; exact original Job/watchdog/worker code and checks."""
from pathlib import Path
from types import ModuleType
import ast
import hashlib
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = next(p for p in HERE.parents if p.name == "rineya_synth_centralized_reasoning_v1")
SOURCE_PINS = {"short_collection_worker.py": "f0ee2c5fd92397ee1ff8cbbd55596e50c2251e9c313542f493fa3609162b7e5b",
               "short_collection_supervisor.py": "f0d178cda0b20a804791ea9c95e89040d9d8dff38437f12ce0cbdddbaff9be09"}


def _selected(name):
    import arithmetic_capture_contract as contract
    contract.integrated()  # Staging refuses before imports, reads or native APIs.
    if str(PACKAGE) not in sys.path: sys.path.insert(0, str(PACKAGE))
    path = PACKAGE / name; raw = path.read_bytes()
    contract.require(name in SOURCE_PINS and hashlib.sha256(raw).hexdigest() == SOURCE_PINS[name], "exact original owned collector/supervisor bytes")
    module = ModuleType("selected_arithmetic_" + name[:-3]); module.__file__ = str(path)
    tree = ast.parse(raw)
    if name == "short_collection_supervisor.py":
        class ArithmeticScope(ast.NodeTransformer):
            def visit_BinOp(self, node):
                if isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name) and node.left.id == "PACKAGE" \
                        and isinstance(node.right, ast.Constant) and node.right.value == "short_collection_worker.py":
                    return ast.copy_location(ast.Name(id="ARITHMETIC_WORKER_PATH", ctx=ast.Load()), node)
                return self.generic_visit(node)
            def visit_Constant(self, node):
                if node.value == "short_M0_raw_collection_one_fixed_batch_only":
                    return ast.copy_location(ast.Constant("arithmetic_M0_raw_collection_one_fixed_batch_only"), node)
                return node
        tree = ast.fix_missing_locations(ArithmeticScope().visit(tree))
        module.ARITHMETIC_WORKER_PATH = HERE / "arithmetic_capture_owned.py"
    exec(compile(tree, str(path), "exec"), module.__dict__)
    view = contract._original()
    view.preflight = contract.preflight; view.build_scope = contract.build_scope; view.integrated = contract.integrated
    module.contract = view  # Preserve native/qualified/helper attributes used by original worker.
    return module


def dispatch(config_path, output_directory):
    return _selected("short_collection_supervisor.py").dispatch(config_path, output_directory)


def worker(config_path, output_directory):
    return _selected("short_collection_worker.py").run(config_path, output_directory)


if __name__ == "__main__":
    import arithmetic_capture_contract as contract
    contract.integrated()
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); worker(args.config, args.output_dir)
