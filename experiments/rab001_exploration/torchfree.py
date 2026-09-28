"""Assert that an analysis module is torch-free, in a way that survives suite ordering.

`frontier_report` and `headroom_audit` recompute panel evidence with no model, tokenizer,
GPU or torch. That is a real design property and worth a test: it is what lets the evidence
be re-derived on a CPU-only box, and it is the reason those two modules can be trusted not
to quietly depend on the training stack.

Both tests asserted it as `self.assertNotIn("torch", sys.modules)` at the end of the method.
Run alone, that works. Run inside the full suite it **always fails**, because some earlier
test has already imported torch into the shared process -- so the assertion reports a
property of the *test runner's history*, not of the module under test. The result was two
permanently red tests that pass in isolation, which is the worst of both worlds: it teaches
whoever reads the output to ignore red.

The property is checked here where it actually lives, in two halves:

  * `assert_import_is_clean` imports the module in a **fresh interpreter** and asserts the
    banned modules are absent there. This is strictly stronger than the original: it holds
    no matter what the rest of the suite did, and it catches a top-level `import torch`
    that the in-process check would miss once torch was already loaded.
  * `no_new_imports` is a context manager asserting the guarded call imported nothing new.
    That covers a lazy `import torch` inside a function body, which the fresh-interpreter
    import check cannot see.

Together they cover both routes without depending on test order.
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BANNED = ('torch', 'transformers', 'experiments.architecture_benchmark_001.run')


def assert_import_is_clean(case, module: str, banned: tuple[str, ...] = BANNED) -> None:
    """Import `module` in a fresh interpreter; fail if any banned module came with it."""
    code = (
        'import importlib, json, sys\n'
        f'importlib.import_module({module!r})\n'
        f'print(json.dumps([n for n in {list(banned)!r} if n in sys.modules]))\n'
    )
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          cwd=str(REPO))
    if done.returncode != 0:
        case.fail(f'importing {module} in a fresh interpreter failed:\n{done.stderr.strip()}')
    line = done.stdout.strip().splitlines()[-1]
    loaded = json.loads(line)
    case.assertEqual(loaded, [], f'{module} pulled in {loaded} at import time; it is '
                                f'supposed to recompute evidence without the training stack')


def loaded(banned: tuple[str, ...] = BANNED) -> set[str]:
    """Which banned modules are present right now. Pair with `assert_no_new`."""
    return {name for name in banned if name in sys.modules}


def assert_no_new(case, before: set[str], banned: tuple[str, ...] = BANNED) -> None:
    """Assert nothing banned appeared since `before`. For bodies too long to wrap."""
    new = loaded(banned) - before
    case.assertEqual(new, set(), f'this analysis path imported {sorted(new)}; it must stay '
                                 f'free of the training stack')


@contextlib.contextmanager
def no_new_imports(case, banned: tuple[str, ...] = BANNED):
    """Assert the guarded block imported none of `banned` that was not already present."""
    before = loaded(banned)
    yield
    assert_no_new(case, before, banned)
