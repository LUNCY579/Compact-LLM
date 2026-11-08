"""Static validation of the CI workflow's inline Python steps.

The workflow embeds several `python - <<'PY'` heredocs that assert the evidence
and claim-scope contracts. Those scripts are otherwise only exercised when CI
actually runs, so a typo or a use-before-definition costs a full round trip. A
real `NameError: name 'expected' is not defined` shipped this way once.

These tests parse each heredoc and check it compiles and that no name is loaded
before it is bound.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tgsa-phase1.yml"

HEREDOC = re.compile(r"python - <<'PY'\n(.*?)\n\s*PY(?:\n|$)", re.S)


def _heredocs() -> list[tuple[int, str]]:
    text = WORKFLOW.read_text(encoding="utf-8")
    blocks: list[tuple[int, str]] = []
    for match in HEREDOC.finditer(text):
        raw = match.group(1)
        indent = min(
            (len(line) - len(line.lstrip()) for line in raw.splitlines() if line.strip()),
            default=0,
        )
        code = "\n".join(line[indent:] if len(line) > indent else line for line in raw.splitlines())
        blocks.append((text[: match.start()].count("\n") + 1, code))
    return blocks


def _locally_scoped_names(statement: ast.AST) -> set[str]:
    """Names bound by a nested scope inside one statement.

    Comprehension targets, loop targets, `with ... as` targets, lambda and
    function parameters, and walrus targets are all bound within the statement
    that introduces them, so a Load of one is not a use-before-definition.
    """

    scoped: set[str] = set()

    def _names(node: ast.AST) -> None:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name):
                scoped.add(inner.id)

    for node in ast.walk(statement):
        if isinstance(node, ast.comprehension):
            _names(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _names(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            _names(node.optional_vars)
        elif isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            optional = [arguments.vararg, arguments.kwarg]
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *[item for item in optional if item is not None],
            ):
                scoped.add(argument.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            scoped.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            _names(node.target)
    return scoped


def _bound_names(tree: ast.AST) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    bound.add(name.id)
    return bound


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repo-root workflow not present")
def test_every_inline_ci_script_compiles() -> None:
    blocks = _heredocs()
    assert blocks, "no inline python steps found in the workflow"
    for line_number, code in blocks:
        try:
            compile(code, f"{WORKFLOW.name}:{line_number}", "exec")
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{WORKFLOW.name}:{line_number} does not compile: {exc}")


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repo-root workflow not present")
def test_no_inline_ci_script_uses_a_name_before_binding_it() -> None:
    """Catch use-before-definition at module level.

    Walks top-level statements in source order, tracking which names are bound.
    A load of an unbound, non-builtin name is a latent NameError.
    """

    builtins_names = set(dir(__builtins__)) | set(dir(__import__("builtins")))
    for line_number, code in _heredocs():
        tree = ast.parse(code)
        bound: set[str] = set()
        for statement in tree.body:
            loads = {
                node.id
                for node in ast.walk(statement)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            # Names stored anywhere inside this statement (loop bodies, `with`
            # blocks, nested scopes) are available to loads within it. What this
            # check targets is the cross-statement case: a name bound only by a
            # LATER top-level statement, which is exactly the NameError that
            # shipped.
            unbound = (
                loads
                - bound
                - builtins_names
                - _locally_scoped_names(statement)
                - _bound_names(statement)
            )
            assert not unbound, (
                f"{WORKFLOW.name}:{line_number} uses {sorted(unbound)} before binding "
                f"(statement on inline line {statement.lineno})"
            )
            bound |= _bound_names(statement)


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repo-root workflow not present")
def test_workflow_references_only_configs_that_exist() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for match in re.finditer(r"configs/([A-Za-z0-9_\.]+\.json)", text):
        path = LAB_ROOT / "configs" / match.group(1)
        assert path.is_file(), f"workflow references a missing config: {match.group(1)}"


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repo-root workflow not present")
def test_workflow_never_destroys_an_evidence_root() -> None:
    """Finding P1-3 must stay closed."""

    text = WORKFLOW.read_text(encoding="utf-8")
    assert "rm -rf full_evidence" not in text
    assert "archive_existing_evidence" in text


@pytest.mark.skipif(not WORKFLOW.is_file(), reason="repo-root workflow not present")
def test_workflow_registered_config_matches_the_declared_study() -> None:
    import json

    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"FULL_CONFIG:\s*\S+/configs/([A-Za-z0-9_\.]+\.json)", text)
    assert match, "workflow does not declare FULL_CONFIG"
    config = json.loads((LAB_ROOT / "configs" / match.group(1)).read_text(encoding="utf-8"))
    assert config["confirmatory"] is True
    assert config["study_id"].startswith("tgsa-phase1-identifiability-v3")


def test_declared_extras_cover_every_workflow_request() -> None:
    if not WORKFLOW.is_file():
        pytest.skip("repo-root workflow not present")
    declared = set(
        tomllib.loads((LAB_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
    )
    requested = {
        extra.strip()
        for match in re.finditer(
            r"pip install[^\n]*\[([^\]]+)\]", WORKFLOW.read_text(encoding="utf-8")
        )
        for extra in match.group(1).split(",")
        if extra.strip() and "/" not in extra
    }
    assert requested <= declared, f"undeclared extras requested: {sorted(requested - declared)}"
