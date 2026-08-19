from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from rineya_harness import __version__
from rineya_harness.config import load_config, redact_config
from rineya_harness.domain.enums import ActorType, DeleteScope, ReasonCode, RunMode
from rineya_harness.domain.models import DeleteMemoryRequest, HarnessRunRequest
from rineya_harness.evals.runner import run_eval, write_report
from rineya_harness.examples import (
    build_hanoi_demo,
    build_restaurant_demo,
    fixture_container,
    run_fixture_research,
)
from rineya_harness.factory import create_container, resolve_config_relative_paths

app = typer.Typer(
    name="rineya-harness",
    help="Rineya Context Harness: bounded search, evidence, and recursive-state forgetting.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _container(config_path: Path | None):
    config = resolve_config_relative_paths(load_config(config_path), config_path)
    return create_container(config)


def _print(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))


@app.command()
def doctor(
    config: Annotated[Path | None, typer.Option("--config", help="YAML configuration path.")] = None,
) -> None:
    """Validate configuration and initialize the local state store."""

    container = _container(config)
    _print(
        {
            "status": "ok",
            "version": __version__,
            "configuration": redact_config(container.config),
            "database_exists": container.store.path.exists(),
            "model_backend": container.model.name,
            "search_provider": container.search_provider.name,
            "web_fetcher": type(container.fetcher).__name__,
        }
    )


@app.command("run")
def run_command(
    question: Annotated[str, typer.Argument(help="User question or task.")],
    mode: Annotated[RunMode, typer.Option("--mode")] = RunMode.AUTO,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
) -> None:
    """Execute one bounded harness run."""

    container = _container(config)
    response = asyncio.run(
        container.workflow.run(HarnessRunRequest(question=question, mode=mode, run_id=run_id))
    )
    _print(response.model_dump(mode="json"))


@app.command()
def state(
    run_id: Annotated[str, typer.Argument()],
    config: Annotated[Path | None, typer.Option("--config")] = None,
    include_inactive: Annotated[bool, typer.Option("--include-inactive")] = False,
) -> None:
    """Inspect versioned current state."""

    container = _container(config)
    run = container.store.get_run(run_id)
    memories = container.store.list_memory(run_id, include_inactive=include_inactive)
    _print(
        {
            "run": run.model_dump(mode="json"),
            "memory": [item.model_dump(mode="json") for item in memories],
            "state_hash": container.store.current_state_hash(run_id),
        }
    )


@app.command("delete-memory")
def delete_memory_command(
    run_id: Annotated[str, typer.Argument()],
    memory_ids: Annotated[list[str], typer.Argument(help="One or more stable memory IDs.")],
    expected_revision: Annotated[int, typer.Option("--expected-revision", min=0)],
    scope: Annotated[DeleteScope, typer.Option("--scope")] = DeleteScope.ACTIVE_CONTEXT,
    reason_code: Annotated[ReasonCode, typer.Option("--reason-code")] = (
        ReasonCode.COMPLETED_NO_FUTURE_DEPENDENCY
    ),
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    replacement_state_id: Annotated[
        list[str] | None,
        typer.Option("--replacement-state-id", help="Active replacement IDs preserving sufficiency."),
    ] = None,
    confirmation_token: Annotated[str | None, typer.Option("--confirmation-token")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Explicitly evict, archive, or guardedly purge memory."""

    container = _container(config)
    result = container.memory.delete_memory(
        DeleteMemoryRequest(
            run_id=run_id,
            memory_ids=memory_ids,
            scope=scope,
            reason_code=reason_code,
            reason=reason,
            replacement_state_ids=replacement_state_id or [],
            expected_state_revision=expected_revision,
            confirmation_token=confirmation_token,
        ),
        actor=ActorType.ADMIN if scope is DeleteScope.PERSISTENT_PURGE else ActorType.USER,
    )
    _print(result.model_dump(mode="json"))


@app.command("restore-memory")
def restore_memory_command(
    run_id: Annotated[str, typer.Argument()],
    memory_ids: Annotated[list[str], typer.Argument()],
    expected_revision: Annotated[int, typer.Option("--expected-revision", min=0)],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Restore reversibly archived or evicted memory."""

    container = _container(config)
    result = container.memory.restore(
        run_id=run_id,
        memory_ids=memory_ids,
        expected_revision=expected_revision,
        actor=ActorType.USER,
    )
    _print(result.model_dump(mode="json"))


@app.command()
def checkpoint(
    run_id: Annotated[str, typer.Argument()],
    label: Annotated[str, typer.Option("--label")] = "manual",
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Create a recoverable state checkpoint."""

    container = _container(config)
    record = container.store.create_checkpoint(run_id, label=label)
    _print(record.model_dump(mode="json"))


@app.command("restore-checkpoint")
def restore_checkpoint_command(
    run_id: Annotated[str, typer.Argument()],
    checkpoint_id: Annotated[str, typer.Argument()],
    expected_revision: Annotated[int, typer.Option("--expected-revision", min=0)],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Atomically restore a known checkpoint."""

    container = _container(config)
    result = container.store.restore_checkpoint(
        run_id, checkpoint_id, expected_revision=expected_revision
    )
    _print(result.model_dump(mode="json"))


@app.command("export-run")
def export_run_command(
    run_id: Annotated[str, typer.Argument()],
    output: Annotated[Path, typer.Option("--output")] = Path("run_export.json"),
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Export state, provenance, checkpoints, and events with a content hash."""

    container = _container(config)
    payload = container.store.export_run(run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print({"output": str(output), "export_sha256": payload["export_sha256"]})


@app.command()
def trace(
    run_id: Annotated[str, typer.Argument()],
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Read the connected JSONL execution trace for one run."""

    container = _container(config)
    _print([event.model_dump(mode="json") for event in container.traces.read(run_id)])


@app.command()
def evaluate(
    cases: Annotated[Path, typer.Option("--cases")] = Path("evals/cases/rsf_trials.jsonl"),
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Evaluate recursive-state-forgetting trial records."""

    metrics = run_eval(cases)
    if output is not None:
        write_report(metrics, output)
    _print(metrics.model_dump(mode="json"))


@app.command()
def demo(
    scenario: Annotated[
        str,
        typer.Argument(help="restaurant, hanoi, or research"),
    ] = "restaurant",
    workdir: Annotated[Path | None, typer.Option("--workdir")] = None,
) -> None:
    """Run a deterministic, network-free demonstration."""

    owned_temp: tempfile.TemporaryDirectory[str] | None = None
    if workdir is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="rineya-harness-demo-")
        workdir = Path(owned_temp.name)
    workdir.mkdir(parents=True, exist_ok=True)
    container = fixture_container(workdir)
    normalized = scenario.casefold()
    if normalized == "restaurant":
        payload = build_restaurant_demo(container)
    elif normalized == "hanoi":
        payload = build_hanoi_demo(container)
    elif normalized == "research":
        payload = asyncio.run(run_fixture_research(container))
    else:
        raise typer.BadParameter("scenario must be restaurant, hanoi, or research")
    payload["workdir"] = str(workdir)
    _print(payload)
    if owned_temp is not None:
        owned_temp.cleanup()


@app.command()
def serve(
    config: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Start the optional local FastAPI service."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise typer.BadParameter("Install the `api` extra: pip install -e '.[api]'") from exc
    from rineya_harness.api.app import create_app

    loaded = resolve_config_relative_paths(load_config(config), config)
    uvicorn.run(
        create_app(config=loaded),
        host=loaded.api.host,
        port=loaded.api.port,
        reload=loaded.api.reload,
    )


if __name__ == "__main__":
    app()
