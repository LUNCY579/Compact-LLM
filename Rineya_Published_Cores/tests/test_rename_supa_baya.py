from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "_rename_supa_baya.py"
SPEC = importlib.util.spec_from_file_location("rename_supa_baya", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def minimal_core(spec, sibling):
    return {
        "core_id": spec.old_core_id,
        "codename": spec.old_codename,
        "directory_name": spec.old_directory,
        "public_claim": f"Dense {spec.old_codename} result",
        "pair": {
            "sibling_core_id": sibling.old_core_id,
            "sibling_directory": sibling.old_directory,
        },
        "dense_validation": {"core_id": spec.old_core_id},
    }


def test_transform_core_changes_only_current_identity() -> None:
    spec, sibling = MODULE.SPECS
    core = minimal_core(spec, sibling)
    result = MODULE.transform_core(core, spec, sibling, "now")
    assert result["core_id"] == spec.new_core_id
    assert result["codename"] == "Supa"
    assert result["pair"]["sibling_core_id"] == sibling.new_core_id
    assert result["former_identity"]["core_id"] == spec.old_core_id
    assert result["dense_validation"] == core["dense_validation"]


def test_transform_catalog_preserves_unknown_and_superseded() -> None:
    first, second = MODULE.SPECS
    cores = {
        first.new_codename: MODULE.transform_core(
            minimal_core(first, second), first, second, "now"
        ),
        second.new_codename: MODULE.transform_core(
            minimal_core(second, first), second, first, "now"
        ),
    }
    catalog = {
        "custom": {"keep": True},
        "superseded": [{"id": "old"}],
        "cores": [
            {
                "core_id": first.old_core_id,
                "codename": first.old_codename,
                "directory": first.old_directory,
                "custom_entry": 1,
            },
            {
                "core_id": second.old_core_id,
                "codename": second.old_codename,
                "directory": second.old_directory,
                "custom_entry": 2,
            },
        ],
    }
    result = MODULE.transform_catalog(catalog, cores, "now")
    assert result["custom"] == catalog["custom"]
    assert result["superseded"] == catalog["superseded"]
    assert [entry["custom_entry"] for entry in result["cores"]] == [1, 2]
    assert [entry["codename"] for entry in result["cores"]] == ["Supa", "Baya"]


def test_readme_records_former_identity() -> None:
    spec = MODULE.SPECS[0]
    text = MODULE.render_readme(
        "# Rineya Taria\nTaria and Lumina are paired models.\n", spec
    )
    assert "# Rineya Supa" in text
    assert "Supa and Baya are paired models" in text
    assert "Formerly published internally as **Taria**" in text
