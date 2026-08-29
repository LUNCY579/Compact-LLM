from __future__ import annotations

import json
from pathlib import Path

VAULT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def entries_by_codename():
    catalog = load_json(VAULT / "CATALOG.json")
    return catalog, {entry["codename"]: entry for entry in catalog["cores"]}


def test_current_protocol_halo_and_nimbus_metadata_is_consistent() -> None:
    catalog, entries = entries_by_codename()
    del catalog
    expected = {"Halo": 29, "Nimbus": 30}
    for codename, hits in expected.items():
        entry = entries[codename]
        core = load_json(VAULT / entry["directory"] / "CORE.json")
        assert entry["core"]["showcase_hits"] == hits
        assert core["showcase_hits"] == hits
        assert entry["core"] == core


def test_a_series_tracking_resolves_real_lineages() -> None:
    catalog, entries = entries_by_codename()
    labels = catalog["tracking_labels"]
    assert labels["Supa-A1"]["core_id"] == entries["Supa-A1"]["core_id"]
    assert labels["Nimbus-A1"]["core_id"] == entries["Nimbus"]["core_id"]
    assert entries["Nimbus"]["tracking_label"] == "Nimbus-A1"
    assert entries["Nimbus"]["core"]["tracking_label"] == "Nimbus-A1"


def test_no_bias_ablation_is_not_a_duplicate_catalog_core() -> None:
    catalog, _ = entries_by_codename()
    encoded = json.dumps(
        [
            {"core_id": entry["core_id"], "directory": entry["directory"]}
            for entry in catalog["cores"]
        ]
    )
    assert "Rineya-Nimbus-Dense-1.0-nobias" not in encoded


def test_supa_a1_catalog_core_matches_its_directory() -> None:
    _, entries = entries_by_codename()
    entry = entries["Supa-A1"]
    core = load_json(VAULT / entry["directory"] / "CORE.json")
    assert entry["core"] == core
    assert core["evaluation"]["checkpoint_results"]["showcase64"]["hits"] == 43
    assert core["evaluation"]["checkpoint_results"]["stability192"]["hits"] == 125
