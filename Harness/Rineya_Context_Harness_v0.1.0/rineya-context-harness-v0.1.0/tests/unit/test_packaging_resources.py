from __future__ import annotations

from pathlib import Path


def test_packaged_migrations_match_repository_copies() -> None:
    root = Path(__file__).resolve().parents[2]
    repository = root / "migrations"
    packaged = root / "src" / "rineya_harness" / "migrations"
    repository_files = {path.name: path.read_bytes() for path in repository.glob("*.sql")}
    packaged_files = {path.name: path.read_bytes() for path in packaged.glob("*.sql")}
    assert packaged_files == repository_files


def test_release_file_selection_excludes_local_secrets_and_runtime(tmp_path) -> None:
    from scripts.generate_manifest import files_for

    (tmp_path / "README.md").write_text("ok", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    (tmp_path / ".coverage").write_text("coverage", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "example.yaml").write_text("safe: true", encoding="utf-8")
    (tmp_path / "configs" / "local-rineya.yaml").write_text(
        "api_key: secret", encoding="utf-8"
    )
    (tmp_path / "artifacts" / "runtime").mkdir(parents=True)
    (tmp_path / "artifacts" / "runtime" / "state.sqlite3").write_bytes(b"db")
    (tmp_path / "artifacts" / "release").mkdir(parents=True)
    (tmp_path / "artifacts" / "release" / "validation_report.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "artifacts" / "release" / "repository_manifest.json").write_text(
        "{}", encoding="utf-8"
    )

    release_paths = {relative for _path, relative in files_for(tmp_path)}
    manifest_paths = {
        relative for _path, relative in files_for(tmp_path, for_manifest=True)
    }

    assert "README.md" in release_paths
    assert "configs/example.yaml" in release_paths
    assert "artifacts/release/validation_report.json" in release_paths
    assert "artifacts/release/repository_manifest.json" in release_paths
    assert "artifacts/release/repository_manifest.json" not in manifest_paths
    assert ".env" not in release_paths
    assert ".coverage" not in release_paths
    assert "configs/local-rineya.yaml" not in release_paths
    assert "artifacts/runtime/state.sqlite3" not in release_paths


def test_packager_does_not_include_its_own_output(tmp_path) -> None:
    import zipfile

    from scripts.package_release import package

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("ok", encoding="utf-8")
    release = tmp_path / "artifacts" / "release"
    release.mkdir(parents=True)
    (release / "validation_report.json").write_text("{}", encoding="utf-8")
    (release / "repository_manifest.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "release.zip"

    _digest, _count = package(tmp_path, output, validate=False)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert not any(name.endswith("release.zip") for name in names)
    assert not any(name.endswith("release.zip.sha256") for name in names)
