"""Preflight must disclose missing corpus admission without loading a model."""
import json
import sys
from pathlib import Path
from sbhm import __main__ as cli, data


def test_preflight_reports_unreviewed_corpus(tmp_path, monkeypatch, capsys):
    data.make_fixture_corpus(tmp_path / 'corpus')
    config = json.loads((Path(__file__).parents[1] / 'configs/native.example.json').read_text())
    config['corpus_manifest'] = str(tmp_path / 'corpus/manifest.json')
    config['grant'] = 'pending-grant.json'
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(cli, 'resources', lambda *_: {})
    monkeypatch.setattr(cli, 'source_lock', lambda *_: 'a' * 64)
    monkeypatch.setattr(cli, 'environment', lambda: {'versions': {'transformers': '4.51.3'}})
    monkeypatch.setattr(sys, 'argv', ['sbhm', 'preflight', '--config', str(path)])
    assert cli.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'BLOCKED' and result['model_loaded'] is False
    assert result['blockers'] == ['corpus has not been marked reviewed for this experiment']
