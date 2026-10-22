"""Bind completed CPU test output to this code, tests and installed environment."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

experiment = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(experiment))
from sbhm.protocol import code_hash

output = Path(__file__).resolve().parent
full = (output/'pytest-full.txt').read_text(encoding='utf-8')
integration = (output/'pytest-integration.txt').read_text(encoding='utf-8')
assert '80 passed' in full and 'skipped' not in full
assert '1 passed, 56 deselected' in integration and 'skipped' not in integration
assert (output/'pip-check.txt').read_text(encoding='utf-8').strip() == 'No broken requirements found.'
summary = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS_OFFLINE_PINNED_LIBRARY_MECHANICS_ONLY',
    'cwd': str(experiment),
    'python_command': r'.\.venv\Scripts\python.exe',
    'environment': json.loads((output/'environment.json').read_text(encoding='utf-8')),
    'code_sha256': code_hash(),
    'tests_sha256': {str(path.relative_to(experiment)): sha256(path.read_bytes()).hexdigest()
                     for path in sorted((experiment/'tests').glob('*.py'))},
    'commands': [
        r'py -3.10 -m venv .venv',
        r'.\.venv\Scripts\python.exe -m pip install --upgrade pip',
        r'.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu',
        r'.\.venv\Scripts\python.exe -m pip install -r requirements.txt',
        r'$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:CUDA_VISIBLE_DEVICES="-1"',
        r'.\.venv\Scripts\python.exe -m pytest -q tests/test_experiment.py -k real_transformers --disable-warnings',
        r'.\.venv\Scripts\python.exe -m pytest -q tests --disable-warnings',
        r'.\.venv\Scripts\python.exe -m compileall -q sbhm tests',
        r'.\.venv\Scripts\python.exe -m pip check',
        r'.\.venv\Scripts\python.exe evidence/qualification_20261003/inspect_pinned_environment.py',
        r'.\.venv\Scripts\python.exe evidence/qualification_20261003/inspect_corpus_token_geometry.py',
    ],
    'tests': {'full_suite_passed': 80, 'full_suite_skipped': 0,
              'original_suite_including_integration': 57, 'new_comparison_cases': 23,
              'targeted_real_library_passed': 1},
    'official_library_checks': [
        'Tiny random CPU Qwen3 with explicit head_dim=8 != hidden_size/heads=4 and native sdpa',
        'Same frozen random source across A/B/C/D; A/B zero-initialized logits equal frozen base exactly',
        'Exact identity-wrapper logits',
        'Full/tokenwise/chunked prefix logit and greedy-token parity at chunk sizes1/3/full',
        'Fresh request independence and no converted-site full-history KV',
        'Hybrid zero-mix logits equal local-only logits exactly',
    ],
    'installation_repair': 'Bundled Python3.10 pip23.0.1 rejected typing_extensions normalized wheel name; upgrading only scoped pip to26.2.1 resolved installation. No shared environment changes.',
    'pretrained_weights_loaded': False,
    'model_download_or_upload': False,
    'gpu_execution': False,
    'protected_data_access': False,
    'native_model_qualified': False,
    'native_training_fit_established': False,
    'behavior_or_efficiency_result': False,
    'corpus_reviewed': False,
    'operator_grant_created': False,
}
(output/'offline_qualification.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': summary['status'], 'tests': summary['tests'],
                  'code_sha256': summary['code_sha256']}, indent=2))
