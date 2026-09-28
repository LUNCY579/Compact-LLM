#!/usr/bin/env python3
"""Register an already-materialized core in CATALOG.json without rewriting the catalog.

`_materialize_core.py` rebuilds the whole catalog from its own hard-coded `CORES` table,
which is right for cores that table produces and wrong for anything else: running it would
drop every entry it does not know about. Cores promoted out of an experiment are written by
their own experiment-side script, so they need a registration step that only adds.

    python Rineya_Published_Cores/_register_exploration_core.py Rineya-Opallo-1.0.0-devfreeze

Refuses to register a core whose panel validation has not been measured. A catalog entry is
how the rest of the family finds a model, and an entry carrying `NOT_MEASURED` is an
invitation to quote a score nobody took.

Idempotent: re-running refreshes an existing entry in place rather than appending a
duplicate, so the catalog can be resynced after a core's CORE.json is amended.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

VAULT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='core directory name inside the vault')
    parser.add_argument('--allow-unmeasured', action='store_true',
                        help='register despite an unmeasured panel validation')
    args = parser.parse_args()

    core_dir = VAULT / args.directory
    core_path = core_dir / 'CORE.json'
    if not core_path.is_file():
        raise SystemExit(f'no CORE.json at {core_dir}')
    core = json.loads(core_path.read_text(encoding='utf-8'))

    validation = core.get('panel_validation') or {}
    status = validation.get('status')
    if status != 'PASS' and not args.allow_unmeasured:
        raise SystemExit(f'panel_validation status is {status!r}; run validate_published_core.py '
                         f'before registering, or pass --allow-unmeasured deliberately')

    catalog_path = VAULT / 'CATALOG.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    entry = {
        'core_id': core['core_id'],
        'codename': core['codename'],
        'directory': args.directory,
        'present': (core_dir / 'model.safetensors').is_file(),
        'core': core,
    }
    existing = [i for i, e in enumerate(catalog['cores'])
                if e.get('directory') == args.directory or e.get('core_id') == core['core_id']]
    if len(existing) > 1:
        raise SystemExit(f'catalog already holds {len(existing)} entries for this core; '
                         f'resolve by hand rather than guessing which one to keep')
    if existing:
        catalog['cores'][existing[0]] = entry
        action = 'updated'
    else:
        catalog['cores'].append(entry)
        action = 'added'
    catalog['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8')
    print(f'{action} {core["codename"]} ({core["core_id"]})')
    print(f'  present={entry["present"]} panel={validation.get("correct_total")}/'
          f'{validation.get("rows")} status={status}')
    print(f'  catalog now holds {len(catalog["cores"])} cores')
    return 0


if __name__ == '__main__':
    sys.exit(main())
