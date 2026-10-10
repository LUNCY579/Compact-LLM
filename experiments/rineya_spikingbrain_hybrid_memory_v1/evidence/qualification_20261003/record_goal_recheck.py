"""Revalidate the frozen pilot and resource blocker without loading weights."""
import argparse
import json
from pathlib import Path
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sbhm.data import load_corpus
from sbhm.protocol import (Blocked, authorize, code_hash, digest, environment,
                           now, read_json, resources, source_lock, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--turn', type=int, required=True)
    parser.add_argument('--previous-turn', choices=['progress', 'verified_wait', 'no_progress'], required=True)
    args = parser.parse_args()
    config = read_json(ROOT / 'configs/titan_transition.v2.json')
    binding = read_json(Path(__file__).parent / 'registered_v2_pilot.json')['binding']
    assert code_hash() == binding['code_sha256']
    assert digest(config) == binding['config_sha256']
    assert source_lock(config) == binding['source_sha256']
    _, corpus_sha = load_corpus(config['corpus_manifest'])
    assert corpus_sha == binding['corpus_sha256']
    for action in ('qualify', 'teacher', 'distill', 'recover', 'evaluate'):
        authorize(config, action, binding, True, True)
    record = {'created_utc': now(), 'goal_turn': args.turn,
              'previous_goal_turn_classification': args.previous_turn,
              'environment': environment(),
              'host_available_ram_bytes': psutil.virtual_memory().available,
              'source_and_corpus_bindings_unchanged': True,
              'dated_registered_actions_authorized': True,
              'native_model_loaded': False, 'native_comparison_completed': False}
    try:
        record['resources'] = resources(config['device'])
        record['resource_status'] = 'ADMITTED'
    except Blocked as error:
        record.update(resource_status='BLOCKED', reason=str(error))
    write_json(Path(__file__).parent / f'goal_continuation_{args.turn}.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
