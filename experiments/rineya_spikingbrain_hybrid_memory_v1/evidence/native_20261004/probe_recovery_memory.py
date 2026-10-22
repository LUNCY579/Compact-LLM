"""Instrument the unchanged guarded D recovery; never bypass a stop rule."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sbhm import pipeline
from sbhm.protocol import Blocked, code_hash, now, write_json


def main():
    original = pipeline.resources
    events = []
    def instrumented(*args, **kwargs):
        free, total = torch.cuda.mem_get_info(0)
        event = {'created_utc': now(), 'call': len(events),
                 'step': len(events)//2, 'boundary': 'before' if len(events)%2 == 0 else 'after',
                 'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
                 'peak_allocated': torch.cuda.max_memory_allocated(),
                 'peak_reserved': torch.cuda.max_memory_reserved(), 'free': free, 'total': total}
        events.append(event)
        try:
            return original(*args, **kwargs)  # The original checks still execute.
        except Blocked as error:
            event['guard_failure'] = str(error)
            raise
    pipeline.resources = instrumented
    previous = ROOT / 'runs/titan_transition_v2_20261004'
    args = SimpleNamespace(command='recover', arm='D', config=str(ROOT/'configs/titan_transition.v2.json'),
                           output=str(ROOT/'runs/titan_transition_v2_20261004_memory_probe'),
                           qualification=str(previous/'D-distilled-qualification/receipt.json'),
                           delta=str(previous/'D-distill'), teacher=None,
                           allow_model_load=True, allow_training=True)
    result = {'created_utc': now(), 'code_sha256': code_hash(), 'original_guards_unchanged': True}
    try:
        receipt = pipeline.execute(args)
        result.update(status='GUARDED_RECOVERY_COMPLETED', steps=len(receipt['records']))
    except (Blocked, RuntimeError) as error:
        result.update(status='GUARDED_RECOVERY_FAILED', type=type(error).__name__, reason=str(error))
    result['events'] = events
    write_json(Path(__file__).parent/'D-recovery-memory-probe.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'events'}))
    if events:
        print(json.dumps(events[-1]))
    return 0 if result['status'] == 'GUARDED_RECOVERY_COMPLETED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
