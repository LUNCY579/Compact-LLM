"""Tiny fixed JSON-RPC child fixture. No framework/model imports or native grant."""
import argparse
import json
import os
import sys

SCHEMA = 'CE8-BOUNDED-RPC-1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    with open(args.request, 'r', encoding='utf8') as stream: request = json.load(stream)
    if request.get('schema') != 'CE8-PROCESS-FIXTURE-REQUEST-1' or request.get('native') is not False:
        raise ValueError('fixture request is not native execution authority')
    binding = json.loads(os.environ['CE8_RPC_BINDING'])
    def emit(kind, sequence, payload):
        message = {'schema': SCHEMA, **binding, 'kind': kind, 'sequence': sequence, 'worker_pid': os.getpid(), 'payload': payload}
        raw = json.dumps(message, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if len(raw.encode('utf8')) + 1 > request['max_message_bytes']: raise ValueError('fixture message ceiling')
        sys.stdout.write(raw + '\n'); sys.stdout.flush()
    emit('ready', 0, {'fixture': True, 'model_loads': 0, 'framework_imports': 0, 'native_qualified': False})
    expected = 1
    while True:
        raw = sys.stdin.buffer.readline(request['max_message_bytes'] + 1)
        if not raw: return 0
        if len(raw) > request['max_message_bytes'] or not raw.endswith(b'\n'): raise ValueError('fixture control size')
        message = json.loads(raw)
        if message.get('schema') != SCHEMA or message.get('kind') != 'control' or message.get('sequence') != expected:
            raise ValueError('fixture control phase/sequence')
        if any(message.get(key) != binding[key] for key in ('session_id', 'request_sha256', 'scope_sha256')):
            raise ValueError('fixture foreign control binding')
        if message.get('worker_pid') != os.getpid(): raise ValueError('fixture foreign control PID')
        command, payload = message['payload']['command'], message['payload']['data']
        if any(key in payload for key in ('approved', 'native_permit', 'training_authorized', 'qualification_authorized')):
            raise ValueError('RPC flags cannot grant native execution')
        if command == 'ping': response = {'echo': payload, 'fixture': True, 'native_qualified': False}
        elif command == 'finish': response = {'fixture_complete': True, 'native_qualified': False}
        else: raise ValueError('fixed fixture command only')
        emit('response', expected, response); expected += 1
        if command == 'finish': return 0


if __name__ == '__main__': raise SystemExit(main())
