# Opallo behavior repair: cheapest controlled screen

**OPALLO-BEH-001** isolates response formatting and literal UNKNOWN compliance without
changing weights or reopening retrieval. Read [PROTOCOL.md](PROTOCOL.md) before execution.
The authoritative scope is a **design plus tested offline reference harness**, not an
integrated native inference runner or a completed repair.

## What is here

`screen.py` prepares fresh synthetic calibration/check fixtures, freezes a single
calibration-only threshold, and replays B / F / G / FG from two prompt-policy captures.
It keeps the historical RAB scorer separate from literal full-response scoring and refuses
missing rows, changed inputs, runtime-identity changes, malformed probabilities, failed
termination and altered calibration specs. Evidence files are create-only.

`REGISTRATION.json` pins the source, fixture hashes, budget and closed boundaries.
`PROVENANCE.json` records the live parent, cited subject, audited Git blobs and observed
branch heads. `evidence/LOCAL_RECEIPT.json` records **synthetic, stdlib-only** execution.
None of these is model-performance evidence.

## Current recommendation

Run the inference-only format/gate screen before adding another training sweep. The parent
already has a distinct three-seed repair-vs-control training registration; leave it intact.
Use the gate lab's first-token exact UNKNOWN signal, not another model's threshold or baked
bias. Keep the short-response corpus available for later, narrowly selected SFT only after
its proposed rows and exact response contracts are reviewed.

The historical probe is not identical to the suite scorer, and its UNKNOWN counter is a
substring counter. The protocol documents both discrepancies rather than silently
rewriting history. Post-hoc containment is not a success endpoint here.

## Local mechanics

From the repository root, with Python 3.10 or newer and no third-party packages:

```bash
python -m unittest discover -s experiments/opallo_behavior_screen_v1/tests -v
python experiments/opallo_behavior_screen_v1/screen.py prepare --out /path/to/new-fixtures
```

`prepare` requires a new destination and writes 64 calibration rows and 128 check rows,
two prompt-only views of each, keys and fixture hashes. A generated manifest explicitly
says **not leakage-cleared**. Inspect the scenario packs and record the protocol's overlap
review before model execution. Do not expose check model outputs during calibration.

## Native capture handoff (NOT implemented / NOT qualified here)

A separately qualified collector must emit one JSON object per capture, not raw prediction
JSONL. The inherited runner is only a candidate building block. Its confidence schema
contains `confidence.unknown.exact_logprob`; the collector must validate the complete EOS
list and convert that finite logprob to `unknown_probability = exp(logprob)`.
Do not infer termination from the runner helper's tokenizer-only EOS flag.

Each envelope has this structure (the row array must contain *all* expected IDs in order):

```json
{
  "protocol": "OPALLO-BEH-001",
  "kind": "native_capture",
  "split": "calibration",
  "formatted": false,
  "prompts_sha256": "canonical digest from screen.prompts, not file-byte hash",
  "identity": {
    "weight_sha256": "e726d6bea1139eddfbcdadeb4c8d7b979e7a55e43490d9787ab99a3ab12ee76b",
    "runtime_sha256": "SHA-256 of the reviewed complete runtime/asset/capture manifest",
    "unknown_single_token": true,
    "unknown_token_id": 0
  },
  "rows": [
    {
      "id": "OB1-calibration-00-base",
      "status": "ok",
      "terminated": true,
      "prediction": "actual unmodified decoded response",
      "input_tokens": 100,
      "generated_tokens": 12,
      "unknown_probability": 0.01
    }
  ]
}
```

Values above are schema examples, **not native evidence**. Replace the illustrative token
ID with the verified native tokenizer ID; never copy it. `identity` must match exactly
across captures. The offline validator checks structure and identity equality, not honesty
of a claimed model execution; the independently reviewed native receipt is indispensable.
The optional `synthetic_fixture` kind is always reported as SYNTHETIC_ONLY.

After qualified calibration captures, freeze the spec:

```bash
python experiments/opallo_behavior_screen_v1/screen.py calibrate \
  --baseline-cal /path/to/calibration_B.json \
  --formatted-cal /path/to/calibration_F.json --out /path/to/new-gate-spec.json
```

Commit that spec and the calibration capture hashes **before check capture/outcome access**.
Then, once only:

```bash
python experiments/opallo_behavior_screen_v1/screen.py score \
  --baseline-cal /path/to/calibration_B.json \
  --formatted-cal /path/to/calibration_F.json --spec /path/to/new-gate-spec.json \
  --baseline-check /path/to/check_B.json --formatted-check /path/to/check_F.json \
  --out /path/to/new-replay-result.json
```

`screen_criteria_met` is only an engineering screen flag. Even a native-labeled envelope
produces **OFFLINE_REPLAY_ONLY**, not NATIVE_PASS, restored abstention, improved retrieval,
or release authorization. No model loader, training path, watchdog, baking command, CI
dispatch or registry mutation is included. Those remain separate gates.
