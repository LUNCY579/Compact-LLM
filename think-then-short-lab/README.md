# Think-then-short lab

Standalone side project next to `short-response-model-phase-0-1-2`.

**Question:** can Qwen3-native thinking (`enable_thinking=True`) improve answerable reasoning while keeping the **visible** assistant reply short?

## Why this is separate

Official Phase 1–9 trainers and evals force `enable_thinking=False`. That contract produced the short-response frozen-anchor / greedy-soup behavior. Turning thinking on inside those phases would change the sealed experimental story.

This lab is the **only** place thinking is allowed. It may read official greedy-soup / frozen-anchor checkpoints as **init weights only**. It must never write into official `artifacts/` or change phase code.

## Design

```
foreign JSONL                  converted record              future train/eval
instruction + reasoning  -->   messages (system/user)   -->  enable_thinking=True
+ output                       thinking = reasoning          score answer only
                               answer = output
```

- `reasoning` maps into the Qwen3 think channel.
- `output` maps into the short visible answer.
- Do **not** concatenate reasoning into the visible reply.
- Answer budget default: **32** tokens. Thinking budget default: **512** tokens.
- Official sealed benchmarks stay unused.

## Layout

| Path | Role |
|---|---|
| `configs/think_then_short.json` | Budgets, init model path, conversion filters |
| `data/schemas/think_then_short_record.schema.json` | Converted-record schema |
| `data/raw/` | Drop zone for foreign `instruction` / `reasoning` / `output` JSONL |
| `data/converted/` | Converter output |
| `scripts/convert_to_think_then_short.py` | Working converter + validator |
| `scripts/train_think_then_short.py` | Stub (GPU training not enabled yet) |
| `scripts/evaluate_think_then_short.py` | Stub (eval not enabled yet) |
| `artifacts/` | Future lab outputs only |

## Recorded commands

Run from this folder. Prefer the repo `.venv`. Do **not** share GPU with a live Phase 9 grokking / factor-sweep job.

```powershell
cd C:\Users\clclc\BusinessProject\Chat-Bubble\think-then-short-lab
$py = "C:\Users\clclc\BusinessProject\Chat-Bubble\.venv\Scripts\python.exe"
```

### 1) Convert a foreign dataset

Drop a JSONL file with `instruction` + `output` (optional `reasoning` or `thinking`) into `data/raw/`, then:

```powershell
& $py .\scripts\convert_to_think_then_short.py `
  --input .\data\raw\example.jsonl `
  --output .\data\converted\example.think_then_short.jsonl `
  --config .\configs\think_then_short.json
```

The converter writes a sibling audit JSON (`*.conversion_audit.json`) with kept/dropped counts.

### 2) Future training (stub)

```powershell
& $py .\scripts\train_think_then_short.py `
  --config .\configs\think_then_short.json `
  --train-file .\data\converted\example.think_then_short.jsonl `
  --output-dir .\artifacts\train_smoke
```

Currently exits with “not enabled yet”. When enabled, it will LoRA-SFT greedy soup with `enable_thinking=True` and answer-token upweighting.

### 3) Future evaluation (stub)

```powershell
& $py .\scripts\evaluate_think_then_short.py `
  --config .\configs\think_then_short.json `
  --model .\artifacts\train_smoke\adapter `
  --probe .\data\converted\example.think_then_short.jsonl `
  --output-dir .\artifacts\eval_smoke
```

Currently exits with “not enabled yet”. When enabled, it will score **answer only**, and report thinking vs answer token counts separately.

## Firewall checklist

- [ ] Do not edit `short-response-model-phase-0-1-2/scripts/` for this experiment
- [ ] Do not flip official `enable_thinking` to true
- [ ] Do not write into official Phase artifacts
- [ ] Do not consume sealed benchmarks
- [ ] Keep GPU exclusive when training eventually runs
