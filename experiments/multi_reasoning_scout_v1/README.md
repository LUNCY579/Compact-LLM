# Multi-reasoning scout MRS-1

Inference-only, isolated, open development. Start with `research/multi_reasoning_scout_v1/CURRENT_STATE_AUDIT.md` and `MULTI_REASONING_PROTOCOL.md`. Exact manifests, protocol, future contract and receipts live in that research directory. No historical branch or protected bank is an input to generation.

Run commands from the isolated worktree root with the pinned Python environment (`C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe`). No package install or model download is needed.

```powershell
$py = 'C:\Users\clclc\BusinessProject\.venv-mq001\Scripts\python.exe'
& $py -m pytest experiments/multi_reasoning_scout_v1/tests -q
& $py -m experiments.multi_reasoning_scout_v1.runner verify
& $py -m experiments.multi_reasoning_scout_v1.runner estimate
& $py -m experiments.multi_reasoning_scout_v1.runner run --model supa-1.7-frozen --out research/multi_reasoning_scout_v1/evidence/native-supa-001
& $py -m experiments.multi_reasoning_scout_v1.runner run --model d1-seed17-bf16-export --out research/multi_reasoning_scout_v1/evidence/native-d1-001
& $py -m experiments.multi_reasoning_scout_v1.analysis score --run research/multi_reasoning_scout_v1/evidence/native-d1-001 --out research/multi_reasoning_scout_v1/evidence/scored-d1-001
```

Use a fresh output path after a refused preload. A loaded model attempt has a create-only launch claim and cannot silently retry. A successful score requires a complete native generation receipt, the successful supervisor receipt, and exact raw file hashes. Fixture runs never satisfy this path. The worker rejects direct launch without a live foreground supervisor. It shares the existing `rineya-gpu-0.lock` lease and never clears a foreign lock.

For the future candidate, create the registration described in MULTI_REASONING_POSTTRAIN_CONTRACT.md, then use `run --model NEW_ID --future-registration ABSOLUTE_REGISTRATION_PATH --out NEW_DIRECTORY`. Score it with the same `analysis score` command. Compare:

```powershell
& $py -m experiments.multi_reasoning_scout_v1.analysis factorial --old OLD_COMPLETE_RESULTS.json --new NEW_COMPLETE_RESULTS.json --out NEW_FACTORIAL_DIRECTORY
```

`bank.generate()` reproduces the already frozen artifacts in memory; do not replace them. `bank.build()`, `audit`, and `core.make_freeze()` are construction commands and refuse existing create-only outputs. The optional M2 function is software-tested and deliberately not dispatched in MRS-1.

Evidence categories are explicit: SOFTWARE PASS, NATIVE EXECUTION PASS, OPEN DEV RESULT, MECHANISM SCOUT RESULT, CONFIRMATORY CLAIM, RELEASE CLAIM. The last two remain false. Current blockers and actual test/run status are recorded in the result and implementation reports, not inferred from this runbook.
