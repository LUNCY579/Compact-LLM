# Continue RRP-001

## Mandatory scope

Read [GOAL.md](GOAL.md), [STATUS.json](STATUS.json), [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [RESULTS.md](RESULTS.md), [EXPERIMENT_REGISTRY.json](EXPERIMENT_REGISTRY.json) and the package [AGENTS.md](../../experiments/reasoning_posttraining_v1/AGENTS.md). The goal is still active. Publication and passing fixtures do not complete it.

Worktree: `C:/Users/clclc/BusinessProject/ChatBubble-rineya-reasoning-posttraining-v1`.
Branch: `research/rineya-reasoning-posttraining-v1`.
Implementation commit: `58bf64cf93ff918ede4b1079b8a5716ee0e1e312`.
Parent: `59df72c2e24e6c4d3c7e4e86c1dd4ae143054400`.

Refresh actual state first:

```powershell
git status --short
git rev-parse HEAD
git ls-remote --heads origin research/rineya-reasoning-posttraining-v1 codex/model-quality-audit-v2 research/model-quality-targeted-gpu-v1
```

Do not modify the original `ChatBubble-ced-native-loader-qualification` checkout: its untracked `rineya_reasoning_forge_rl_v2/` is original source data. Source assets and both archives remain read-only there/in Downloads. All previously existing research branches are preserved.

## Exact prepared inputs

Actual source directory: `C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/Supa`.
Weight SHA256: `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
Prepared directory: `C:/Users/clclc/BusinessProject/rineya-reasoning-posttraining-records/inspection-20260915/prepared-003`.
Data contract file SHA256: `ad3fc4c052d6c7a50c08c915069911772add1088f0c1e3db746dae94b712e5c8`.
Selected input SHA256: `ccd5d04170935874b567961dff4c1ae29771eea203cbc4f4373181b5403f3b2d`.
Dedicated Python: `C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe`.

The source/tokenizer bind the existing4.57.1 native lane; native model execution on this successor has not occurred. Human review remains pending. All96v2/46v1 source files matched the archives before imports, and the source-weight header was counted without instantiation. Reverify bindings before actual model use.

## Next work and approval

The user was asked through an asynchronous question for the exact [engineering envelope](BUDGET_REQUEST.json). **No answer had arrived when this record was written.** A preselected UI choice is not approval. Record a real session approval and its exact scope before model execution. Do not reuse old MQ reviews, consumed permits or mere goal-mode authority. The numeric request covers qualification, failed attempts/retries, the32-world microfit and fresh-process resume verification; broad SFT/RL/teachers remain closed.

Continue permitted code work while awaiting that answer: implement the missing **supervised native controller** around `engine.microfit`, using the existing lightweight MQ supervisor, guarded child context and cooperative GPU lease. It must validate source/data/code/runtime bindings before load; enforce a durable all-attempt budget journal; count qualification/CE/prefill/decode/update/disk costs; fail closed on incomplete state; collect a real zero-update source/longest-shape qualification; then execute the fixed plan and a separate fresh-process resume verification. A lower-level engine callback is not a budget or native qualification.

Keep current resource guards. The last observed5.46GiB free host memory (23:36 local) was below the6GiB training preload threshold even before Torch overhead; it is not current telemetry. Check actual live Python jobs, RAM/VRAM/disk and unexplained locks before any model command. Never close other applications, evict workers, delete checkpoints, assume a lock proves a live process, or retry a timed-out observation as if the model died. All owned preprocessing handles recorded here are terminal; no pretrained job was launched.

Exact current CPU check command after controller edits:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe' -m pytest experiments/reasoning_posttraining_v1/tests -q
```

There is deliberately **no claimed runnable native experiment command yet**: controller implementation and qualification are unfinished. Add the exact supervised command here once it exists and has passed fixture gates. Do not substitute the historical `--_worker` bypass or call the lower-level engine on real weights directly.

After microfit, preserve results whether it passes or fails and select one falsifying diagnosis. A successful training-only fit opens consideration of the A/B/C ladder; it is not reasoning improvement. A failure retires or diagnoses that recipe, not sub-billion models. Keep the full remaining goal and its confirmation/retention obligations intact.
