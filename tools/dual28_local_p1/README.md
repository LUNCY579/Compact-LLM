# Local official P1 preparation — D28-001

This helper prepares a foreground local CPU run of **only P1** at ChatBubble
`cd6ef9bd49724da6879bfa4d945af3bfaa6d9758`. Preparation is not an official receipt.
No official weights were available when this helper was tested; no official P1
has been executed by this delivery. The original 94 tests are historical scoped
random-weight evidence. The helper reruns that scoped suite on your local runtime.

The helper lives on a separate preparation branch. It always creates a detached,
full worktree at the original experiment commit, so a resulting P1 belongs to that
original head. It does not modify the experiment, dependency locks, tolerances,
source assets in your existing folder, workflows, or your current working branch.

## Requirements

- Windows x64 or Linux x64, Git, and CPython 3.11–3.13; 3.11 recommended.
  Use the ASUS/Windows machine for this exact lane. Native macOS/Apple Silicon is
  intentionally rejected rather than substituting a different torch build.
- An existing full, non-shallow ChatBubble clone with the pinned commit fetched.
  A dirty existing checkout is allowed: it is not used for execution or changed.
- Your exact ungated Supa 1.7.0-devfreeze directory with all seven locked files.
  The directory can be named `Supa` or `Rineya-Supa-1.7.0-devfreeze`; the bytes must
  match. No generic Qwen, GGUF, re-merged core, symlink or LFS pointer is accepted.
- Weight file: **1,192,135,096 bytes**, SHA-256
  `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
- About **23 GB free plus room for the full checkout**, package caches and any
  later retries. P1 retains 13,168,262,656 bytes of checkpoint tensor data, plus
  metadata and the runner's 1 GiB disk reserve. The helper separately copies your
  source and creates a virtual environment. It never deletes failed attempts.
- Available system RAM must accommodate full FP32 models and temporary copies.
  This preparation has no measured RAM-fit bound. The ASUS's 16 GB is not certified;
  close memory-heavy applications and monitor RAM. A larger-RAM x64 CPU host is an
  option if allocation fails. Neither disk checks nor successful setup prove fit.
- Internet is needed for Git fetching and installation from the official PyTorch
  CPU index and PyPI. Model execution is local-only with Hub offline flags.

## Windows PowerShell

Download `local_p1.py` into a convenient folder outside the repository. Open
PowerShell there. Replace the three example paths below with your actual paths.
`--work-dir` must not exist and must be outside both your repo and source folder.
Its parent directory must already exist. Paths containing spaces are supported.

First fetch the exact original commit (this does not check out another branch):

```powershell
git -C "C:\Research\ChatBubble" fetch origin cd6ef9bd49724da6879bfa4d945af3bfaa6d9758
```

For a shallow clone only, first run `git -C "C:\Research\ChatBubble" fetch --unshallow origin`.
If you have no clone, use `git clone https://github.com/LUNCY579/ChatBubble.git`
and authenticate through your normal Git credential flow. The helper does not
collect credentials, download weights, or alter Git authentication settings.

Prepare (hash check, source copy, isolated runtime install, import/binding checks,
inspect, and scoped random-weight tests; **no official forward pass**):

```powershell
py -3.11 .\local_p1.py prepare --repo "C:\Research\ChatBubble" --source "C:\Models\Supa" --work-dir "C:\Research\dual28-p1-001"
```

After it prints `PREPARED_NOT_P1`, run the registered official P1:

```powershell
py -3.11 .\local_p1.py run --work-dir "C:\Research\dual28-p1-001"
```

No PowerShell activation script or execution-policy change is needed. The launcher
uses the virtual environment's Python directly. Use the same Python version for
both helper commands. Keep PowerShell open and the machine awake; this is not a
background or scheduled task. Commands print the current log path; you can watch
it from another PowerShell window with `Get-Content "<log path>" -Wait`.

## Linux x64

Use the same commands with Linux paths and `python3.11` instead of `py -3.11`:

```bash
git -C /path/to/ChatBubble fetch origin cd6ef9bd49724da6879bfa4d945af3bfaa6d9758
python3.11 local_p1.py prepare --repo /path/to/ChatBubble --source /path/to/Supa --work-dir /path/to/dual28-p1-001
python3.11 local_p1.py run --work-dir /path/to/dual28-p1-001
```

## Runtime and evidence

The isolated environment installs torch `2.10.0+cpu`, Transformers `4.51.3`, numpy
`2.3.5`, safetensors `0.7.0`, and pytest `9.0.2`. The protocol executes CPU FP32/eager
with four intra-op threads. There is no CUDA/MPS, quantization, source substitution,
precision change or tolerance override in this helper.

Transitive packages are resolved during installation and recorded in
`installed_packages.txt`; this is not a universal wheel/hash lock across operating
systems. The official receipt binds its actual runtime fingerprint. Setup checks
native Qwen3 imports before allocating official model tensors, and verifies the
original experiment implementation hash and all nine dependency blobs.

Each `run` creates a unique output directory:

`<work-dir>/checkout/experiments/dual28_ced_v1/outputs/official-p1-<timestamp>-<id>/`

The unmodified official command is:

```bash
python -m experiments.dual28_ced_v1 qualify --source Supa --out <new-output-directory>
```

It checks native B28 parity, source inheritance, H/P matching and identity, D56
identity, active causality/cache/mask grids, pure constructions, H/P/C checkpoint
roundtrips and unchanged source/runtime/code. These are mechanical checks, not
scored capability evaluation. There is no optimization during official P1.

Success requires the runner's `result.json` and a successful recheck using its
own `validate_parent` function against the current binding and retained H/P/C
checkpoint hashes. Only then does the helper write
`<work-dir>/p1-<timestamp>-<id>/completed.json` as `P1_RECEIPT_VALIDATED`.

Keep the complete work directory, its virtual environment, original Git clone,
and all checkpoints. Do not commit evidence into the execution worktree or switch
its HEAD: any head change makes the receipt stale for subsequent execution.
The helper never starts P2. A P1 PASS only permits a separately requested 25-step
engineering smoke; efficacy, recovery, DEV/TEST, GPU-fit, pure migration and
deployment remain closed.

## If anything stops

- Setup: read `preparation_failure.json` and the indicated log. Correct the concrete
  problem and prepare into a **new** work directory. No preparation status is P1.
- P1: keep `launcher_failure.json`, `qualify.log`, and the official output's
  `blocked.json` if present. An OS kill can prevent the runner writing a receipt;
  absence of `result.json` is never a PASS. Preserve checkpoint fragments.
- Weight mismatch: locate the exact source; do not edit the source lock.
- Import/version error: preserve the installation logs and dependency list; do not
  change the pinned recipe to get past the gate.
- Native parity or active-path failure: stop and inspect the failure. Do not relax
  tolerances. Any implementation repair requires a separate branch and fresh P1.
- Resource failure: use a suitable x64 CPU machine with the same prescribed lane;
  no inference about architecture quality or GPU fit follows from an OOM.

To review the outcome, share `result.json` and `execution.json`, or the failure
JSONs and log, together with `installed_packages.txt` and `scoped_tests.log`.
Do not upload the multi-GB checkpoints; keep them locally for receipt validation.

## Helper validation

```bash
python -m unittest discover -s tools/dual28_local_p1 -p 'test_*.py' -v
```

Eleven stdlib tests cover real Git worktree isolation plus synthetic negative
paths. Package installation and model execution are mocked in those tests.
See `verification.json` for the exact delivery scope. They are not the original
94 model-mechanics tests, and they cannot establish an official pretrained result.
