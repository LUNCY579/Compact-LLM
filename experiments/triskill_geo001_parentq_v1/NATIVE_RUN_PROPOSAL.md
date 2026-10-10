# P0 exact native-run review candidate — blocked, no authorization

P0 precisely specifies the existing generated-bank instrument so its costs and worker
can be reviewed. It is **not launch eligible**: the preserved bank has operative collisions
and unmatched conditional signs; the prior, gate reliability and exposure policy are
unresolved. READINESS_PARENTQ.json is pending, no FREEZE_PARENTQ.json exists, and full CPU
registration/native containment are unqualified. Independent review of this proposal is
recorded as a blocked review, never as permission or parent qualification.

One candidate is supplied in this packet. Direct measurement of the exact eight frozen
parents (16 relevant generations; 24 if all arms cover every parent) is an alternative
requiring an explicit design/exposure decision and a newly reviewed schedule/implementation.
It is not silently substituted for P0. A stronger GEO gate requires a GEO amendment first.

## Exact source and membership

- Worktree/branch/parent HEAD: those in REVISION_PARENTQ.md. Revised uncommitted source
  hashes are in the external verification index; commit all execution dependencies after
  decisions/review. Preserve REVIEW_PARENTQ.md untracked.
- GEO freeze: `f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687`.
  GPU lease: `594e32212f8fb5321d4b42dc13beb068776cf0c190b535d562f98051194161ca`.
- BASE source SHA256: `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`.
  Sparse: `a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284`.
  TT source: `a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567`.
  These are declared frozen identities; a genuine TT backup/custody receipt remains required.
- Rows: unchanged parentq_bank.json (24 T, 32 H_ON, same 32 H_OFF variants), seed
  20261002/order seed 20261003, and unchanged eight-row exposed canary. Hashes in the index.
- Arms: BASE=Supa, T=Sparse, H=FP32(Supa+TT-Sparse), using pinned GEO arm construction.
  Block-14 hook at IDENTITY. No TH/ROTATION/geometric arm, J/K generation or protected bank.
- Runtime: existing read-only hermes Python, numpy 2.4.6, tokenizers .22.2,
  psutil 7.2.2, torch 2.6.0+cu124, transformers 5.14.1, safetensors .8.0.
  No install, model download or other lane's permit. Missing runtime/source identity blocks.

## Exact schedule and caps

Padding parity first: first two frozen development mechanics prompts, each unpadded and
padded for four token steps (16 forwards, upper 1,164 positions). Then:

| Arm, in weight-load order | Units, each in frozen row order | Generations |
|---|---|---:|
| BASE | canary 8; T 24; H_ON 32; H_OFF 32 | 96 |
| H | canary 8; H_ON 32; H_OFF 32 | 72 |
| T | canary 8; T 24 | 32 |
| Total | 24 canary + 176 parent/diagnostic | 200 |

One model load; two in-memory arm weight reloads. Greedy batch one, FP32/eager,
TF32 disabled, deterministic algorithms requested, 288 input slots, at most 32 output
slots, first EOS stop (151645 or 151643). Upper bounds: 6,416 forwards,
64,964 token positions including padding, 540 s worker wall, 600 s supervisor wall,
at most three cumulative load-begun attempts. Additional cleanup waits are separately
bounded (taskkill <=60 s per invocation, launcher wait <=60 s, real-worker polling <=60 s,
sharing retries <=10 s per release path); 600 s is the worker hard wall,
not a promise that receipt cleanup completes by 600 s.

Preload floors: host 6.5 GiB (provisional conservative floor), GPU 6 GiB, disk 6 GiB.
Active reserves: host 1.5 GiB, GPU .5 GiB; CUDA fraction .78. Fresh resource measurement,
source identity and ownership at admission and units are mandatory; yesterday's memory
or free GPU alone is insufficient. A named exclusive slot and native process/memory
containment review are still required. No model checkpoint is produced; attempts,
logs, tokens and caches use the canonical external ledger
`C:/Users/clclc/BusinessProject/triskill-geo001-parentq-runtime/` across all freezes.
Moving to a new root cannot reset accounting.

## Decision and failure interpretation

The existing worker currently implements the historical Dirichlet(.5) prediction with
.70/.40 mean-P thresholds and 7/8 canary text matching. Those constants are preserved
for auditing, not endorsed as calibrated readiness. P0 cannot freeze until the operator
resolves the estimand/calibration, bank and exposure decisions and the implementation
matches that decision. Any membership/rule change makes this exact P0 schedule and hashes
obsolete and requires re-review; no after-outcome choice is permitted.

Padding/canary failure is an identity/mechanics hold with retained diagnostics, not a
parent finding. Preload refusal writes BLOCKED. LOAD_BEGUN before model loading spends
an attempt even on failure/hard kill; partial exposure is retained. Missing coverage is
incomplete, exhausted attempts inconclusive. Mismatched/unverifiable overlap prevents
automatic qualification. BASE ceilings do not imply missing transport, and A/B success
cannot change J's contract. No PARENTS verdict grants a GEO screen. GEO has its own
amendment, freeze, one-shot safety, hardware and grant requirements.

## Exact command form after all blockers are genuinely cleared

The following is a prospective command form, **not an instruction or current permit**.
FREEZE digest, the JSON grant and named slot must come from the approved source-bound
packet and human authorization; none is filled or fabricated here. Use a literal argument
array so grant JSON is not shell-evaluated:

```powershell
$parentqPython = 'C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe'
$parentqLauncher = 'C:/Users/clclc/BusinessProject/ChatBubble-triskill-geo001-sectionh/experiments/triskill_geo001_parentq_v1/parentq_launch_wmi.py'
$parentqRuntime = 'C:/Users/clclc/BusinessProject/triskill-geo001-parentq-runtime'
$parentqFreeze = Join-Path $parentqRuntime 'approved-freeze/FREEZE_PARENTQ.json'
$parentqGrant = Get-Content -Raw -LiteralPath (Join-Path $parentqRuntime 'operator-grant-attempt-0001.json')
$parentqDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $parentqFreeze).Hash.ToLowerInvariant()
& $parentqPython -B -u $parentqLauncher --freeze $parentqFreeze --freeze-sha256 $parentqDigest --grant $parentqGrant --attempt-number 1 --runtime-root $parentqRuntime
```

This command defaults to dry run. Only an actually authorized, reviewed, fully frozen
attempt adds `--launch`. WMI must keep ShowWindow=0. Grant JSON requires exact string
experiment TRISKILL-GEO-001-PARENTQ, integer attempt_number, matching freeze_sha256,
fresh operator_grant text and a real named hardware_slot. The attempt is 1 only if the
canonical historical ledger still has no previous START. No grant file was created.

Review disposition: P0 is a concrete cost/source specification, **blocked for native use**.
Next decide fixed-row feasibility versus calibrated new-world readiness together with
exact-parent exposure eligibility, then revise this one packet and repeat independent
review before any freeze or native parent screen.
