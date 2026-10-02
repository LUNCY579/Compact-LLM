# TRISKILL-GEO-001 section H amendment: CPU verified and frozen; native behavior unmeasured

**What this is.** The section H fixes from the independent review of the pilot at `09a0514e` were
applied on top of the Codex gapfix successor (`332b7f9b`). The result went through a second independent
review of its own, and those findings were fixed too. The registration is
[AMENDMENT_SECTIONH.md](AMENDMENT_SECTIONH.md).

**Branch and commits.**
- Branch `research/triskill-geo001-sectionh-fixes` in worktree
  `C:/Users/clclc/BusinessProject/ChatBubble-triskill-geo001-sectionh`.
- Implementation commits: `5a82e8d7` (first pass) and `14b733c4` (review fixes).
- The freeze records `14b733c4` as its implementation commit.

**Freeze.** [FREEZE_SECTIONH.json](FREEZE_SECTIONH.json), SHA-256
`f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687`. It pins:
- 32 local files and the six dependency-binding roles;
- the programme's `gpu_lease.py`;
- the source sidecars, weight digests and runtime packages;
- the budget and allocation;
- the passing CPU validation receipt [CPU_VALIDATION_SECTIONH.json](CPU_VALIDATION_SECTIONH.json),
  SHA-256 `4c8a88bc63cebd50810d9ae07a06397ef47149d9e8deafb5612581c62941ca97`.

The freeze builder refuses to build from uncommitted code, and refuses unless a fresh validation passes.

**CPU evidence.**
- **Tests.** [CPU_TESTS_SECTIONH.log](CPU_TESTS_SECTIONH.log): 120 tests and 54 subtests passed, with
  `CUDA_VISIBLE_DEVICES=-1`. They include end-to-end CPU runs of the real worker path, using a tiny
  random Qwen3 with the real vocabulary and pinned tokenizer.
- **Validation receipt:**
  - membership disjoint from development and every legacy row;
  - T rows inside trained support with no modulo-97 wrap;
  - no collision across 1,408 exposed T payloads and δ_T's training rows (canary collides);
  - the HSKILL answerability contract is verbatim;
  - the boundary-rule digest and patterns match;
  - longest prompt 281 ≤ 288, longest reply 27 ≤ 32;
  - allocation 5,024 forwards / 75,562 positions within the 5,100 / 76,000 caps.
- **Bootstrap chain.** Against this freeze on CPU: the supervisor's freeze check and pinned lease
  load, then the worker's freeze validation, the renderer self-test (384 prompts byte-exact) and the
  bound benchmark membership all pass. A wrong digest is refused.

**Accounting.**

| item | count |
|---|---:|
| model loads | 0 |
| model forwards | 0 |
| native generations | 0 |
| training examples or updates | 0 |
| GPU leases taken | 0 |
| paid calls | 0 |

The only checkpoint reads were the single layer-14 `o_proj` tensor of each of the three pinned cores,
plus streaming hashes, both during CPU geometry preparation. No protected bank was opened, and TRISKILL
Stage B remains sealed.

**What still gates a native run.**
1. An explicit operator grant, passed to `launch_wmi.py --launch`.
2. Live resources at preflight. The game held about 6.9 GB of VRAM on 2026-10-01; free RAM has ranged
   from 5.3 to 15 GiB.

**Interpretation limits.** These are stated in AMENDMENT section 9:
- a low prior;
- a likely parent-gate stop;
- low power, so NO_SIGNAL is not evidence of absence;
- unchanged controls C2–C4.

A screen signal would justify only a separately registered, powered study.
