# CPU-only runtime artifact003: independent science review001

2026-10-05. Reviewer `/root/cpu_runtime_control_audit`; coordinator/builder separate. Shared model/workspace and earlier feasibility advice are disclosed. This create-only review statically reads frozen documents, hashes scoped artifacts and reads selected ZIP members in memory. It does not rerun either builder script, refetch metadata, transfer wheels, execute/install/extract target code, import frameworks, touch model assets, observe live host/process/GPU/DB state, register a worker or mutate budgets. Only this report is written.

**Verdict: COMPLETE_ARTIFACT_HASH_LOCK_ACCEPTED_AT_STATIC_SCOPE; ONE OFFLINE SETUP IS CONDITIONALLY JUSTIFIABLE AFTER ITS EXACT EXECUTION/PATH/CONTAINMENT PREREQUISITES.** No material artifact-identity or dependency-closure blocker was found. This is not an unconditional setup approval, complete native-path proof, no-CUDA hardware qualification, actual bootstrap or model result. The frozen records accurately leave runtime boundaries NOT_READY.

## Exact preserved bindings

All31 files listed by FINAL_FREEZE physically match their frozen hashes. The earlier incomplete static checkpoint is retained unchanged and linked by the final batch; completion did not rewrite it into a successful run. The standing receipt and both concrete artifact-scope receipts physically match their declared hashes.

| Artifact | SHA256 |
| --- | --- |
| FINAL_FREEZE.json | `b8e2c626c1112e18944ede5b066082d11288db496f5c721917700ef6f1322fe1` |
| ARTIFACT_BATCH_AND_HASH_LOCK.json | `f95186e2d56c8039063af8a5f923daad76eefeee2b06515b0af669e30ca7279c` |
| requirements-cpu-lock.txt | `a39fae225772267d5973e88f75e949697d4894969a16080bb101b585411475c7` |
| earlier FREEZE.json | `cf3b181d7e3daebf63354844db0d3d7a69d24929bc6d851d96a38db115f3b902` |
| earlier STATIC_INSPECTION_AND_CLOSURE.json | `3cafa8d37d73557113b8c5935d522197bb666ec0047d7e5c73a1a2158660eb91` |
| ACQUISITION_PLAN.json | `d82df9049e0b63c8baa7e708f2b4c2769df5b99fc320f7860f9cc1674336eaef` |
| inspect_static.py | `e2608322bb848d1a16c6f53aaea848ada7475dc5fb4c02d87cd8caf994d23c9d` |
| acquire_batch.py | `99f30c56ceef4090cd2bc6fd822cd7240c9cc0304210f88e9b00b9ad2495c1c5` |
| artifact scope003 | `eef6b84f5fd7d69e5d63160508a4aee6ec116dceecd2dc7dfb66ac674a90d498` |
| prospective metadata-selection amendment004 | `7ddb209c5b6bba259ba4507fb20d92410828f4117726fc1f600ad29d05d142be` |
| standing actual human authority | `479b6c5284da978c491a6477a9064f21b585fa29e9991bf84d75960c380d18f0` |
| original25-version baseline | `c3f09cfb8161a835ae0beff6f3a77afcf88670aaa34b40ac4c80429cfef0d7a7` |
| retained first successful Torch wheel inspection | `e6172d2cba146a0de39ebbcd483ebea98bc0757a8921adb98b0582447795c756` |

The metadata-selection amendment prospectively raises64 selected metadata/source files to100 because25 wheels require75 metadata members before the selected Torch source. It changes no version, device/resource/PE/byte/time limit. Final82 selected metadata/source files/2421905 bytes and42 selected PEs/329942576 bytes fit those explicit amended bounds. Final25 wheels contain152491224 compressed bytes and570999036 declared uncompressed bytes, below the respective768MiB/2GiB limits. Recorded24 metadata responses/831220 bytes,24 new wheel transfers/38854746 bytes,295.07117768478383s stage elapsed and11.498999999999796s cumulative active inspection are builder provenance, not independently remeasured clocks. The prior113636478-byte Torch transfer and failed official transport remain separate history.

## Artifact identity, metadata and closure

I physically hashed all25 wheels against their frozen fields: Torch uses `sha256`; the24 batch items use `physical_sha256`. All match. Each of the24 new wheels also matches its exact filename/size/published SHA256 in the retained official PyPI JSON. No new online observation was needed or performed. Torch physical SHA256 matches the retained published official CPU index pin:

`17a09465bab2aab8f0f273410297133d8d8fb6dd84dccbd252ca4a4f3a111847`

Its exact artifact is `torch-2.10.0+cpu-cp311-cp311-win_amd64.whl`,113636478 bytes, from the recorded official `download.pytorch.org/whl/cpu/` origin. The registered failed r2 action and successful second official transport are preserved; this review authorizes no further transfers.

Independent data-only in-memory reads of all75 WHEEL/METADATA/RECORD members total2220675 bytes. Their hashes and recorded RECORD row counts match; no duplicate RECORD paths were found. Builder source checks ZIP safety, required member coverage and selected metadata/source/PE RECORD digest+size consistency. These are bounded selected member checks, not an independent rehash of every uncompressed member. The physical complete wheel hashes bind the entire official artifacts. No wheel `.pth`, `sitecustomize.py` or `usercustomize.py` members were found in the scoped wheel inventories; external startup/search hooks remain a later path-control question.

The canonicalized25-package baseline is retained. Every non-Torch version is unchanged; Torch alone changes2.10.0+cu126 to2.10.0+cpu. The complete lock has25 exact versions and artifact hashes. Physical wheel Requires-Dist information is checked against the declared Windows/AMD64/CPython3.11.15 marker environment with `extra=""`; all31 active dependency edges resolve to the frozen set and their versions satisfy the retained constraints. Examples that matter are transformers requiring huggingface-hub>=0.34.0,<1.0/tokenizers>=0.22.0,<=0.23.0, requests requiring urllib3>=1.21.1,<1.27, sympy requiring mpmath>=1.1.0,<1.4, and Torch typing-extensions>=4.10.0/sympy>=1.13.3. Torch setuptools is inactive for Python3.11; optree/opt-einsum and other unrequested extras are inactive. Windows colorama is active and present. No mandatory absent dependency, direct dependency URL, active extra or fallback version is reported or needed by this closure.

Tags are the exact cp311-win_amd64 builds where required, compatible cp37/cp38/cp39-abi3-win_amd64 for psutil/safetensors/tokenizers, or pure-Python tags. Python constraints are checked for3.11.15. This establishes selected artifact/ABI compatibility at metadata scope, not successful DLL resolution or import execution.

## CPU backend evidence and remaining boundary

The selected proof binds all ten Torch native payloads,27 dependency native payloads and five base-interpreter executable/DLL files:42 PEs total, recorded AMD64. The helper proof now includes safetensors/tokenizers/psutil, MarkupSafe/PyYAML/regex and NumPy's native files plus bundled OpenBLAS and renamed MSVCP helper. The final complete hashes close the earlier checkpoint's helper-artifact gap. Its inherited `native_boundary_limits` sentence saying helpers must be bound is historical/stale; the final wheel table supplies those bindings. It is not an additional absent artifact.

The physical selected Torch `version.py` states2.10.0+cpu and CUDA/HIP/ROCm/XPU version fieldsNone. Independently read generated Caffe2/Torch CMake data disable CUDA with enclosing `if(0)` and disable HIP/XPU. The inner `if(ON)` around C10_CUDA in TorchConfig is inside that inactive enclosing block, so it must not be presented as enabled CUDA. Recorded `_C` imports torch_python; torch_python imports torch_cpu/c10 and Python311. Selected static import tables do not show a CUDA backend library. Together with the exact official CPU wheel these are substantive CPU-build provenance evidence; absence of a filename alone would not be enough.

They do not prove zero hardware activity for an execution graph. Import tables omit computed LoadLibrary/GetProcAddress targets and internal C dispatch; Python and native helpers possess dynamic-loader capabilities. The loader contains Python-base/userbase-Library-bin/PATH search branches. Unbound system resolution includes MSVCP140, Windows API sets and other direct system imports; actual loaded-module identities, optional plugins and startup routes have not been observed. An imported CUDA-named Python module is metadata, not observed CUDA hardware participation. Neither this static review nor a future installation can qualify no-device dispatch or the old failed classifier/frontier.

## Exact next gate and finite disposition

ONE offline setup may be registered under standing authority after nonbuilder OPS acceptance and a narrowly scoped setup proof. Before executing its installer/interpreter, bind the **one chosen setup executable/installer and exact command**, the existing five base Python executable/DLL identities, any installer-specific directly required native helper, sanitized startup/import environment and canonical allowed DLL roots. Record its resolved destination under the package, exclude old/global/user-site/PYTHONPATH/PYTHONHOME/current-directory shadowing and inherited accelerator/search roots, and declare containment, disk/write/time/journal/unknown-outcome handling. A missing installer identity/path policy is a concrete current setup blocker; the artifact batch supplies no setup command or installer proof.

Any necessary extra static binding must be a preregistered finite manifest of these exact setup dependencies and directly selected named runtime imports, with at most64 files/1GiB and no recursive system-directory audit, target execution, generic coverage implementation or network. System DLL names/API-set contracts must map to canonical allowed system-root/known-resolution evidence, with literal files hashed where present; missing physical API-set stub files are not proof of missing operating-system functionality. Resolve/refuse any relevant unknown search branch rather than silently assume the canonical file loads. New venv launcher bytes can only be pinned after their one creation, before any target bootstrap. Torch/NumPy helper runtime system paths and actual module-resolution witnesses must pass the separately reviewed bootstrap gate; they cannot be conjured by install success. This report does not waive the earlier OPS path/ABI requirement or attest an unreviewed setup implementation.

The setup dose stays one fresh environment and one offline complete hash-locked wheel-only installation, together<=300s and<=2GiB newly written environment bytes, no resolver fallback, dependency fetch, sdist/build hook, target package test/import, bytecode compilation or second environment. Failure/unknown outcome stops with receipts. A source/fixture/exact-commit/admission-qualified no-model bootstrap, and later accepted raw evidence, remain separate prerequisites before a new120s exact Titan load-only dose. Preserve one native worker,2GiB RSS and separately2GiB Job commit, host1.25/pre3.3125/disk4,7200s epoch/14400s rolling24h and47.42100000000573s historical charge. All old loader3/3/GPU/frontier/reader/acquisition quarantines, original keys/Probe002, protected banks and canonical weights remain unchanged. The full scientific mission remains incomplete.
