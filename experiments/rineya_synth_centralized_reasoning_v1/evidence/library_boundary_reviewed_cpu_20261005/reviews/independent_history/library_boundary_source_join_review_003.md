# Independent DLL boundary source join review

The frozen helper/loader milestone passes 51 focused inert controls plus 19 loader regressions, with all six tested source hashes unchanged and no forbidden imports or real host/runtime/model/cache byte attempts. Receipt: `library_boundary_review_1791244129259022400.json`, body SHA256 `612814bd4e9a1e3ac3e2a946f9ca64ab7a091ed49d40f6f006ae8a9674a1ea5f`.

This is CPU source/control evidence only. The complete native library event observer, current managed-directory registration trace, fresh bootstrap, current runtime bytes and actual host module qualification remain missing. No NativePASS or model/evaluation membership is issued.

## Additional source finding

`prefix_input_owned_worker.py:67-75` checks only `callable(framework_boundary)` in `real_torch_bindings`. `prefix_native_pipeline.py:53-59` forwards the public caller's callback. The installed selected `RegisteredFinder`/guard check occurs only in worker `main` at lines 288-290. A direct future integrated `run_registered` caller could supply `contextlib.nullcontext`, bypassing the actual library guard enclosure while retaining the rest of that route's context/ownership checks. Current stage refusal prevents native execution now; no such native attempt was made.

Required source fix: bind the shared factory or registered entry to the actual installed selected finder and its guard/bound method before native imports. Add a focused inert negative rejecting a no-op or unrelated callable. Callable presence is not observation of an enforced boundary. The earlier positive source-join conclusion covers `main`'s route only.

## Finite registration limitation

`prefix_tensor_contract.py:173` admits System32 DLLs only when their exact paths already occur in inherited65. The new provider requires registration of a complete initial loaded-module snapshot. If that snapshot includes additional System32 dependencies, current `runtime_role` cannot qualify them. Add a separately reviewed finite additive Windows dependency set/role with external current byte pins while preserving inherited65 exactly. No actual loaded host modules or binary bytes were inspected by this reviewer.

## Reviewed root join source anchors

- `prefix_tensor_contract.py`: `35ce106c094207d5b74455820ae856fe090f1fcaf48baf86fb56bd32a6c53e41`
- `prefix_input_owned_worker.py`: `948581df0f6fad7c9570ac9fbacc962d61cfb3d4e5769e33acdcfca30b0592e0`
- `prefix_native_pipeline.py`: `e4fbde1a21871cde75f77b62fb9035fa72b2250eaea2d6b4c1e048b83670fcdc`
- `prefix_registered_loader.py`: `9e3cfdb40a6d8d21b285618af7280bc93fe029c81af91b95ccb0f5ba5ad3b6b1`
- `prefix_library_policy.py`: `c88c5c5116e491cd23b30d3f8d82c5a02bcc92c22133fce6c246d9ebe9e5eb05`

No root/tracked source changes, native probes, framework imports, model reads or launches were performed.
