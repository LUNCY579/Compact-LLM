# Selected parent extension metadata observation

A bounded source-selected parent observation reproduced the original dispatcher,
supervisor and monitor dependency setup, then imported psutil. All11 declared
preloads were present;10 passed the existing verifier. The exact remaining
mismatch is `psutil._psutil_windows`: its actual module name is `psutil_windows`,
while spec/loader identity uses `psutil._psutil_windows`.

[Actual evidence](evidence/import_parent_metadata_observed_20261009/ROOT_PRESERVATION.json)
records module/spec/file/loader/cache metadata, verifier outcomes, selected-byte
checks, original Win32 RAM/RSS and disk samples, and restored import containers.
The observation completed in0.875s within the60-second bound and unchanged
RAM/RSS/disk limits. Initial diagnostics lacking the real parent preload setup
are preserved as failures; their missing imports do not broaden the application.

This imported selected native parent dependencies, including the psutil extension.
It did not import Torch, create a user CUDA context/tensor/model/optimizer/worker,
or qualify GPU admission, framework effects or model quality. Correct only the
observed extension-name identity while retaining exact descriptor/path/spec/
loader checks, then prepare a reviewed successor under standing authorization.
