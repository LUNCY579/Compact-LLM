# Security and Data-Safety Notes

This repository is a local research tool. It does not upload models or datasets by itself.

- Model and dataset paths may contain private material. Keep run directories on trusted storage.
- `trust_remote_code: true` executes model-repository Python code. Leave it `false` unless the checkpoint is trusted and isolated.
- `external` and `official_phase_v45` backends execute local commands. Review configured paths and scripts before running.
- PEFT materialization loads local model code through Transformers/PEFT; use a dedicated virtual environment.
- Every mutable pipeline artifact is resolved beneath its run directory. Absolute/path-traversal escapes are rejected.
- Merge destinations and temporary build directories are rejected when they equal, contain, or are contained by either source checkpoint. Source checkpoints are treated as read-only inputs.
- Existing nonempty named run directories are not reused implicitly. Resume only the exact recorded run, or choose a new run name.
- The tool hashes large model files. Antivirus, sync clients, or concurrent edits can create integrity failures; pause them for research runs.
- Do not commit private checkpoints, generated replies, or sealed evaluation data to a public repository.

No networking, email, deployment, or destructive source-model action is implemented by the pipeline. Filesystem permissions and operating-system isolation remain the final security boundary.
