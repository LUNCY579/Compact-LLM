# Security Notes

- Dataset extraction rejects absolute paths, parent traversal, Windows drive paths, and ZIP symlinks.
- Model output is never executed; Python-literal repair uses `ast.literal_eval` only.
- HTTP API keys are read from environment variables and are not written into configs or manifests.
- The repository does not bundle model weights or credentials.
- Treat `trust_remote_code: true` as code execution and enable it only for a reviewed model repository.
- Run untrusted models and servers in an isolated environment with least privilege.
