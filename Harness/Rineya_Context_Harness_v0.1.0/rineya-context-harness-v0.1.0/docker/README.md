# Container profile

The container profile runs the optional FastAPI boundary under a non-root user and keeps external behavior in fixture/read-only mode by default.

```bash
docker compose -f docker/compose.yaml up --build
```

The host port is bound to `127.0.0.1:8765`; the container itself listens on `0.0.0.0` only so Docker forwarding works. Runtime state is stored in a named volume.

Hardening included in the Compose profile:

- non-root UID/GID 10001;
- read-only root filesystem;
- writable named volume only for runtime state;
- all Linux capabilities dropped;
- `no-new-privileges`;
- process, memory, and CPU limits;
- local host binding;
- health check;
- permanent purge disabled;
- live web fetching disabled.

To connect to local model or search services, copy `configs/docker.yaml`, change providers, mount the new file read-only, and qualify the exact network path. `host.docker.internal` is available by default on Docker Desktop; Linux deployments may require an explicit host-gateway mapping.

This profile is not a public deployment template. Public exposure requires authentication, TLS, rate limiting, isolated egress, secret management, and a fresh threat review.
