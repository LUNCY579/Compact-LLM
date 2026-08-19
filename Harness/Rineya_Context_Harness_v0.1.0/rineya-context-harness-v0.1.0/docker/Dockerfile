# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system --gid 10001 rineya \
    && useradd --system --uid 10001 --gid rineya --home-dir /nonexistent --shell /usr/sbin/nologin rineya

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY migrations ./migrations

RUN python -m pip install --upgrade pip \
    && python -m pip install '.[api]'

COPY configs/docker.yaml ./configs/docker.yaml
COPY artifacts/runtime/.gitkeep ./artifacts/runtime/.gitkeep

RUN mkdir -p /app/artifacts/runtime/traces \
    && chown -R rineya:rineya /app/artifacts

USER 10001:10001

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=2))" || exit 1

ENTRYPOINT ["rineya-harness"]
CMD ["serve", "--config", "/app/configs/docker.yaml"]
