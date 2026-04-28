# syntax=docker/dockerfile:1.7
FROM python:3.10-slim AS deps-base
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl bash && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip setuptools wheel

COPY requirements-common.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 12 --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-common.txt

FROM deps-base AS deps-server
COPY requirements-server-extra.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 12 --prefer-binary -r requirements-server-extra.txt

FROM deps-base AS deps-edge
COPY requirements-edge-extra.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 12 --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-edge-extra.txt

FROM python:3.10-slim AS server
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
COPY --from=deps-server /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY server/   ./server/
COPY shared/   ./shared/
COPY config/   ./config/
COPY metrics/  ./metrics/
COPY agents/   ./agents/
COPY security/ ./security/
RUN mkdir -p logs
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=15s \
    CMD curl -f http://127.0.0.1:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim AS edge
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates bash && \
    rm -rf /var/lib/apt/lists/*
COPY --from=deps-edge /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY edge/    ./edge/
COPY shared/  ./shared/
COPY config/  ./config/
COPY metrics/ ./metrics/
COPY data/    ./data/
COPY entrypoint-edge.sh ./entrypoint-edge.sh
RUN chmod +x entrypoint-edge.sh && mkdir -p logs data/processed data/partitions
CMD ["bash", "entrypoint-edge.sh"]
