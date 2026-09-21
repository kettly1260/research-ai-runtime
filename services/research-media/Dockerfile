ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install contracts
COPY packages/contracts /packages/contracts
RUN pip install --no-cache-dir /packages/contracts

# Install service dependencies from pyproject.toml (includes jmespath, lancedb, pyarrow, etc.)
COPY services/research-media/pyproject.toml /app/pyproject.toml
COPY services/research-media/README.md /app/README.md
COPY services/research-media/research_media /app/research_media
RUN pip install --no-cache-dir /app

# Data directory for LanceDB
RUN mkdir -p /data/media.lance

EXPOSE 8002

CMD ["uvicorn", "research_media.main:app", "--host", "0.0.0.0", "--port", "8002"]
