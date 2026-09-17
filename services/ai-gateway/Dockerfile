FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    USE_TORCH=0 \
    USE_TF=0 \
    USE_FLAX=0 \
    TOKENIZERS_PARALLELISM=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install contracts
COPY packages/contracts /packages/contracts
RUN pip install --no-cache-dir /packages/contracts

# Install service dependencies from pyproject.toml
COPY services/ai-gateway/pyproject.toml /app/pyproject.toml
COPY services/ai-gateway/README.md /app/README.md
COPY services/ai-gateway/research_ai_gateway /app/research_ai_gateway
RUN pip install --no-cache-dir /app

EXPOSE 8000

CMD ["uvicorn", "research_ai_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
