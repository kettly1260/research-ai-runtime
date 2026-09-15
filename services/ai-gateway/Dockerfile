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

# Install python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    transformers \
    tokenizers \
    sentencepiece \
    requests \
    httpx \
    numpy \
    pydantic \
    pyyaml \
    pillow \
    'protobuf>=4.25,<6'

# Copy contracts and install
COPY packages/contracts /packages/contracts
RUN pip install --no-cache-dir -e /packages/contracts

# Copy service code
COPY services/ai-gateway/app /app/app
COPY services/ai-gateway/app/main.py /app/main.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
