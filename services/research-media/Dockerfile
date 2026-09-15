FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    requests \
    pyyaml \
    pillow \
    lancedb \
    pyarrow

# Copy contracts and install
COPY packages/contracts /packages/contracts
RUN pip install --no-cache-dir -e /packages/contracts

# Copy service code
COPY services/research-media/app /app/app
COPY services/research-media/app/main.py /app/main.py

# Data directory for LanceDB
RUN mkdir -p /data/media.lance

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
