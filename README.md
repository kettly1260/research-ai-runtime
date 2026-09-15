# Research AI Runtime

Universal, low-resident, scalable research AI infrastructure.

## Architecture

```text
research-ai-runtime
│
├── ai-gateway          (embeddings, rerank, multimodal, DINO, Device Broker)
│   └── OVMS            (Intel UHD730 GPU / CPU OpenVINO inference)
│
└── research-media      (Parser Manager, Media Ingestion, LanceDB, Search)
    └── Parser Providers (MinerU Cloud/Local, PaddleOCR Cloud/Local, OpenVINO OCR)
```

## Quick Start

### 1. Build and Run with Docker Compose

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### 2. Run Tests

```bash
pytest tests/ -v
```
