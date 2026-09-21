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
pytest tests/ services/ai-gateway/tests/ services/research-media/tests/ -v
```

## Document Parser Providers

Research Media uses one declarative provider registry for cloud parsers and
local OCR. There are no MinerU/PaddleOCR-specific Python adapters: HTTP
providers use the generic workflow driver, while command-line OCR uses the
generic command driver.

The checked-in template is:

```text
config/providers.example.yaml
```

Create the runtime file:

```bash
cp config/providers.example.yaml config/providers.yaml
```

`config/providers.yaml` is intentionally ignored by Git because the runtime
file may contain plaintext API keys. URL, API key, model, API mode and
provider-specific options all live in this one file; do not duplicate
MinerU/PaddleOCR settings into Docker environment variables.

### Hot reload

`ProviderRegistry` checks the provider file when the registry is accessed.
A valid change is loaded without restarting `research-media`. Reload is
atomic: if an edit is invalid YAML or fails provider validation, the service
keeps the last known-good providers. `GET /v1/providers` exposes
`config_path` and `config_reload_error` so operators can see which file is
active and whether the latest edit failed.

### MinerU

The single `mineru` provider supports:

- `api_mode: precision`: set `model` to `pipeline`, `vlm`, or
  `MinerU-HTML`. Local files use the pre-signed upload workflow; remote URLs
  use the URL task workflow. The workflow polls the result and extracts the
  returned ZIP/Markdown.
- `api_mode: agent`: uses MinerU Agent's lightweight pipeline. Request-level
  model selection is not used in this mode.

Fill:

```yaml
providers:
  mineru:
    endpoint: https://mineru.net
    api_mode: precision
    model: vlm
    auth:
      type: bearer
      token: YOUR_MINERU_TOKEN
```

### PaddleOCR

The `paddleocr` provider uses the same workflow driver. Change only the
provider model/options, for example:

```yaml
providers:
  paddleocr:
    endpoint: https://paddleocr.aistudio-app.com
    model: PP-StructureV3
    auth:
      type: bearer
      token: YOUR_PADDLEOCR_TOKEN
    options:
      useDocOrientationClassify: false
      useDocUnwarping: false
      useChartRecognition: false
```

The workflow supports both local-file multipart submission and remote
`fileUrl` submission, then polls the asynchronous job and downloads Markdown
and JSONL results.

### Local OCR

Two disabled templates are provided:

- `local_ocr_http`: any local HTTP OCR endpoint using the generic HTTP driver.
- `local_ocr_cli`: any installed OCR CLI using the generic command driver.

Enable and edit one of them when local OCR is available. This is especially
important for private documents: `ParseRequest.privacy=private` is a hard
routing boundary and remote providers are excluded. To use MinerU/PaddleOCR
Cloud, explicitly submit the document as `privacy: public`.

Routing normally follows provider `priority`. A request can set
`preferred_provider` to choose the first attempt without removing the other
eligible providers from the fallback chain.

### Unraid

Recommended persistent paths:

```text
/mnt/user/appdata/ovms/runtime/providers.yaml
/mnt/user/appdata/ovms/media/
```

On the production Unraid host, `research-media` is part of the existing
Compose Manager project at
`/boot/config/plugins/compose.manager/projects/OVMS/docker-compose.yml`.
Production should use the GitHub Actions-built GHCR image pinned by immutable
digest; it does not require a local research-media source/build directory.
Mount the existing OVMS runtime directory read-only as the parser config:

```yaml
volumes:
  - /mnt/user/appdata/ovms/runtime:/config:ro
  - /mnt/user/appdata/ovms/media:/data:rw
environment:
  - PARSER_CONFIG_PATH=/config/providers.yaml
  - MEDIA_DATA_DIR=/data/media.lance
```

Because the provider file can contain API credentials, restrict its host
permissions (for example `chmod 600 providers.yaml`) and do not commit or
export the filled runtime copy.

Check the live registry:

```bash
curl http://localhost:28003/v1/providers
```

Example remote parse:

```bash
curl -X POST http://localhost:28003/v1/document/parse \
  -H 'Content-Type: application/json' \
  -d '{
    "file_url": "https://example.org/paper.pdf",
    "privacy": "public",
    "preferred_provider": "mineru"
  }'
```

## Model Export

### Contract-test IR

Contract mode creates small synthetic OpenVINO graphs that preserve the runtime
tensor contracts. They are for plumbing, OVMS lifecycle, DeviceBroker, and CI
tests only; they are **not pretrained Jina/DINO weights**.

```bash
python scripts/convert_models.py \
  --mode contract \
  --output-dir ./exported_models \
  --catalog-path ./exported_models/model_catalog.json \
  --catalog-models-base /models
```

### Official pretrained IR

Install the optional exporter dependencies in a conversion environment:

```bash
pip install -e '.[model-export]'
```

Convert Meta DINOv3 ViT-S/16 independently first:

```bash
export HF_TOKEN='...'
python scripts/convert_models.py \
  --mode official \
  --models dinov3 \
  --output-dir ./exported_models \
  --catalog-path ./exported_models/model_catalog.json \
  --catalog-models-base /models
```

The official Meta checkpoint is gated on Hugging Face. The account associated
with `HF_TOKEN` must have accepted the DINOv3 license before conversion.

> **Current UHD730 production recommendation (2026-09-15):** use official
> `facebook/dinov2-small` for the image-image DINO backend. DINOv3 remains an
> experimental track: two independent DINOv3 implementations showed materially
> worse GPU/CPU numerical fidelity on the Intel UHD730, while official
> DINOv2-small reached about 0.99996 cosine. See
> `docs/OFFICIAL_DINOV3_ACCEPTANCE_STATUS.md` and
> `docs/PRODUCTION_RUNTIME_STATUS_2026-09-15.md`.

Jina-CLIP-v2 can be converted separately:

```bash
python scripts/convert_models.py \
  --mode official \
  --models jina \
  --output-dir ./exported_models \
  --catalog-path ./exported_models/model_catalog.json \
  --catalog-models-base /models
```

The exporter writes separate `jina-clip-v2-vision` and
`jina-clip-v2-text` physical models, matching the runtime's modality routing.
It also stores tokenizer/preprocessor metadata under `exported_models/assets/`.

Official mode is strict: dependency, authorization, download, conversion, or
IR-signature failures return a non-zero exit code. It never falls back to
contract-test synthetic IR.

Licensing must be reviewed for the intended deployment. The upstream
`jinaai/jina-clip-v2` repository currently declares CC-BY-NC-4.0, while Meta
DINOv3 uses the DINOv3 license and gated model access.
