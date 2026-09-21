# AI Gateway

Universal inference gateway for Research AI Runtime.

The service exposes a stable HTTP API in front of OVMS and the runtime model
registry, including embedding, reranking, multimodal inference, and on-demand
model lifecycle management.

This file is intentionally kept with the service package because
`services/ai-gateway/pyproject.toml` declares `README.md` as package
metadata. Keeping it present allows clean editable installs in CI and local
development environments.
