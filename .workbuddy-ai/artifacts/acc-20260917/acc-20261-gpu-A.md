## Acceptance run — ovms-2026.1-tfs-gpu

* gateway: `http://192.168.22.102:28031`
* OVMS: `http://192.168.22.102:28361`
* protocol: `tfs`
* captured at: `2026-09-16T19:02:54.823395+00:00`

### Verdict

| Dimension | Result |
| --- | --- |
| Compatibility | **PASS** |
| Correctness | **PASS** |
| Long-text liveness | **PASS** |
| Concurrency | **PASS** |
| Reranker | **PASS** |
| Zero-resident | **N/A** |
| Failure counters | **PASS** |

### Model matrix

| Model | Kind | Required | Status | HTTP | Dim | Norm | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-embedding-0.6b-int4 | embedding | yes | **PASS** | 200 | 1024 | 0.999785 |  |
| qwen3-embedding-0.6b-int8 | embedding | yes | **PASS** | 200 | 1024 | 1.000161 |  |
| bge-m3-i8 | embedding | yes | **PASS** | 200 | 1024 | 1.0 |  |
| bge-m3 | embedding | yes | **PASS** | 200 | 1024 | 1.0 |  |
| arctic-embed-m-v2-int8 | embedding | yes | **PASS** | 200 | 768 | 1.0 |  |
| qwen-reranker | rerank | yes | **PASS** | 200 | - | - |  |
| DINO | image_embedding | no | **N/A** | - | - | - | no production OVMS deployment (official DINOv3 blocked on Meta gated weights) |
| multimodal Classic IR | image_embedding | no | **N/A** | - | - | - | export-only track; not present in the production registry |

### Long text

| Target tok | Actual tok | Single | Dim | Norm | Live | Concurrency=2 wall | Live |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 252 | 200 / 1.047s | 1024 | 0.999913 | 200 | 1.062s | 200 |
| 512 | 512 | 200 / 0.906s | 1024 | 1.000188 | 200 | 1.812s | 200 |
| 1024 | 1022 | 200 / 2.187s | 1024 | 1.000035 | 200 | 4.375s | 200 |
| 4591 | 4592 | 200 / 10.609s | 1024 | 1.0 | 200 | 21.141s | 200 |

### Reranker

* biology score: `0.9933588873579225`
* Eiffel score: `3.021244067895014e-05`
* ordering (biology > Eiffel): `True`
* all scores within 0..1: `True`

### Zero-resident

* loaded before request: `-`
* loaded after request: `-`
* idle unload observed: `None` (-s)

### Counters

* HTTP 5xx: `0`
* client exceptions: `0`

### Capability capture — ovms-2026.1-tfs-gpu

Captured at `2026-09-16T19:02:56.624383+00:00`, base `http://192.168.22.102:28361`.

| Endpoint | Status | Response |
| --- | --- | --- |
| `GET /v2/health/live` | 200 | `""` |
| `GET /v2/health/ready` | 200 | `""` |
| `GET /v1/config` | 200 | `{}` |
| `GET /v1/models` | 404 | `{"error": "Model with requested name is not found"}` |
| `GET /v3/embeddings` | 400 | `{"error": "Invalid request URL"}` |
| `GET /v1/models/qwen-reranker` | 404 | `{"error": "Model with requested name is not found"}` |
| `GET /v2/models/qwen-reranker` | 404 | `{"error": "Model with requested name is not found"}` |
| `GET /v2/models/qwen-reranker/ready` | 404 | `{"error": "Model with requested name is not found"}` |
| `POST /v1/models/qwen-reranker:predict` | 404 | `{"error": "Model with requested name is not found"}` |
| `POST /v2/models/qwen-reranker/infer` | 400 | `{"error": "Invalid JSON structure, missing inputs"}` |

`OVMS_PROTOCOL=auto` would resolve to **tfs** (source `auto_probe_predict_marker`), evidence: `{"kserve_health_ready_status": 200, "tfs_config_status": 200, "tfs_config_lists_models": false, "tfs_predict_probe_status": 404, "tfs_predict_probe_mediapipe_rejection": false, "tfs_predict_probe_classic_model_lookup": true}`.
