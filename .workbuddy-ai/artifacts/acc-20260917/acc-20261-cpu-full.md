## Acceptance run — ovms-2026.1-tfs-cpu

* gateway: `http://192.168.22.102:28021`
* OVMS: `http://192.168.22.102:28351`
* protocol: `tfs`
* captured at: `2026-09-16T18:48:41.326688+00:00`

### Verdict

| Dimension | Result |
| --- | --- |
| Compatibility | **PASS** |
| Correctness | **PASS** |
| Long-text liveness | **PASS** |
| Concurrency | **PASS** |
| Reranker | **PASS** |
| Zero-resident | **PASS** |
| Failure counters | **PASS** |

### Model matrix

| Model | Kind | Required | Status | HTTP | Dim | Norm | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-embedding-0.6b-int4 | embedding | yes | **PASS** | 200 | 1024 | 1.0 |  |
| qwen3-embedding-0.6b-int8 | embedding | yes | **PASS** | 200 | 1024 | 1.0 |  |
| bge-m3-i8 | embedding | yes | **PASS** | 200 | 1024 | 1.0 |  |
| bge-m3 | embedding | yes | **PASS** | 200 | 1024 | 1.0 |  |
| arctic-embed-m-v2-int8 | embedding | yes | **PASS** | 200 | 768 | 1.0 |  |
| qwen-reranker | rerank | yes | **PASS** | 200 | - | - |  |
| DINO | image_embedding | no | **N/A** | - | - | - | no production OVMS deployment (official DINOv3 blocked on Meta gated weights) |
| multimodal Classic IR | image_embedding | no | **N/A** | - | - | - | export-only track; not present in the production registry |

### Long text

| Target tok | Actual tok | Single | Dim | Norm | Live | Concurrency=2 wall | Live |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 252 | 200 / 1.641s | 1024 | 1.0 | 200 | 0.734s | 200 |
| 512 | 512 | 200 / 0.782s | 1024 | 1.0 | 200 | 1.5s | 200 |
| 1024 | 1022 | 200 / 2.344s | 1024 | 1.0 | 200 | 4.203s | 200 |
| 4591 | 4592 | 200 / 10.031s | 1024 | 1.0 | 200 | 20.672s | 200 |

### Reranker

* biology score: `0.993563367455851`
* Eiffel score: `3.0763067699016474e-05`
* ordering (biology > Eiffel): `True`
* all scores within 0..1: `True`

### Zero-resident

* loaded before request: `['qwen-reranker__cpu', 'qwen3-embedding-0.6b-int4-pooled__cpu']`
* loaded after request: `['qwen3-embedding-0.6b-int4-pooled__cpu']`
* idle unload observed: `True` (100.797s)

### Counters

* HTTP 5xx: `0`
* client exceptions: `0`

Transport failures that were retried once (GET only, idempotent).  Not
counted as failures; if a retry also failed, the attempt additionally
appears in the exception log above.  Listed for transparency:

| method | URL | exception | message |
|---|---|---|---|
| GET | `http://192.168.22.102:28021/v1/broker/metrics` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) |
| GET | `http://192.168.22.102:28021/v1/broker/metrics` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) |
| GET | `http://192.168.22.102:28021/v1/broker/metrics` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) |
| GET | `http://192.168.22.102:28021/v1/broker/metrics` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) |

### Capability capture — ovms-2026.1-tfs-cpu

Captured at `2026-09-16T18:48:42.328813+00:00`, base `http://192.168.22.102:28351`.

| Endpoint | Status | Response |
| --- | --- | --- |
| `GET /v2/health/live` | 200 | `""` |
| `GET /v2/health/ready` | 200 | `""` |
| `GET /v1/config` | 200 | `{"arctic-embed-m-v2-int8__cpu": {"model_version_status": [{"version": "1", "state": "END", "status": {"error_code": "OK", "error_message": "OK"}}]}, "bge-m3-i8__cpu": {"model_version_status": [{"versi...` |
| `GET /v1/models` | 404 | `{"error": "Model with requested name is not found"}` |
| `GET /v3/embeddings` | 400 | `{"error": "Invalid request URL"}` |
| `GET /v1/models/qwen-reranker` | 404 | `{"error": "Model with requested name is not found"}` |
| `GET /v2/models/qwen-reranker` | 404 | `{"error": "Model with requested name is not found"}` |
| `GET /v2/models/qwen-reranker/ready` | 404 | `{"error": "Model with requested name is not found"}` |
| `POST /v1/models/qwen-reranker:predict` | 404 | `{"error": "Model with requested name is not found"}` |
| `POST /v2/models/qwen-reranker/infer` | 400 | `{"error": "Invalid JSON structure, missing inputs"}` |

`OVMS_PROTOCOL=auto` would resolve to **tfs** (source `auto_probe_predict_marker`), evidence: `{"kserve_health_ready_status": 200, "tfs_config_status": 200, "tfs_config_lists_models": false, "tfs_predict_probe_status": 404, "tfs_predict_probe_mediapipe_rejection": false, "tfs_predict_probe_classic_model_lookup": true}`.
