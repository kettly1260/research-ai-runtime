## Acceptance run — ovms-2026.3.1-kserve-gpu

* gateway: `http://192.168.22.102:28032`
* OVMS: `http://192.168.22.102:28362`
* protocol: `kserve`
* captured at: `2026-09-16T19:13:20.972312+00:00`

### Verdict

| Dimension | Result |
| --- | --- |
| Compatibility | **N/A** |
| Correctness | **N/A** |
| Long-text liveness | **N/A** |
| Concurrency | **N/A** |
| Reranker | **N/A** |
| Zero-resident | **PASS** |
| Failure counters | **PASS** |

### Model matrix

| Model | Kind | Required | Status | HTTP | Dim | Norm | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Long text

| Target tok | Actual tok | Single | Dim | Norm | Live | Concurrency=2 wall | Live |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Reranker

* biology score: `-`
* Eiffel score: `-`
* ordering (biology > Eiffel): `None`
* all scores within 0..1: `None`

### Zero-resident

* loaded before request: `[]`
* loaded after request: `['qwen3-embedding-0.6b-int4-pooled__gpu']`
* idle unload observed: `True` (96.875s)

### Counters

* HTTP 5xx: `0`
* client exceptions: `0`

Transport failures that were retried once (GET only, idempotent).  Not
counted as failures; if a retry also failed, the attempt additionally
appears in the exception log above.  Listed for transparency:

| method | URL | exception | message |
|---|---|---|---|
| GET | `http://192.168.22.102:28032/v1/broker/metrics` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) |
| GET | `http://192.168.22.102:28032/v1/broker/metrics` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) |
| GET | `http://192.168.22.102:28032/v1/broker/metrics` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) |
| GET | `http://192.168.22.102:28032/v1/broker/metrics` | ConnectionError | ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response')) |
