# OVMS 2026.1 (TFS) vs OVMS 2026.3.1 (KServe v2) — Acceptance Report

Status: **GATE 1 PASSED / GATE 2 PASSED / GATE 3 PASSED / CLEANUP PASSED**
Verdict: **READY_FOR_PRODUCTION_PROMOTION_REVIEW**

Production has **not** been switched. This document is the evidence package for a
promotion decision, not the promotion.

Every number below was measured against real servers in an isolated acceptance
stack on Unraid. Nothing is inferred from documentation. Where a measurement was
not possible, the cell says so and says why.

Run ID: `20260917-kserve-0e7af82`. Raw captures: `.workbuddy-ai/artifacts/acc-20260917/`
(20 files: 10 JSON + 10 markdown, one pair per stack and section).

---

## 1. Gating model

| Gate | Scope | Environment | State |
| --- | --- | --- | --- |
| **Gate 1 — CPU / protocol / functional** | adapter correctness, wire format, error semantics, regression | isolated worktree, CPU only | **PASSED** (§4) |
| **Gate 2 — CPU acceptance against real OVMS** | 2026.1 TFS + 2026.3.1 KServe on CPU, capability capture, model matrix, long text, reranker, zero-resident, GenAI v3 | Unraid, isolated `acc-*` stack, no VM | **PASSED** (§5) |
| **Gate 3 — UHD730 GPU acceptance** | real GPU latency, memory, load times, spillover, long-text liveness | Unraid, isolated `acc-*` GPU stack | **PASSED** (§6) |

Three rules governed everything:

1. **A CPU PASS is not a GPU acceptance PASS.** CPU correctness says nothing about
   the UHD730 GPU plugin, FP16 handling, memory pressure or spill behaviour — so
   Gate 3 re-runs the whole matrix on the GPU rather than extrapolating.
2. **Nothing touched production.** Production `ovms-server`, `ai-gateway` and
   `research-memory-gateway` were never restarted, reconfigured or written to,
   and `/mnt/user/appdata/ovms/runtime` was never mounted. Proven in §11.
3. **A gate with no Cleanup PASS is not complete.** Cleanup is a gate, not a
   chore (§11).

---

## 2. Artefacts under test

| Item | Value |
| --- | --- |
| Commit under test | `5fb9073` (adapter `a4b035a` + protocol fix `18350fa` + broker fix `b778afa` + harness fix `5fb9073`) |
| Base commit (validated production long-text code) | `5567dbb8190d8315877389a92c5157f23a8a7c2a` |
| Branch | `feat-ovms-kserve-v2-adapter` |
| Production gateway image | `research-ai-runtime-gateway:local-5567-hotfix` (`sha256:4dbd6160e267…`) |
| Acceptance gateway image | `research-ai-runtime-gateway:acc-20260917-kserve-0e7af82` (`sha256:a08e89013802…`), built as a thin layer over the frozen production image |
| Production OVMS digest | `openvino/model_server@sha256:22f92a1a5ad6784384e47296c43bde239887a623e75de45b0e1802659de416a6` |
| Acceptance OVMS 2026.1 | same digest, pinned — **not** a tag |
| Acceptance OVMS 2026.3.1 | `openvino/model_server@sha256:52c86504cc2f4a86c2c58b558656bf8f6b52745c7bef123b262ac9f8841f4533` |
| Hardware (Gate 3) | Unraid host, Intel UHD 730 iGPU (`/dev/dri/renderD128`) |

Versions actually reported by the servers at startup, not assumed:

| | 2026.1 | 2026.3.1 |
| --- | --- | --- |
| OVMS | `2026.1.0.72cc06244` | `2026.3.1.3a28d490b` |
| OpenVINO backend | `2026.1.0-21367-63e31528c62-releases/2026/1` | `2026.3.1-22476-759c5a6ab8c-releases/2026/3` |
| OpenVINO GenAI backend | `2026.1.0.0-2957-1dabb8c2255` | `2026.3.1.0-3290-56d9685302d` |
| Devices reported | `CPU, GPU` | `CPU, GPU` |

`latest-gpu` was never used as an acceptance configuration. It had already
drifted from 2026.1 to 2026.3.1, which is precisely the drift this task exists to
survive: a tag is not a compatibility contract.

---

## 3. What changed in the gateway

`ovms_predict()` keeps its signature and its canonical payload, so no business
module changed. The protocol decision moved into a new adapter package:

```
inference/{embeddings,rerank,dino,multimodal}.py
                    │
                    ▼
              ovms_predict()                 ovms_client.py
                    │
            ovms_protocol.resolve_adapter()
            ┌───────┴────────┐
            ▼                ▼
      TfsAdapter        KserveAdapter
   /v1/models/{m}:predict  /v2/models/{m}/infer
      (2026.1)              (2026.3+)
```

* Canonical request shape is unchanged: `{"instances": [{"<tensor>": [...]}]}`.
* Canonical response shape is unchanged: `{"predictions": [...]}`.
* No model-specific tensor name, dtype or output shape is hardcoded.
* `OVMS_PROTOCOL=tfs|kserve|auto`, default `tfs`.
* `auto` probes live capability, caches once per process, and **fails closed**
  (502) rather than silently falling back to a different protocol.

### 3.1 Three real defects the acceptance run found

These are the substantive output of Gate 2/3. Each was found by running against
real servers, and each is fixed with a regression test.

**D1 — `auto` resolved to the wrong protocol (fixed in `18350fa`).**
The original ladder required `model_config_list` in the `/v1/config` body and
treated `/v2/health/ready` as KServe-only. Both assumptions are false on real
servers. Measured with nothing resident:

| Signal | 2026.1 | 2026.3.1 |
| --- | --- | --- |
| `GET /v2/health/ready` | 200 | 200 |
| `GET /v1/config` | 200 `{}` | 200 `{}` |
| `GET /v1/config` (models resident) | 200, **flat map keyed by model name** — never `model_config_list` | 200 `{}` |

So the ladder resolved to `kserve` even against a 2026.1 server, and the answer
depended on whether a model happened to be loaded. It is now keyed on the
negative predict probe, which discriminates cleanly and cannot cold-load
anything:

| Probe | 2026.1 | 2026.3.1 |
| --- | --- | --- |
| `POST /v1/models/{unknown}:predict` | 404 `Model with requested name is not found` (Classic Model registry) | 412 `The file is not valid json - model field is missing in JSON body` (MediaPipe handler) |

Verified live afterwards: 2026.1 → `tfs` (`classic_model_lookup: true`),
2026.3.1 → `kserve` (`mediapipe_rejection: true`).

**D2 — the registry's `preferred_device` was ignored on the embedding paths
(fixed in `b778afa`).** `main.py` reads `preferred_device` from the model
registry for all seven routes it owns; the three embedding call sites in
`inference/embeddings.py` called `broker.lease(model_name)` with no preference,
silently taking `lease()`'s `"GPU"` default. On a GPU-less deployment that meant:

* every embedding request attempted a doomed GPU alias first and paid a full
  `_wait_for_model_state` timeout (`OVMS_TIMEOUT`, 120 s in the acceptance stack)
  before falling back to CPU;
* the failed alias was left in `config.json` — `_set_model_enabled` only rolls
  back aliases tracked in `model_loaded`, which a failed load never reaches — so
  OVMS re-attempted the failing compile **once per filesystem-poll interval for
  the lifetime of the container**: 579 occurrences in a single run;
* the deployment could therefore never reach a true zero-resident state, which
  made the zero-resident gate unprovable.

`lease()`'s `preferred_device` now defaults to `None`, resolving from the
registry, with explicit `"GPU"`/`"CPU"` still overriding (so `main.py` and all
GPU-host behaviour are unchanged). A definitively failed alias is now rolled back
out of `config.json` via a non-blocking `_set_model_enabled(..., wait=False)`.
Measured after the fix: `gpu_inferences: 0`, only `__cpu` aliases in
`config.json`, cold pooled-int4 down from a 120 s timeout to **9.90 s**.

**D3 — the failure counter was unactionable (fixed in `5fb9073`).** Every run
ended with `client exceptions: 1` and no way to tell what failed, because
`calibrate_text` probes and retries and so a transient failure there is invisible
in the per-row results while still tripping the gate. The counter now records
method, URL and exception type. The attributed cause was a stale keep-alive
connection reset by uvicorn on a GET to the gateway's own `/v1/broker/metrics` —
a client-side artifact, with the immediately following poll succeeding.
`_timed_get` now retries once on a transport-level disconnect; POST is
deliberately **not** retried, because a POST already processed before the drop
would be duplicated, and a genuinely unreachable server still fails both
attempts and is counted.

### 3.2 A fourth finding, recorded but not a defect

**`GET /v1/config` on 2026.1 returns a flat map, not `{"model_config_list": []}`.**
With models resident the body is `{"<alias>__cpu": {"model_version_status": [...]}}`.
Anything that parses `model_config_list` from the TFS config endpoint is parsing
a key that never appears. The adapter does not depend on it; the old `auto`
ladder did, which is D1.

---

## 4. Gate 1 — CPU / protocol / functional (PASSED)

Adapter correctness, wire format, error semantics and regression, all offline
against a scripted OVMS double. 207 passed, 2 skipped.

New coverage added by this work:

| Area | What is pinned |
| --- | --- |
| `auto` detection | the verbatim captured surfaces of both versions; a regression guard that 2026.1 does not flip when nothing is resident; fail-closed on an unreachable backend; fail-closed on an inconclusive probe; caching; thread-safety (exactly one probe); 502 mapping |
| Protocol contract | a real HTTP fake modelling both servers, including the zero-resident 2026.1 case and the MediaPipe 412 |
| Broker device preference | registry resolution, explicit override, non-blocking rollback, and the no-leak invariant |
| Acceptance harness | gate logic, `N/A` never silently `PASS`, retry semantics, `/v3` alias resolution |

---

## 5. Gate 2 — CPU acceptance against real OVMS (PASSED)

CPU-only, directly on Unraid. No VM. No `/dev/dri`, not privileged, no
level_zero selector, `target_device=CPU`, `restart: "no"`, own network
`acc-20260917-kserve-0e7af82-net`, own ports (28351/28352 OVMS, 28021/28022
gateway), own runtime directories, never joined the production `ai_network`.
Production models and tokenizers were mounted read-only; the production runtime
directory was never mounted.

Run with the committed harness — no hand-rolled curl, no hand-filled tables:

```
python scripts/ovms_acceptance_suite.py \
    --gateway-base http://192.168.22.102:28021 --ovms-base http://192.168.22.102:28351 \
    --protocol tfs --label ovms-2026.1-tfs-cpu --sections capability,matrix,long_text,reranker,zero_resident
python scripts/ovms_acceptance_suite.py \
    --gateway-base http://192.168.22.102:28022 --ovms-base http://192.168.22.102:28352 \
    --protocol kserve --label ovms-2026.3.1-kserve-cpu --sections capability,matrix,long_text,reranker,zero_resident
```

**CPU Gate 2 makes no performance verdict.** It judges protocol, correctness and
liveness only; the performance comparison is §6/§7.

### 5.1 Verdicts

| Dimension | 2026.1 + TFS | 2026.3.1 + KServe |
| --- | --- | --- |
| Compatibility | **PASS** | **PASS** |
| Correctness | **PASS** | **PASS** |
| Long-text liveness | **PASS** | **PASS** |
| Concurrency | **PASS** | **PASS** |
| Reranker | **PASS** | **PASS** |
| Zero-resident | **PASS** | **PASS** |
| Failure counters | **PASS** | **PASS** |
| HTTP 5xx / client exceptions | 0 / 0 | 0 / 0 |

### 5.2 API capability capture (recorded from the real servers)

| Endpoint | 2026.1 | 2026.3.1 |
| --- | --- | --- |
| `GET /v2/health/live` | 200 `""` | 200 `""` |
| `GET /v2/health/ready` | 200 `""` | 200 `""` |
| `GET /v1/config` | 200 `{}` (zero-resident) | 200 `{}` |
| `GET /v1/models` | 404 `Model with requested name is not found` | 200 `{"data": [], "object": "list"}` |
| `GET /v3/embeddings` | 400 `Invalid request URL` | 400 `Invalid request URL` |
| `GET /v1/models/qwen-reranker` | 404 `Model with requested name is not found` | **500** `{"error": "Model not found"}` |
| `GET /v2/models/qwen-reranker` | 404 | 404 |
| `GET /v2/models/qwen-reranker/ready` | 404 | 404 |
| `POST /v1/models/qwen-reranker:predict` | 404 Classic Model lookup | **412** MediaPipe rejection |
| `POST /v2/models/qwen-reranker/infer` | 400 `missing inputs` | 400 `missing inputs` |

`OVMS_PROTOCOL=auto` resolved to **tfs** on 2026.1 and **kserve** on 2026.3.1,
both via `auto_probe_predict_marker`.

### 5.3 Model matrix — Classic TFS / KServe v2 compatibility

Identical on both stacks:

| Model | Kind | Required | Status | HTTP | Dim | Norm |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-embedding-0.6b-int4` (pooled IR) | embedding | yes | **PASS** | 200 | 1024 | 1.0 |
| `qwen3-embedding-0.6b-int8` (GenAI v3) | embedding | yes | **PASS** | 200 | 1024 | 1.0 |
| `bge-m3-i8` | embedding | yes | **PASS** | 200 | 1024 | 1.0 |
| `bge-m3` | embedding | yes | **PASS** | 200 | 1024 | 1.0 |
| `arctic-embed-m-v2-int8` | embedding | yes | **PASS** | 200 | **768** | 1.0 |
| `qwen-reranker` | rerank | yes | **PASS** | 200 | – | – |
| DINO | image_embedding | no | **N/A** | – | – | – |
| multimodal Classic IR | image_embedding | no | **N/A** | – | – | – |

The BGE-M3 deployment path is exercised through the real registry mapping
(`bge-m3` → `ovms_model: bge-m3-i8`), not a synthetic alias.

**DINO / multimodal — `N/A`, with evidence.** This is not a Compatibility FAIL,
because there is no current production deployment to regress. Evidence, read from
the production runtime:

* production `model_catalog.json` contains exactly six models —
  `arctic-embed-m-v2-int8`, `bge-m3-i8`, `qwen-reranker`, `qwen3-embedding-0.6b`,
  `qwen3-embedding-0.6b-int4`, `qwen3-embedding-0.6b-int4-pooled`;
* production `model_registry.json` exposes exactly six logical models — the same
  set;
* `grep -ic 'dino|clip|multimodal|jina'` over both files returns **0**;
* no DINO/CLIP directory exists under `/mnt/user/appdata/ovms/models/`;
* the gateway's own `DEFAULT_MODEL_REGISTRY` documents itself as mirroring the
  production set exactly, precisely so a transient config read failure cannot
  advertise a model that is not deployed (naming the jina-clip-v2 / dinov3
  export-only tracks as the example).

So the image-embedding tracks exist only as export artefacts; there is no
production path through which an OVMS upgrade could regress them.

### 5.4 Long-text ladder and liveness

Text length is calibrated from the gateway's own `usage.prompt_tokens`, so the
ladder measures tokens rather than assuming a characters-per-token ratio. A short
embedding is fired **immediately after** each long round as the liveness probe.

**2026.1 + TFS (CPU)**

| Target tok | Actual tok | Single | Dim | Norm | Liveness after | Concurrency=2 wall | Liveness after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 252 | 200 / 1.641 s | 1024 | 1.0 | **200** | 0.734 s | **200** |
| 512 | 512 | 200 / 0.782 s | 1024 | 1.0 | **200** | 1.500 s | **200** |
| 1024 | 1022 | 200 / 2.344 s | 1024 | 1.0 | **200** | 4.203 s | **200** |
| 4591 | 4592 | 200 / 10.031 s | 1024 | 1.0 | **200** | 20.672 s | **200** |

**2026.3.1 + KServe (CPU)**

| Target tok | Actual tok | Single | Dim | Norm | Liveness after | Concurrency=2 wall | Liveness after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 252 | 200 / 1.250 s | 1024 | 1.0 | **200** | 0.703 s | **200** |
| 512 | 512 | 200 / 0.813 s | 1024 | 1.0 | **200** | 1.531 s | **200** |
| 1024 | 1022 | 200 / 2.500 s | 1024 | 1.0 | **200** | 4.656 s | **200** |
| 4591 | 4592 | 200 / 10.328 s | 1024 | 1.0 | **200** | 20.187 s | **200** |

All 16 long-text responses and all 16 liveness probes returned 200 with 1024
dimensions and unit norm. **No long→short liveness failure on either version.**

### 5.5 Reranker correctness

| | biology score | Eiffel score | ordering | 0..1 range |
| --- | --- | --- | --- | --- |
| 2026.1 + TFS | `0.993563367455851` | `3.0763067699016474e-05` | **True** | **True** |
| 2026.3.1 + KServe | `0.993563324763129` | `3.076311903888812e-05` | **True** | **True** |

Scores agree to ~9 significant figures across a protocol migration, which is the
strongest available evidence that the KServe adapter reproduces TFS inference
semantics rather than merely returning a 200.

### 5.6 Zero-resident lifecycle

| | unload observed | unload time |
| --- | --- | --- |
| 2026.1 + TFS | **True** | 100.797 s |
| 2026.3.1 + KServe | **True** | 95.656 s |

Observed through the gateway's own `/v1/broker/metrics`; the harness never writes
a runtime config. Idle unload TTL was deliberately shortened to 90 s for the
acceptance stack (production uses 1200 s) so the path is observable inside the
window; that change is confined to the acceptance compose.

### 5.7 GenAI v3 `/v3/embeddings` — independent gate

`qwen3-embedding-0.6b-int8` is served by the GenAI v3 OpenAI-compatible surface,
which is **outside** the TFS/KServe Classic Model split the adapter covers. The
adapter's tests therefore say nothing about it, and "not covered by the adapter"
is not "safe to skip": if 2026.3.1 had removed or changed `/v3/embeddings`, the
overall promotion readiness would have to be FAIL/PARTIAL even with a green
KServe adapter.

Run with the committed probe:

```
python scripts/ovms_v3_embeddings_probe.py --gateway-base … --ovms-base … --allow-inference-probes
```

| Probe | 2026.1 | 2026.3.1 |
| --- | --- | --- |
| `GET /v3/embeddings` | 400 `Invalid request URL` | 400 `Invalid request URL` |
| `POST /v3/embeddings` (no model) | 412 | 412 |
| `POST /v3/embeddings model=…__cpu` | **200**, 1024-d, finite, ‖v‖ = 1.0 | **200**, 1024-d, finite, ‖v‖ = 0.999999 |
| gateway cold | **2.872 s** | **2.865 s** |
| gateway warm | 0.042 s | 0.048 s |

**Verdict: PASS on both.** Endpoint presence is established by a real POST — a
GET cannot establish it, since both versions answer `400 Invalid request URL`,
which is indistinguishable from "no such route" in this API.

Cold is measured at the gateway, and the report says why rather than implying a
cold OVMS-level number it does not have: a genuinely cold OVMS-level call is
unreachable through the supported path, because the broker owns the runtime
config and removes an idle alias, after which OVMS answers 404.

---

## 6. Gate 3 — UHD730 GPU acceptance (PASSED)

A separate GPU stack, isolated the same way, adding only `/dev/dri:/dev/dri`.
**Not privileged** — `/dev/dri` is mode 0777 (`root:video`) on this host, so an
unprivileged container can open it, and no permission shortfall was observed, so
privileged was never escalated to. Own network, own ports (28361/28362 OVMS,
28031/28032 gateway), own runtime directories, `restart: "no"`, never connected
to the production runtime.

### 6.1 Verdicts

| Dimension | 2026.1 + TFS (GPU) | 2026.3.1 + KServe (GPU) |
| --- | --- | --- |
| Compatibility | **PASS** | **PASS** |
| Correctness | **PASS** | **PASS** |
| Long-text liveness | **PASS** | **PASS** |
| Concurrency | **PASS** | **PASS** |
| Reranker | **PASS** | **PASS** |
| Zero-resident | **PASS** | **PASS** |
| Failure counters | **PASS** | **PASS** |
| HTTP 5xx / client exceptions | 0 / 0 | 0 / 0 |

### 6.2 Model load, memory and InferRequests

| Metric | 2026.1 + TFS | 2026.3.1 + KServe |
| --- | --- | --- |
| `qwen3-embedding-0.6b-int4-pooled` load | 19.67 s | 13.73 s |
| `bge-m3-i8` load | 17.22 s | 15.71 s |
| `arctic-embed-m-v2-int8` load | 7.80 s | 9.46 s |
| `qwen-reranker` load | 21.41 s | 33.05 s |
| InferRequests (`arctic` / `pooled-int4`) | 1 / 1 | 1 / 1 |
| InferRequests (`qwen-reranker`) | 4 | 4 |
| OVMS container memory, 2 models resident | **3.166 GiB** | **2.997 GiB** |
| OVMS container memory, idle | 103.2 MiB | 108.5 MiB |
| `Cannot compile model` errors | 0 | **0** |

The GPU is a UHD 730 iGPU with unified memory, so GPU-resident allocations appear
as the OVMS container's memory; that is what is reported, and it is labelled as
such rather than presented as a discrete-VRAM figure.

### 6.3 Broker behaviour — GPU-first, zero spillover

| Metric | 2026.1 + TFS | 2026.3.1 + KServe |
| --- | --- | --- |
| `gpu_inferences` | 51 | 51 |
| `cpu_spillover_inferences` | **0** | **0** |
| `cpu_coexistence_inferences` | **0** | **0** |
| `evictions` | 8 | 8 |

Every inference ran on the GPU; nothing spilled to CPU. All runtime aliases
carried `target_device: GPU` and no `__cpu` alias was created. This is also the
direct confirmation that D2's fix is behaviour-preserving on a GPU host: the
registry says `GPU_PREFERRED` there, so resolving `preferred_device` from it
yields exactly the previous GPU-first placement.

### 6.4 Long-text ladder and liveness (GPU)

**2026.1 + TFS (GPU)**

| Target tok | Actual tok | Single | Dim | Norm | Liveness after | Concurrency=2 wall | Liveness after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 252 | 200 / 1.047 s | 1024 | 0.999913 | **200** | 1.062 s | **200** |
| 512 | 512 | 200 / 0.906 s | 1024 | 1.000188 | **200** | 1.812 s | **200** |
| 1024 | 1022 | 200 / 2.187 s | 1024 | 1.000035 | **200** | 4.375 s | **200** |
| 4591 | 4592 | 200 / 10.609 s | 1024 | 1.0 | **200** | 21.141 s | **200** |

**2026.3.1 + KServe (GPU)**

| Target tok | Actual tok | Single | Dim | Norm | Liveness after | Concurrency=2 wall | Liveness after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 252 | 200 / 1.312 s | 1024 | 0.999968 | **200** | 0.906 s | **200** |
| 512 | 512 | 200 / 0.906 s | 1024 | 0.999963 | **200** | 1.812 s | **200** |
| 1024 | 1022 | 200 / 2.218 s | 1024 | 0.999902 | **200** | 4.391 s | **200** |
| 4591 | 4592 | 200 / 10.640 s | 1024 | 1.0 | **200** | 21.219 s | **200** |

Model-matrix norms on GPU are 0.999785 / 1.000161 / 0.999902–1.0 depending on
model — FP16 rounding, identical across both OVMS versions, and inside the
harness's tolerance. Nothing is non-finite.

### 6.5 Reranker and zero-resident (GPU)

| | biology | Eiffel | ordering | range |
| --- | --- | --- | --- | --- |
| 2026.1 + TFS | `0.9933588873579225` | `3.021244067895014e-05` | **True** | **True** |
| 2026.3.1 + KServe | `0.9933588873579225` | `3.021244067895014e-05` | **True** | **True** |

| | loaded before | loaded after request | idle unload | unload time |
| --- | --- | --- | --- | --- |
| 2026.1 + TFS | `[]` | `['qwen3-embedding-0.6b-int4-pooled__gpu']` | **True** | 101.203 s |
| 2026.3.1 + KServe | `[]` | `['qwen3-embedding-0.6b-int4-pooled__gpu']` | **True** | 96.875 s |

Both GPU stacks started from a genuine zero-resident state `[]` — the direct
proof that D2's fix removed the leaked alias.

### 6.6 `/v3/embeddings` on GPU

| Probe | 2026.1 | 2026.3.1 |
| --- | --- | --- |
| `POST /v3/embeddings model=…__gpu` | **200**, 1024-d, finite, ‖v‖ = 0.999705 | **200**, 1024-d, finite, ‖v‖ = 0.999705 |
| gateway cold | 0.085 s | 0.117 s |
| gateway warm | 0.063 s | 0.064 s |

### 6.7 GPU contention

Declared as required. Containers mapping `/dev/dri` on this host: `immich`,
`immich-machine-learning`, `emby`, and production `ovms-server`.

Sampled during the acceptance window, all `/dev/dri` holders were idle
(`immich` 0.06 %, `ovms-server` 0.07 %, `immich-machine-learning` 0.11 %,
`emby` 0.00 % CPU). **The GPU was not contended, so the Gate 3 performance
figures are `VALID` rather than `CONTENDED / INVALID_FOR_PERFORMANCE`.**

One caveat, stated rather than buried: the host CPU was busy with unrelated
workloads during the window (`wechat-agent-*` containers at 54–120 % CPU). That
can inflate CPU-side stages of the long-text path (tokenisation, chunk
preparation) without touching GPU compute. §7.2 shows the measured numbers
reproduce the pre-existing production baseline within ~0.2 %, which bounds the
practical impact — but the caveat stands for any future tighter benchmark.

---

## 7. 2026.1 vs 2026.3.1 benchmark

### 7.1 CPU (correctness and liveness only — no performance verdict)

CPU Gate 2 is explicitly not a performance gate, so these numbers are reported
for completeness and **must not** be read as a version ranking.

| Metric | 2026.1 + TFS | 2026.3.1 + KServe | Delta |
| --- | --- | --- | --- |
| 4591-tok single | 10.031 s | 10.328 s | +3.0 % |
| 4591-tok × 2 wall | 20.672 s | 20.187 s | −2.3 % |
| 1024-tok single | 2.344 s | 2.500 s | +6.7 % |
| Reranker ordering | True | True | — |
| Zero-resident unload | 100.797 s | 95.656 s | −5.1 % |

Deltas of this size are inside run-to-run noise for a CPU stack sharing a host
with unrelated workloads.

### 7.2 GPU (UHD730) — the performance comparison

| Metric | 2026.1 + TFS | 2026.3.1 + KServe | Delta |
| --- | --- | --- | --- |
| 4591-tok single | 10.609 s | 10.640 s | +0.3 % |
| 4591-tok × 2 wall | 21.141 s | 21.219 s | +0.4 % |
| 1024-tok single | 2.187 s | 2.218 s | +1.4 % |
| 512-tok single | 0.906 s | 0.906 s | 0.0 % |
| `pooled-int4` load | 19.67 s | 13.73 s | −30 % |
| `bge-m3-i8` load | 17.22 s | 15.71 s | −8.8 % |
| `arctic` load | 7.80 s | 9.46 s | +21 % |
| `qwen-reranker` load | 21.41 s | 33.05 s | +54 % |
| Resident memory (2 models) | 3.166 GiB | 2.997 GiB | −5.3 % |
| `/v3` cold / warm | 0.085 / 0.063 s | 0.117 / 0.064 s | warm parity |
| Reranker scores | 0.9933588873579225 | 0.9933588873579225 | identical |

**Reading of this table:**

* **Steady-state inference is at parity.** Every ≤1.4 % delta is noise. The
  KServe migration costs nothing measurable on the UHD730 for this workload.
* **Cold model load moves in both directions and is not a regression signal.**
  `pooled-int4` and `bge-m3-i8` load faster on 2026.3.1; `arctic` and
  `qwen-reranker` load slower. These are single samples on a shared host with
  unrelated CPU load, so the honest conclusion is "no established load-time
  regression", not "2026.3.1 loads faster".
* **Reranker output is bit-identical** across the migration, which is the
  strongest correctness signal in the whole report.
* **The long-text path does not benefit from the GPU.** 4591 tokens takes ~10.6 s
  on the UHD730 versus ~10.0–10.3 s on CPU. That is expected, not a fault: the
  pooled-IR path is dominated by tokenisation, chunk preparation and per-chunk
  serialisation (`POOLED_CHUNK_BATCH_SIZE=1`), which are CPU-side, and the
  per-chunk inference is small. Worth knowing before anyone budgets a GPU for
  long-text throughput.
* **Validity check against the pre-existing production baseline.** The task brief
  records production 2026.1 as 4591-tok × 2 wall = **21.11 s**. This acceptance
  stack measured **21.141 s** (2026.1) and **21.219 s** (2026.3.1) — within
  0.2–0.5 %. The isolated acceptance stack therefore reproduces the production
  baseline, which is the best available evidence that these numbers are
  representative rather than an artefact of the test rig.

---

## 8. Verdict table

| Requirement | Result | Evidence |
| --- | --- | --- |
| Classic TFS (2026.1) compatibility | **PASS** | §5.2, §5.3 |
| KServe v2 (2026.3.1) compatibility | **PASS** | §5.2, §5.3 |
| `OVMS_PROTOCOL=auto` correctness | **PASS** | §5.2, after fixing D1 |
| GenAI v3 `/v3/embeddings` on 2026.1 | **PASS** | §5.7 |
| GenAI v3 `/v3/embeddings` on 2026.3.1 | **PASS** — not removed, not changed | §5.7, §6.6 |
| Long-text 256/512/1024/~4591 | **PASS** on 4 stacks | §5.4, §6.4 |
| Long→short liveness after every long round | **PASS**, 32/32 probes 200 | §5.4, §6.4 |
| Concurrency = 2 | **PASS** on 4 stacks | §5.4, §6.4 |
| Reranker correctness | **PASS**, ordering and range | §5.5, §6.5 |
| Zero-resident load and unload | **PASS** on 4 stacks | §5.6, §6.5 |
| Qwen3 INT8 not skipped | **PASS** | §5.3, §5.7 |
| DINO / multimodal | `N/A` — no production path, with evidence | §5.3 |
| HTTP 5xx / client exceptions | 0 / 0 on all 4 stacks | §5.1, §6.1 |
| CPU Gate (2026.1 + 2026.3.1) | **PASS** | §5 |
| UHD730 Gate (2026.1 + 2026.3.1) | **PASS**, `VALID` (GPU uncontended) | §6 |
| Production untouched | **PASS** — sections byte-identical | §11 |
| Cleanup | **PASS** | §11 |
| Proxy contamination | Eliminated — `trust_env=False` by default | §9 |

---

## 9. Proxy contamination

This machine exports `HTTP_PROXY` / `HTTPS_PROXY`, and `requests` honours them —
so an acceptance run would otherwise be routed through the proxy and every
latency number would describe the proxy instead of the target. The symptom is a
`ReadTimeout` naming the *proxy's* port rather than the target's.

Both acceptance scripts build a `requests.Session` with `trust_env = False` by
default; `--use-env-proxy` opts back in explicitly. Every run in this report
printed `[info] ignoring ambient proxy vars ['HTTP_PROXY', 'HTTPS_PROXY']`, and
each capture records `trust_env_proxy: false`. **No number in this report was
measured through a proxy.**

---

## 10. Production posture

* Production stays on OVMS **2026.1** at digest `sha256:22f92a1a…`, with
  `OVMS_PROTOCOL=tfs`.
* Production `ovms-server`, `ai-gateway` and `research-memory-gateway` were
  never restarted, never reconfigured, and never written to.
* The production runtime directory was never mounted by any acceptance
  container.
* No production promotion was performed and no production PR was merged.
* The acceptance gateway image was built as a thin layer over the **frozen**
  production gateway image, so production files could not drift underneath it.

---

## 11. Cleanup

`Cleanup: PASS`

| Check | Result |
| --- | --- |
| RUN_ID containers remaining | **0** |
| RUN_ID networks remaining | **0** |
| RUN_ID images remaining | **0** |
| RUN_ID compose projects remaining | **0** |
| RUN_ID appdata directories remaining | **0** |
| Production runtime changed by acceptance | **NO** |
| Production container abnormal restart caused by acceptance | **NO** |
| Pre-existing unrelated resources touched | **0** |
| Space reclaimed | **~0 GiB** (see note) |
| Remaining acceptance artifacts | **NONE** |

Deleted, all RUN_ID-scoped: 8 containers (`acc-20260917-kserve-0e7af82-*`
across both the CPU and GPU projects), the compose projects, the network
`acc-20260917-kserve-0e7af82-net`, the image tag
`research-ai-runtime-gateway:acc-20260917-kserve-0e7af82`, and
`/mnt/user/appdata/_acc-rai-ovms/20260917-kserve-0e7af82/`.

Pre-existing Research_AI_Runtime leftovers removed as instructed:
`ghcr.io/kettly1260/research-ai-runtime-gateway:cutover-8953f87a11159225b7814b17682a255ff427f00e`
and `ovms-api:latest`.

**`latest-gpu` handling.** The floating `openvino/model_server:latest-gpu` tag
was **removed**. Because the report's verdict is
`READY_FOR_PRODUCTION_PROMOTION_REVIEW` rather than an executed promotion, the
2026.3.1 image was kept under an explicit pinned tag
`openvino/model_server:2026.3.1` (`sha256:f52109ff2d03…`) instead of being
deleted — this satisfies "do not rely on the floating tag" without forcing a
1.11 GB re-pull at the moment someone acts on the review. Say the word and it can
be deleted instead.

**Never deleted, verified still present:**
`research-ai-runtime-gateway:local-5567-hotfix` (`sha256:4dbd6160e267…`) and
`openvino/model_server:2026.1-gpu` (`sha256:08ad73dfd651…`, the production
2026.1 digest).

No `docker system prune`, `docker image prune -a`, `docker volume prune` or
`docker network prune` was ever run — this host carries many other projects'
dangling images (36.81 GB of reclaimable volumes and 11.83 GB of build cache were
left strictly alone).

**Space reclaimed note.** `docker system df` reports image SIZE unchanged at
51.01 GB. The three deleted images were thin layers over retained bases
(`local-5567-hotfix`, `ovms-api-base`), so the honest figure is ≈0 GiB rather
than the ~4 GB their nominal sizes suggest. Reported as measured, not as
advertised.

**Manifest comparison.** A preflight manifest was captured before Gate 2 and a
post-clean manifest after cleanup. The production sections — container IDs,
image IDs, `StartedAt`, `RestartCount`, and the full production runtime SHA256
list including `config.json` — are **byte-identical** between the two (`diff`
reports no differences). A post-Gate-2 manifest was also captured and is
likewise byte-identical to preflight.

---

## 12. Verdict

All acceptance gates pass:

* Classic TFS/KServe compatibility — **PASS** on both versions
* GenAI v3 `/v3/embeddings` — **PASS** on both versions (not removed, not changed)
* Long-text liveness — **PASS**, 32/32 liveness probes 200
* Concurrency — **PASS**
* Reranker — **PASS**, identical scores across the migration
* Zero-resident — **PASS** on all four stacks
* CPU Gate — **PASS**
* UHD730 Gate — **PASS**, GPU uncontended so figures are `VALID`
* Cleanup — **PASS**

**READY_FOR_PRODUCTION_PROMOTION_REVIEW**

Production has not been switched. The promotion decision is a separate,
explicit step.

---

## 13. Open items

1. **`openvino/model_server:2026.3.1` is retained on disk.** Delete it if
   promotion is deferred.
2. **Cold model-load times are single samples on a shared host.** If load time
   matters to the promotion decision, re-measure on a quiet host with repeats;
   the current data supports "no established regression", not a ranking.
3. **The long-text path is CPU-bound, not GPU-bound** (§7.2). Worth knowing
   before sizing hardware for long-text throughput.
4. **`.gitattributes` is absent.** `git archive` on Windows rewrites line endings
   to CRLF, which broke a hash check during acceptance payload shipping. Not a
   correctness problem (Python tolerates CRLF) but it makes byte-level
   verification of shipped artefacts unreliable.
5. **The idle-unload TTL in the acceptance compose is 90 s**, not production's
   1200 s. Deliberate, to make the unload path observable inside the window;
   confined to the acceptance stack.
6. **`ensure_model_available()` still defaults to `preferred_device="GPU"`.** It
   is not used by the embedding paths and is documented as a
   backward-compatible shim, so it was left alone — but it carries the same
   latent trap D2 was.
