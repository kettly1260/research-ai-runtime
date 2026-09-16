# OVMS 2026.1 (TFS) vs OVMS 2026.3.1 (KServe v2) — Acceptance Report

Status: **GATE 1 PASSED / GATE 2 AND 3 NOT STARTED / HARNESS READY**

This report is filled in only from real measurements. Every cell that has not
been measured yet is marked `PENDING` and carries the exact command or gate that
must produce it. Nothing here is inferred from documentation.

Gate 2 and Gate 3 are **blocked on inputs the report cannot supply for itself**:
a pinned 2026.3.1 image digest, an isolated CPU-only environment, and explicit
authorisation to create the Unraid acceptance stack. The measurement tooling is
built, smoke-tested and unit-tested (§4.7), so those gates are a single command
each once the inputs arrive.

---

## 1. Gating model

The acceptance work is split into three gates, and they must be cleared in order.

| Gate | Scope | Environment | State |
| --- | --- | --- | --- |
| **Gate 1 — CPU / protocol / functional** | adapter correctness, wire format, error semantics, regression | isolated VM, CPU only, no GPU | **PASSED** (this document, §4) |
| **Gate 2 — VM integration** | real OVMS 2026.1 + 2026.3.1 on CPU, capability probe capture | isolated VM, CPU only | **NOT STARTED** — tooling ready (§4.7), blocked on image digest + environment |
| **Gate 3 — UHD730 GPU acceptance** | real GPU latency, memory, concurrency, long-text liveness | Unraid, isolated `acc-*` stack | **NOT STARTED — blocked on Gate 2** |

Two rules govern everything below:

1. **A VM PASS is not a GPU acceptance PASS.** CPU correctness says nothing about
   the UHD730 GPU plugin, FP16 handling, memory pressure or spill behaviour.
2. **Nothing in this task touches production.** Production `ovms-server`,
   production `ai-gateway` and the production runtime config directory are read
   only. Production stays on the validated 2026.1 digest and `OVMS_PROTOCOL=tfs`.

---

## 2. Artefacts under test

| Item | Value |
| --- | --- |
| Commit under test | `eb17ace` (adapter `a4b035a` + acceptance harness) |
| Base commit (validated production long-text code) | `5567dbb8190d8315877389a92c5157f23a8a7c2a` |
| Branch | `feat-ovms-kserve-v2-adapter` |
| Production gateway image | `research-ai-runtime-gateway:local-5567-hotfix` |
| Production OVMS digest | `openvino/model_server@sha256:22f92a1a5ad6784384e47296c43bde239887a623e75de45b0e1802659de416a6` |
| Production OVMS version | OVMS 2026.1.0.72cc06244 / OpenVINO 2026.1.0 / OpenVINO GenAI 2026.1 |
| Production OVMS image ID | `sha256:08ad73dfd651b3535c1069371aac082eff1ab96101219326c641e1dd4f81e13b` |
| Acceptance OVMS 2026.1 image | PENDING — must be pinned by digest, never `latest-gpu` |
| Acceptance OVMS 2026.3.1 image | PENDING — must be pinned by digest, never `latest-gpu` |
| Hardware (GPU gate only) | Unraid host, Intel UHD 730 iGPU |

`latest-gpu` must never be used for an authoritative result. It has already
drifted from 2026.1 to 2026.3.1, which is exactly the drift this task exists to
survive: a tag is not a production compatibility contract.

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
* No model-specific tensor name, dtype or output shape is hardcoded; everything
  is derived from the caller's payload and from the shapes the backend reports.
* `OVMS_PROTOCOL=tfs|kserve|auto`, default `tfs`.
* `auto` probes live capability, caches once per process, and fails closed (502)
  rather than silently falling back to a different protocol.

---

## 4. Gate 1 — CPU / protocol / functional (PASSED)

### 4.1 Command

```
pytest tests/ services/ai-gateway/tests/ services/research-media/tests/ -q
```

### 4.2 Result

| Metric | Baseline (before change) | After change |
| --- | --- | --- |
| Passed | 85 | **186** |
| Skipped | 2 | 2 |
| Failed | 0 | **0** |
| New protocol tests | — | 63 |
| New acceptance-harness tests | — | 38 |

No existing test expectation was modified, deleted or weakened. All 101 new
tests are additive.

### 4.3 New test coverage

| Area | Tests | Evidence |
| --- | --- | --- |
| Protocol selection (`tfs` / `kserve` / `auto` / invalid) | 4 | invalid value fails fast at import with a non-zero exit |
| TFS serialisation | 4 | request payload passed through by identity, response returned verbatim |
| KServe tensor serialisation | 8 | `inputs[]` carries name/shape/datatype/data; `position_ids` preserved for the reranker; `pixel_values` for vision; ragged rows rejected |
| Response normalisation parity | 11 | 2-D, 3-D and multi-output `outputs` all normalise to the same `predictions` shape TFS produces |
| Error semantics | 9 | 400/404/412/500 → 502, timeout → 504, connection error → 502, malformed JSON → 502, unrepresentable payload → 502 |
| Auto detection | 8 | tfs-only, kserve-only, MediaPipe-412 tie-break, MediaPipe-404 tie-break, fail-closed, single-probe caching, 8-thread thread safety |
| Diagnostics | 3 | configured vs effective vs unresolved-auto |
| Protocol-aware availability | 4 | TFS `model_version_status`, KServe `/ready`, metadata fallback |
| CPU contract over real HTTP | 8 | see §4.4 |
| Acceptance harness gate logic | 38 | see §4.7 |

### 4.4 CPU contract test (real HTTP, no GPU)

A real `ThreadingHTTPServer` stands in for OVMS so the wire format is exercised
end to end rather than asserted in-process:

* the request really lands on `/v1/models/{m}:predict` or `/v2/models/{m}/infer`;
* the bytes on the socket really carry `instances` or shaped `inputs`;
* both protocols return **identical vectors** for the same logical input;
* `auto` detects a KServe-only backend and a MediaPipe-backed 2026.3 backend;
* the KServe availability probe tracks model load state.

### 4.5 Long-text parity (hard gate, protocol level)

A 2500-token input is driven through the real pooled-IR path
(`prepare_qwen_chunked_texts` → per-window inference → token-weighted merge →
L2 normalisation) under **both** protocols:

| Assertion | Result |
| --- | --- |
| Exactly one vector returned per logical text | PASS |
| Vector is L2-normalised (`‖v‖ = 1`) | PASS |
| Vector identical between TFS and KServe (`rtol=1e-6`) | PASS |
| Text was actually chunked (more than one backend request) | PASS |
| No backend request exceeded `POOLED_CHUNK_TOKENS` | PASS |
| Chunk boundaries identical between protocols | PASS |

This proves the adapter does not truncate, reject or re-shape long inputs. It
does **not** prove GPU latency or liveness — that is Gate 3.

### 4.6 Gate 1 verdict

| Check | Verdict |
| --- | --- |
| Compatibility (adapter present, protocol selectable) | **PASS** |
| Correctness (serialisation + normalisation parity) | **PASS** |
| Long-text integrity (protocol level) | **PASS** |
| Error semantics | **PASS** |
| Regression (no existing test weakened) | **PASS** |
| Lint (`ruff check`) | **PASS** |
| `git diff --check` | **PASS** |

### 4.7 Acceptance harness (the tooling that produces Gate 2 / Gate 3)

The remaining ~40 report cells would otherwise be hand-transcribed from `curl`
output, which is exactly how a red run gets written up as green. Two scripts
replace that, and their gate logic is itself unit-tested so a wrong verdict
cannot pass CI.

| Script | Purpose |
| --- | --- |
| `scripts/ovms_protocol_capability_probe.py` | records the live API surface endpoint by endpoint (§5.1) and reports what `OVMS_PROTOCOL=auto` would decide, by reusing the gateway's own `probe_protocol()` |
| `scripts/ovms_acceptance_suite.py` | runs the model matrix, long-text ladder, liveness gate, reranker and zero-resident checks; emits JSON plus a markdown fragment for §6 |

Safety properties, deliberately built in:

* **GET-only by default.** `POST /v1/models/{m}:predict` with `{"instances": []}`
  is a *negative* probe — it can never produce a successful inference, so it
  cannot cold-load a model. It is the MediaPipe tie-breaker the `auto` ladder
  depends on, so it is on by default.
* **`POST /v3/embeddings` is a real inference** and can cold-load a model. It is
  off unless `--allow-inference-probes` is passed, and must never be pointed at
  production.
* **Nothing is reconfigured.** The harness is a client. It never writes a runtime
  config directory and never restarts OVMS. Zero-resident state is *observed*
  through the gateway's own `/v1/broker/metrics`.
* **Proxy environment variables are ignored by default.** This machine exports
  `HTTP_PROXY=http://127.0.0.1:<port>`, and `requests` honours it — so a loopback
  acceptance run was being routed through the proxy, surfacing as a
  `ReadTimeout` naming the *proxy's* port instead of the target's. That would
  have benchmarked the proxy. Both scripts now build a `requests.Session` with
  `trust_env = False`; `--use-env-proxy` opts back in. The finding is recorded
  here because it silently corrupts every latency number in §6 if reintroduced.
* The suite **exits non-zero** if any verdict is `FAIL`, so it can be a CI gate.

The harness's decision logic is covered by
`services/ai-gateway/tests/test_acceptance_harness.py` (38 tests, offline): an
all-green run must pass every dimension, an all-red run must fail every
dimension, an unmeasured section must report `N/A` and never `PASS`, a norm
outside tolerance must fail correctness, a half-finished concurrency round must
fail concurrency, and a `N/A` row for a model with no production deployment must
not gate compatibility.

---

## 5. Gate 2 — VM integration with real OVMS (NOT STARTED)

Gate 2 must run on a **CPU-only** isolated environment and must not touch
production. It exists to capture real server behaviour instead of guessing it.

### 5.1 OVMS 2026.3.1 API capability capture (task item 25)

Record the literal responses; do not rely on KServe documentation. Produced by:

```
python scripts/ovms_protocol_capability_probe.py \
    --ovms-base http://<acceptance-ovms>:28342 \
    --label ovms-2026.3.1 --model qwen-reranker \
    --out .workbuddy-ai/artifacts/capability-20263.json
```

| Endpoint | 2026.3.1 observed response | 2026.1 observed response |
| --- | --- | --- |
| `GET /v2/health/live` | PENDING (probe) | PENDING (probe) |
| `GET /v2/health/ready` | PENDING (probe) | PENDING (probe) |
| `GET /v2/models/{model}` | PENDING (probe) | PENDING (probe) |
| `GET /v2/models/{model}/ready` | PENDING (probe) | PENDING (probe) |
| `POST /v2/models/{model}/infer` | PENDING (probe) | PENDING (probe) |
| `GET /v1/config` | PENDING (probe) | PENDING (probe) |
| `POST /v1/models/{model}:predict` | expected: 412 `model field is missing in JSON body` (observed on production probe) | expected: 200 |
| `GET /v1/models/{model}` | PENDING (probe) | PENDING (probe) |
| `GET /v3/embeddings` | PENDING (probe) | PENDING (probe) |
| `POST /v3/embeddings` | PENDING (`--allow-inference-probes`) | PENDING (`--allow-inference-probes`) |

The `auto` detection ladder in `ovms_protocol/auto.py` is built on
`/v2/health/ready` and `/v1/config`, with a negative `:predict` probe as the
tie-break. **These three signals must be confirmed against the real 2026.3.1
image before `auto` is used anywhere outside development, acceptance and CI.**
The probe script reports the ladder's own conclusion, so this table and the
ladder can never disagree.

Two scope notes that the capability capture must settle:

* **`qwen3-embedding-0.6b-int8` is served over the GenAI v3 `/v3/embeddings`
  surface**, not the Classic Model REST API. The TFS/KServe adapter split does
  not cover it, so its survival on 2026.3.1 is a *separate* risk that the
  adapter work does not mitigate. If `/v3/embeddings` disappears in 2026.3.1,
  that model is a hard blocker regardless of how green Gate 2 and Gate 3 are.
* **DINO and the multimodal Classic IR are not deployed in production OVMS.**
  Official DINOv3 weights are blocked on Meta gated approval
  (`docs/OFFICIAL_DINOV3_ACCEPTANCE_STATUS.md`:
  `BLOCKED_BY_META_APPROVAL_FOR_OFFICIAL_WEIGHTS`,
  `UHD730_GPU_CANDIDATE_NOT_RECOMMENDED`) and jina-clip-v2 exists only as an
  export. There is no production path to regress, so per the task wording
  ("PASS required if the current production path exists") these rows are
  **N/A**, not PASS and not FAIL. They are excluded from the compatibility gate.

### 5.2 Real model matrix (CPU)

Produced by:

```
python scripts/ovms_acceptance_suite.py \
    --gateway-base http://<acceptance-gateway>:28011 \
    --ovms-base    http://<acceptance-ovms>:28341 \
    --protocol tfs --label ovms-2026.1 \
    --sections capability,matrix --out .workbuddy-ai/artifacts/acc-20261.json
```

…and repeated with `--protocol kserve`, `--label ovms-2026.3.1`, the 2026.3.1
ports and `OVMS_PROTOCOL=kserve` on the gateway.

| Model | 2026.1 TFS | 2026.3.1 KServe |
| --- | --- | --- |
| qwen3-embedding-0.6b-int4 | PENDING | PENDING |
| qwen3-embedding-0.6b-int8 | PENDING (via `/v3/embeddings`, outside the adapter) | PENDING (via `/v3/embeddings`, outside the adapter) |
| bge-m3-i8 | PENDING | PENDING |
| bge-m3 | PENDING | PENDING |
| arctic-embed-m-v2-int8 | PENDING | PENDING |
| qwen-reranker | PENDING | PENDING |
| DINO | **N/A** — no production OVMS deployment | **N/A** — no production OVMS deployment |
| multimodal Classic IR | **N/A** — export-only, not in the registry | **N/A** — export-only, not in the registry |

Per-model tensor signature must be read from the live server, not assumed:
`input_ids` / `attention_mask` / `position_ids` on the reranker, the real output
tensor name and shape, and the image tensor name for the vision models.

A model that fails is recorded as FAILED. Skipping a failing model and declaring
the matrix "overall passing" is not permitted.

---

## 6. Gate 3 — UHD730 GPU acceptance (NOT STARTED)

Gate 3 runs only after Gate 2 passes, using
`deploy/docker-compose.acceptance-20263.yml`. That stack is fully isolated: own
container names, own ports (28341/28342, 28011/28012), own runtime directories
under `/mnt/user/appdata/_acc-ovms-2026{1,3}/runtime`, and a dedicated
`acc_ai_network`. It never mounts `/mnt/user/appdata/ovms/runtime`.

### 6.1 Recorded 2026.1 production baseline (reference only)

These figures were measured on production OVMS 2026.1 before this task and are
reproduced from the task brief as the comparison baseline. They are **not**
re-measured by this report.

| Scenario | Result |
| --- | --- |
| Single long text, ~4052 tokens | HTTP 200, 9.355 s, 1024-d, ‖v‖ = 1.0 |
| Two concurrent long texts, ~4591 tokens each | req 1: 200 / 16.88 s / 1024-d; req 2: 200 / 21.11 s / 1024-d; wall 21.11 s |
| Short text immediately after | HTTP 200, 0.057 s, 1024-d |
| research-memory-gateway, same `EmbeddingClient`, 2 threads, ~4591-token × 2 | 2/2 success, 1024-d, HTTP 200, wall ≈ 21.26 s, `Bad file descriptor` = 0, `last_error = None` |

### 6.2 Required 2026.3.1 measurements

Fill every cell from the acceptance run. Qwen3 INT4 must cover all four lengths
and both concurrency levels. Produced by:

```
python scripts/ovms_acceptance_suite.py \
    --gateway-base http://127.0.0.1:28012 \
    --ovms-base    http://127.0.0.1:28342 \
    --protocol kserve --label ovms-2026.3.1 \
    --sections capability,matrix,long_text,reranker,zero_resident \
    --out    .workbuddy-ai/artifacts/acceptance-20263.json \
    --report .workbuddy-ai/artifacts/acceptance-20263.md
```

Run the identical command with `--protocol tfs`, `--label ovms-2026.1` and the
2026.1 ports (28011/28341) for the comparison column. Text length is calibrated
from the gateway's own `usage.prompt_tokens`, so the ladder measures tokens
rather than assuming a characters-per-token ratio.

| Metric | 256 tok | 512 tok | 1024 tok | ~4591 tok |
| --- | --- | --- | --- | --- |
| HTTP status (single) | PENDING | PENDING | PENDING | PENDING |
| Cold latency | PENDING | PENDING | PENDING | PENDING |
| Warm latency | PENDING | PENDING | PENDING | PENDING |
| Concurrency = 2 wall latency | PENDING | PENDING | PENDING | PENDING |
| Vector dimension | PENDING | PENDING | PENDING | PENDING |
| Vector norm | PENDING | PENDING | PENDING | PENDING |
| Backend latency | PENDING | PENDING | PENDING | PENDING |
| GPU memory peak | PENDING | PENDING | PENDING | PENDING |
| CPU memory peak | PENDING | PENDING | PENDING | PENDING |
| CPU spillover occurred | PENDING | PENDING | PENDING | PENDING |
| 5xx count | PENDING | PENDING | PENDING | PENDING |

The harness reports HTTP status, latency, dimension, norm and 5xx count directly.
GPU/CPU memory peak, spillover and backend latency are read from the OVMS
container's own metrics during the same run and transcribed into the remaining
rows; they are not synthesised.

### 6.3 Liveness gate (highest-priority acceptance item)

After **every** long-text round, immediately issue a short-text embedding.

| Check | Requirement | Result |
| --- | --- | --- |
| Short text after single long text | HTTP 200 | PENDING |
| Short text after 2×4591-token concurrency | HTTP 200 | PENDING |
| Short text after a long-text timeout | HTTP 200 | PENDING |

A long request that completes or times out and then permanently wedges
subsequent short requests is an **overall acceptance failure**, not a degraded
result.

### 6.4 Reranker correctness

| Check | Requirement | Result |
| --- | --- | --- |
| `what is biology?` vs "Biology is the study of living organisms." vs "The Eiffel Tower is in Paris." | biology score > Eiffel score | PENDING |
| Score range | relevance probability in 0..1, not raw logits | PENDING |
| research-memory-gateway threshold semantics | unchanged | PENDING |

### 6.5 Zero-resident lifecycle

Run separately for `OVMS_PROTOCOL=tfs` and `OVMS_PROTOCOL=kserve`:

| Check | Requirement | Result (TFS) | Result (KServe) |
| --- | --- | --- | --- |
| Empty runtime → request → alias load → inference → AVAILABLE | observed | PENDING | PENDING |
| Idle unload back to zero-resident | observed | PENDING | PENDING |
| `MAX_LOADED_MODELS=2` respected | observed | PENDING | PENDING |
| GPU preferred, CPU spillover on contention | observed | PENDING | PENDING |
| Alias naming `<model>__gpu` / `<model>__cpu` | observed | PENDING | PENDING |
| Broker is not bypassed under KServe | observed | PENDING | PENDING |

### 6.6 Failure counters (must all be zero)

| Counter | Requirement | Result |
| --- | --- | --- |
| Gateway tracebacks | 0 | PENDING |
| OVMS abnormal restarts | 0 | PENDING |
| `Bad file descriptor` (RMG) | 0 | PENDING |
| Unhandled 5xx | 0 | PENDING |

---

## 7. Verdict table

| Dimension | 2026.1 TFS | 2026.3.1 KServe |
| --- | --- | --- |
| Compatibility | **PASS** (Gate 1) | **PENDING** (Gate 2/3) |
| Correctness | **PASS** (Gate 1) | **PENDING** (Gate 2/3) |
| Long-text liveness | **PASS** (protocol level, Gate 1) | **PENDING** (Gate 3) |
| Concurrency | **PASS** (recorded baseline) | **PENDING** (Gate 3) |
| Reranker | **PASS** (recorded baseline) | **PENDING** (Gate 3) |
| Zero-resident | **PASS** (recorded baseline) | **PENDING** (Gate 3) |
| DINO | **N/A** — no production OVMS deployment | **N/A** — no production OVMS deployment |
| Multimodal | **N/A** — export-only, not in the registry | **N/A** — export-only, not in the registry |
| Failure counters | **PASS** (recorded baseline) | **PENDING** (Gate 3) |

The Gate 2/3 columns are filled by `evaluate_gates()` in
`scripts/ovms_acceptance_suite.py`, which is unit-tested offline. A dimension
whose section did not run reports `N/A`; it never reports `PASS`.

---

## 8. Production posture

Production remains on the validated OVMS 2026.1 digest with
`OVMS_PROTOCOL=tfs`. This task does **not** authorise any production change.

Even if every gate above turns green, the only permitted statement is:

> OVMS 2026.3.1 acceptance passed — ready for production promotion review.

Promotion itself waits for an explicit, separate authorisation. See
`docs/OVMS_2026_3_MIGRATION_AND_ROLLBACK_PLAN.md`.

---

## 9. Open items

| # | Item | Owner decision needed |
| --- | --- | --- |
| 1 | Provide the pinned OVMS 2026.3.1 image digest for the acceptance stack | yes |
| 2 | Provide/confirm the isolated VM used for Gate 2 | yes |
| 3 | Authorise Gate 3 (Unraid `acc-*` stack, UHD730) once Gate 2 passes | yes |
| 4 | Re-validate the `auto` detection ladder against the real 2026.3.1 server | yes |
| 5 | Decide the fate of `qwen3-embedding-0.6b-int8`, which is served over `/v3/embeddings` and is therefore **not** covered by the TFS/KServe adapter | yes |
| 6 | CI cannot run 2026.3 GPU; GPU acceptance stays an Unraid-run report | informational |
