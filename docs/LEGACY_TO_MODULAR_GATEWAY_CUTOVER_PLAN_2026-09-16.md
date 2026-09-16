# Legacy → Monorepo Modular Gateway Cutover Plan — 2026-09-16

```
[planning pass, earlier 2026-09-16]
LEGACY -> MODULAR GATEWAY CUTOVER READINESS: NOT READY
P0 blockers: 4   P1 decisions: 4   P2 build items: 3

[P0 closure pass, 2026-09-16 — implementation, no production change]
MODULAR GATEWAY PARITY  : PASS
GENAI_V3 PARITY         : PASS
MODEL REGISTRY PARITY   : PASS
ENV PARITY              : PASS
CUTOVER ALIAS PLAN      : PASS
PRODUCTION CUTOVER READY: NO    (all P0 closed; awaiting explicit cutover go-ahead)
```

Status: **planning only.** No production change was made while producing this document.
This plan is the separate follow-up explicitly deferred by
`docs/QWEN3_RERANKER_PRODUCTION_HOTFIX_2026-09-16.md` §9 and gated by
`docs/PRODUCTION_RUNTIME_STATUS_2026-09-15.md` "Deployment gate".

---

## 0.5 P0 closure status — verified 2026-09-16 in an isolated stack

All four P0 blockers below were closed **in the repository only**. Nothing in production was
modified: production `config.json` remained `{"model_config_list": []}`, `ai-gateway`
`RestartCount=0` with an unchanged `StartedAt`, and `ovms-server` was neither restarted nor
recreated.

Verification environment (isolated, no iGPU, disjoint ports):

| component | container | port | image |
|---|---|---|---|
| OVMS | `acc-ovms-cutover` | 29191 | `openvino/model_server:latest-gpu`, `--file_system_poll_wait_seconds 1`, no `/dev/dri` |
| gateway | `acc-gateway-cutover` | 29192 | `ovms-api-modular:acc` |

Evidence summary:

- **P0-1 genai_v3** — ported (`config.py`, `ovms_client.py`, `inference/embeddings.py`,
  `inference/__init__.py`, `main.py`). `services/ai-gateway/tests/test_genai_embeddings.py` adds
  23 tests including **two live OVMS tests**, both passing against `acc-ovms-cutover`:
  the ported client reproduces the legacy `/v3/embeddings` payload byte-for-byte
  (`np.array_equal` on every vector), and the full `/v1/embeddings` route works end-to-end.
- **P0-2 dependencies** — all four production tokenizer directories ship `tokenizer.json`, so
  `protobuf` / `sentencepiece` are **not** required; `pillow` was the only genuine gap.
  `services/ai-gateway/Dockerfile.cutover` adds it on the proven base and the image builds.
- **P0-3 registry** — `config/models.yaml` is now an exact mirror of the production registry
  (6 ids, correct `ovms_model` / `embedding_backend`), with Jina/DINO removed and no per-model
  `idle_ttl`. Locked by `services/ai-gateway/tests/test_registry_parity.py`.
- **P0-4 deployment path** — `deploy/docker-compose.production.yml` declares **only**
  `ai-gateway` (no `ovms` service at all), so `ovms-server` can never be recreated by a gateway
  deploy. Locked by `tests/test_deployment_parity.py`.

Full suite: **99 passed, 0 skipped** with live OVMS tests enabled (97 passed, 2 skipped
hermetically). `ruff` clean on `services/ai-gateway/`, `tests/test_deployment_parity.py`,
`scripts/cutover_normalize_runtime_config.py`, `config/`.

---

## 0. Scope and standing constraints

**In scope:** replacing the legacy single-file `ai-gateway` (bind-mounted `/app/main.py`)
with the modular `services/ai-gateway/research_ai_gateway/` package.

**Out of scope:** `research-media` (not deployed in production — see §1), the DINO backend
decision (closed), and any OVMS model/IR work.

**Constraints that remain in force for the whole cutover:**

| constraint | reason |
|---|---|
| `ovms-server` must not be modified, reloaded, or restarted | standing directive |
| no `POST` to production OVMS `/v1/config/reload` | it reloads unrelated models and leaves `END` tombstones |
| only the `ai-gateway` container may be restarted | standing directive |
| no `commit`, `push`, `git reset --hard`, `git clean` | standing directive |
| must be reversible by restoring one artifact + restarting `ai-gateway` | same pattern as the reranker hotfix |

---

## 1. Verified current production state (read-only, 2026-09-16)

| item | observed |
|---|---|
| `ai-gateway` image | `ovms-api` — built as `FROM ovms-api-base:20260915` + `COPY main.py` |
| `ai-gateway` command | `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1` |
| `ai-gateway` restart policy | `unless-stopped` |
| `ai-gateway` network | `ai_network` (172.21.0.0/16), aliases `ai-gateway`, `api` |
| `ai-gateway` mounts | `/mnt/user/appdata/ovms/api → /app (rw)`, `…/tokenizers → /tokenizers (ro)`, `…/models → /models (ro)`, `…/runtime → /ovms-config (rw)` |
| deployed `main.py` | SHA256 `525513dafa56da8d92f7b5036156d16296ca70a745e291549f82358d723ca557` (post-hotfix) |
| `ovms-server` command | `--config_path /ovms-config/config.json --rest_port 8001` (**no** explicit poll flag) |
| `ovms-server` network | `ai_network`, aliases `ovms-server`, `ovms` |
| OVMS model state | `bge-m3-i8` AVAILABLE, `qwen-reranker` AVAILABLE, `_acceptance-tp-dinov3onnx__cpu` END, `_acceptance-tp-dinov3onnx__gpu` END |
| runtime `config.json` | 481 B, md5 `480a983495c3f98ea81454642e54a38c` — 2 models, **raw names** (no `__gpu`/`__cpu` suffix) |
| `research-media` container | **absent** from `docker ps` — not deployed |

Production registry of record is `/mnt/user/appdata/ovms/runtime/model_registry.json` (6 models):

| model id | `ovms_model` | `embedding_backend` |
|---|---|---|
| `qwen-reranker` | `qwen-reranker` | — (type `rerank`) |
| `bge-m3-i8` | `bge-m3-i8` | *(default)* |
| `bge-m3` | `bge-m3-i8` | *(default)* |
| `qwen3-embedding-0.6b-int8` | `qwen3-embedding-0.6b` | **`genai_v3`** |
| `qwen3-embedding-0.6b-int4` | `qwen3-embedding-0.6b-int4-pooled` | `pooled_ir` |
| `arctic-embed-m-v2-int8` | `arctic-embed-m-v2-int8` | `sentence_transformer` |

`/mnt/user/appdata/ovms/runtime/model_catalog.json` holds 6 physical entries; the modular
`catalog_model_config()` correctly synthesises `<name>__gpu` / `<name>__cpu` aliases from the
base entries, so **the catalog does not need new entries**.

---

## 2. Parity matrix

### 2.1 HTTP surface

| endpoint | legacy | modular | verdict |
|---|---|---|---|
| `GET /v1/models` | yes | yes | parity |
| `GET /models` | yes | yes | parity |
| `GET /v1/models/{id}` | yes | yes | parity |
| `POST /v1/rerank` | yes (now correct) | yes | parity — both use the official protocol |
| `POST /v1/embeddings` | yes | yes | **partial — see P0-1** |
| `GET /v1/embedding-batch-stats` | yes | yes | **semantic diff — see P1-2** |
| `POST /v1/chat/completions` | yes (demo stub) | **absent** | **gap — see P1-1** |
| `GET /health` | absent | yes | additive |
| `GET /v1/broker/metrics` | absent | yes | additive |

### 2.2 Settings — production actual vs modular defaults

| variable | production actual | modular compose | modular code default | impact if unchanged |
|---|---|---|---|---|
| `MODEL_IDLE_UNLOAD_SECONDS` | **1200** | **600** | 1200 | idle eviction twice as aggressive |
| `RERANK_MAX_DOCS` | **64** | unset | **8** | rerank silently truncates 64 → 8 docs |
| `RERANK_MAX_LENGTH` | **128** | unset | **256** | rerank token budget doubles |
| `RERANK_BATCH_SIZE` | **1** | unset | **2** | rerank batch 1 → 2 |
| `BGE_BATCH_SIZE` | **8** | unset | **4** | bge throughput halves |
| `OVMS_TIMEOUT` | 120 | 120 | 60 | ok |
| `MAX_LOADED_MODELS` | 2 | 2 | 2 | ok |
| `MODEL_IDLE_SWEEP_SECONDS` | 60 | 60 | 60 | ok |
| `RERANK_STRICT_RECALL` | 1 | unset | 1 | ok |
| `RERANK_MAX_CHARS` | 1000 | unset | 1000 | ok |
| `BGE_CHUNK_TOKENS` / `BGE_CHUNK_OVERLAP` | 1024 / 128 | unset | 1024 / 128 | ok |
| `HF_ENDPOINT` | `https://hf-mirror.com` | **unset** | unset | tokenizer fallback download unreliable |

### 2.3 Lifecycle semantics

| aspect | legacy | modular |
|---|---|---|
| OVMS alias written to config | raw name (`bge-m3-i8`) | `<name>__gpu` / `<name>__cpu` |
| config update mechanism | write file **+ `POST /v1/config/reload`** | write file, rely on OVMS polling (`OVMS_CONFIG_UPDATE_MODE=poll`) |
| residency accounting | keyed by raw name | keyed by alias |
| idle eviction | `MODEL_IDLE_UNLOAD_SECONDS` + per-model `idle_ttl` | same, plus `PINNED_MODELS` / `pinned` |
| capacity control | `MAX_LOADED_MODELS` | `MAX_LOADED_MODELS` |
| CPU spillover / coexistence | absent | present (semaphore, thread budget, temp replicas) |
| rerank scoring | official `P(yes)` (post-hotfix) | official `P(yes)` |

**Polling is confirmed viable.** Production OVMS is started without
`--file_system_poll_wait_seconds`; the OVMS default for that flag is **1 second**
("Default value is 1. Zero value disables changes monitoring"), so `config.json` edits are
picked up automatically. The modular `poll` default is therefore correct for this host, and
**not** calling `/v1/config/reload` is an improvement — but it means a config change now costs
polling latency plus a cold load (measured on this host: ≈46.7 s per model) before the first
request completes.

---

## 3. P0 blockers — must be resolved before any cutover

### P0-1. `genai_v3` embedding backend is not implemented in the modular gateway

> **RESOLVED 2026-09-16 — option 1 (port) taken.** The graph backend is implemented and verified
> against real OVMS; `qwen3-embedding-0.6b-int8` is **not** retired.

- Production model `qwen3-embedding-0.6b-int8` declares `embedding_backend: genai_v3` and maps to
  the graph-based OVMS model `qwen3-embedding-0.6b` (`graph_path: graph.pbtxt`).
- `grep -rn "genai" services/ai-gateway/` returns **zero** matches. `research_ai_gateway/main.py`
  branches only on `pooled_ir` / `sentence_transformer` / default, so this model id would fall
  through to `call_embedding()` → `run_bge_embedding()` and post a bge-shaped payload to a
  genai graph model.
- `inference/__init__.py` exports no genai entry point, and there is no `AdaptiveGenAIBatcher`
  equivalent.

**Resolution options (pick one):**

1. **Port** `call_genai_embedding` + the genai adaptive batcher from legacy (`legacy/main.py`
   `call_genai_embedding` / `AdaptiveGenAIBatcher`), add the branch to `main.py`, and extend
   `test_gateway.py` with a genai-shaped case. ← **chosen and implemented**
2. **Retire** the model id: remove `qwen3-embedding-0.6b-int8` from the production registry and
   confirm no client calls it. Requires a client survey, and is a *behaviour* change, not a
   pure cutover.

### P0-2. The production image cannot run the modular package (`pillow` missing)

> **RESOLVED 2026-09-16** — `services/ai-gateway/Dockerfile.cutover` adds `pillow>=10.0` on the
> proven `ovms-api-base:20260915` base; the image builds and starts. `protobuf`/`sentencepiece`
> were audited and are **not** needed (see below).

- `research_ai_gateway/inference/__init__.py` imports `multimodal`, which does
  `from PIL import Image` **at module import time**.
- Installed in the running container: `fastapi 0.135.3`, `uvicorn 0.44.0`, `transformers 5.5.4`,
  `tokenizers 0.22.2`, `numpy 2.4.4`, `requests 2.33.1`, `PyYAML 6.0.3`, `httpx 0.28.1`,
  `pydantic 2.13.0`.
- **Not installed: `pillow`, `protobuf`, `sentencepiece`.**
- Dropping `research_ai_gateway/` into the existing `/app` bind mount would therefore fail at
  startup with `ModuleNotFoundError: No module named 'PIL'`.

**Resolution:** the modular gateway needs its own image (see §5). Note the tokenizer directories
contain `tokenizer.json` (fast tokenizers), which is why the legacy gateway works today without
`protobuf` / `sentencepiece`; the modular package declares both, so a fresh build satisfies them.

### P0-3. Model registry precedence would change the served model set

> **RESOLVED 2026-09-16 — option (a) taken.** `config/models.yaml` is now an exact mirror of the
> production registry; `config/models.example.yaml` is byte-identical to it. Jina/DINO are gone,
> so YAML precedence can no longer expose undeployed models. Locked by
> `services/ai-gateway/tests/test_registry_parity.py`.

- `registry.HotModelRegistry` tries `MODELS_CONFIG_PATH` (YAML) **first** and only then falls back
  to `MODEL_REGISTRY_PATH` (JSON).
- `deploy/docker-compose.yml` mounts `../config:/config:ro` and sets
  `MODELS_CONFIG_PATH=/config/models.yaml`, so the **repo YAML wins** over the production JSON.
- The repo YAML's model set differs from production:

| | production registry | repo `config/models.yaml` |
|---|---|---|
| `qwen-reranker` | yes | yes |
| `bge-m3-i8`, `bge-m3` | yes | yes |
| `qwen3-embedding-0.6b-int8` | yes (`genai_v3`) | **missing** |
| `qwen3-embedding-0.6b-int4` | yes (`pooled_ir`) | yes (same `ovms_model`) |
| `arctic-embed-m-v2-int8` | yes (`sentence_transformer`) | **missing** |
| `jina-clip-v2` | no | **added** (not resident in OVMS) |
| `dinov3` | no | **added** (not resident in OVMS) |

Deploying as-is would change `GET /v1/models`, 404 two live model ids, and advertise two models
that have no deployed OVMS counterpart.

**Resolution options:** (a) author `config/models.yaml` as an exact mirror of the production
registry; (b) do not mount `/config` and leave `MODELS_CONFIG_PATH` pointing at a non-existent
path so the JSON stays authoritative; (c) change the precedence so JSON wins when it exists.

### P0-4. The compose file must not be applied wholesale

> **RESOLVED 2026-09-16.** `deploy/docker-compose.production.yml` now declares **only**
> `ai-gateway` — there is no `ovms` service in it at all, so `ovms-server` cannot be recreated by
> a gateway deploy. `deploy/docker-compose.yml` (full stack) carries a prominent
> "DO NOT RUN THIS FILE ON THE PRODUCTION UNRAID HOST" banner. Locked by
> `tests/test_deployment_parity.py`.

- `deploy/docker-compose.yml` declares the `ovms` service with `container_name: ovms-server`,
  `privileged: true`, `/dev/dri`, its own env block, and a command that includes
  `--file_system_poll_wait_seconds 1`.
- The **running** `ovms-server` was started with a different command
  (`--config_path /ovms-config/config.json --rest_port 8001`) and there is no evidence it was
  created from this compose file.
- `docker compose -f deploy/docker-compose.yml up -d` — as the README "Quick Start" instructs —
  would **recreate `ovms-server`**, violating the standing constraint.
- The compose file also declares `research-media`, which is not deployed.

**Resolution:** the cutover must target only the gateway, e.g.
`docker compose -f deploy/docker-compose.yml up -d --no-deps ai-gateway`, or a plain
`docker run`. The README quick-start command must be corrected, and the `ovms` service should be
marked `external` (or split into a separate compose file) so it can never be recreated by a
gateway deploy.

---

## 4. P1 decisions

### P1-1. `POST /v1/chat/completions`

Present in legacy, absent in modular. The legacy implementation is a **demo stub**: it reranks
three hard-coded Chinese document strings and returns the top one as an assistant message. The
string literals in the deployed file are mojibake-corrupted, which indicates the endpoint was
never exercised by a real client.

**Decision required:** confirm no client calls it, then declare it removed (preferred), or port
the stub as-is.

> **INVESTIGATED 2026-09-16 — no consumer found. Recommend: do not port; mark for deletion.**
>
> Read-only evidence:
>
> | probe | result |
> |---|---|
> | production `ai-gateway` access log, ~46 h (container created `2026-09-14T18:36:04Z`) | **0** `POST /v1/chat/completions` calls |
> | repo-wide grep | references exist only in `docs/` and memory notes — no executable caller |
> | host + appdata scan | the string appears only in the legacy `main.py` and its own backups |
> | every running container's env, grepped for `28001` / `ai-gateway` | no internal consumer |
>
> Conclusion: the endpoint is **not** carried into the modular gateway. It is not a modular
> feature and no compatibility shim is required. Deleting it from the legacy surface is a
> separate, later cleanup and is **not** part of this cutover.

### P1-2. `GET /v1/embedding-batch-stats`

> **RESOLVED 2026-09-16** — the endpoint now follows the configured backend, matching legacy.
> Verified by `test_embedding_batch_stats_follows_configured_backend`.

Legacy returns the stats of whichever batcher backs the registry's
`qwen3-embedding-0.6b-int4` entry (pooled vs genai); modular always returns
`adaptive_pooled_batcher.stats()`. If P0-1 is resolved by porting the genai path, this endpoint
should mirror the legacy branch to stay useful.

### P1-3. Alias migration and the duplicate-residency trap

The modular broker writes `<name>__gpu` / `<name>__cpu` entries, while the current runtime config
contains raw names. Because `_set_model_enabled()` only **adds** when the alias is absent, the
first modular request would leave `bge-m3-i8` **and** `bge-m3-i8__gpu` in the config — two OVMS
models over the same `base_path`, i.e. two resident copies on the same iGPU and an incorrect
`MAX_LOADED_MODELS` count.

**Mitigation (mandatory, part of the runbook):** before starting the modular container, normalise
`/ovms-config/config.json` to the state the broker expects — either pre-write the alias entries,
or write `{"model_config_list": []}` and let the broker add exactly what it needs on demand.
Back the file up first. `OVMS_CONFIG_UPDATE_MODE` stays `poll`; no reload API call is made.

> **VERIFIED 2026-09-16 — trap reproduced, then repaired, in the isolated stack.**
> Tooling: `scripts/cutover_normalize_runtime_config.py` (modes `plan` / `check` / `apply`,
> sha256 `b86fd601…`). `apply` refuses to write unless `--apply` is passed; it backs the file up,
> replaces it atomically (tmp + `fsync` + `os.replace`, the same contract the gateway uses), and
> then polls until OVMS reports zero resident models. **No reload API call is ever made.**
>
> Reproduced failure mode (S0 → one `POST /v1/embeddings {"model":"bge-m3",…}`):
>
> ```
> S0   : bge-m3-i8, qwen-reranker                      (raw legacy names)
> POST : http 200 in 6.44 s
> S1   : bge-m3-i8  base_path=/models/bge-m3-i8
>        bge-m3-i8__cpu base_path=/models/bge-m3-i8    <-- duplicate base_path
> OVMS : bge-m3-i8 AVAILABLE + bge-m3-i8__cpu AVAILABLE  (same model loaded twice)
> broker metrics: loaded_models = {bge-m3-i8__cpu} only  <-- raw entry untracked, never evictable
> ```
>
> `check` flags all four invariant classes: duplicate `base_path`, raw-name residency,
> resident-but-undeclared, and multi-residency per `base_path`.
>
> Safe transition, verified end to end:
>
> | step | observed result |
> |---|---|
> | `apply` (dry run, no `--apply`) | file untouched, no backup, no `.tmp` left behind |
> | `apply --apply` | backup written; config → `{"model_config_list": []}`; **zero-resident reached via filesystem polling only** |
> | `POST /v1/embeddings {"model":"bge-m3"}` | http 200; config now holds **only** `bge-m3-i8__cpu` |
> | `POST /v1/rerank` (qwen-reranker) | http 200; config now holds `bge-m3-i8__cpu` + `qwen-reranker__cpu` |
> | `check` | `alias transition invariants: OK`, exit 0 |
> | `POST /v1/embeddings {"model":"qwen3-embedding-0.6b-int8"}` | http 200; config holds **only** `qwen3-embedding-0.6b__cpu` (with `graph_path` + `target_device` correctly injected) |
>
> Two runbook consequences:
>
> 1. The broker **self-heals**: `lease()` calls `_sync_model_loaded_state()`, which re-queries OVMS
>    before deciding, so a stale in-memory map does not strand a request after an external
>    normalisation. The hazard is the *config file*, not the broker's memory.
> 2. Rollback (`cp -p <backup> config.json`) faithfully restores the raw-name state and `check`
>    re-reports the violations. Rolling back therefore means **returning to the legacy gateway** —
>    the modular container must not be left running against a rolled-back config.
> 3. After any config write, allow the ~1 s polling interval to settle before running `check`;
>    during the transition `check` correctly reports "resident but not declared" for models OVMS
>    has not unloaded yet.
>
> Production is currently `{"model_config_list": []}` (legitimate zero-resident idle state), so the
> historical `_acceptance-tp-dinov3onnx__cpu` / `__gpu` `END` tombstones no longer exist. A real
> cutover still runs `plan` → `check` first, because the legacy gateway repopulates raw names the
> moment it serves a request.

### P1-4. Tokenizer source path

Production's `RERANK_TOKENIZER_PATH` / `BGE_TOKENIZER_PATH` values were not readable (redacted in
transit). Both candidate locations exist on the host:

- `/mnt/user/appdata/ovms/tokenizers/bge-m3` (`tokenizer.json`, `tokenizer_config.json`);
- `/mnt/user/appdata/ovms/tokenizers/qwen-reranker` (full fast-tokenizer set);
- `/mnt/user/appdata/ovms/models/qwen-reranker/1` also exists.

The modular defaults (`/tokenizers/bge-m3`, `/tokenizers/qwen-reranker`) resolve correctly, but
the **exact production values must be captured at cutover time** and pinned explicitly rather than
assumed.

---

## 5. P2 build and deploy mechanics

**P2-1. Image.** The legacy image is a two-layer recipe: a fat, pre-built
`ovms-api-base:20260915` plus `COPY main.py`. The modular `services/ai-gateway/Dockerfile` is
`FROM python:3.11-slim` + `pip install /packages/contracts` + `pip install /app`, which requires
PyPI reachability **during the build on the unraid host**.

Recommended low-risk approach — keep the proven base and add only the missing pieces:

```dockerfile
FROM ovms-api-base:20260915
WORKDIR /app
RUN pip install --no-cache-dir pyyaml pillow
COPY research_ai_gateway /app/research_ai_gateway
CMD ["uvicorn", "research_ai_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Notes:
- `packages/contracts` is declared as a dependency but is **never imported** by
  `research_ai_gateway` (zero runtime references), so it can be omitted from the thin image.
- `pyyaml` is already present in the base (6.0.3); installing it is idempotent. `pillow` is the
  only genuinely new runtime requirement.
- Build context: the repo lives on the Windows `G:` drive, not on unraid. Either transfer
  `services/ai-gateway/research_ai_gateway/` to the host and build there, or build on Windows and
  `docker save | gzip` → transfer → `docker load`.

**P2-2. Container identity.** The modular compose keeps `container_name: ai-gateway` and
`28001:8000`, so the cutover is a container replacement with an unchanged client-facing endpoint.

**P2-3. `research-media`.** Not deployed; keep it out of the cutover.

**P2-4. Network.** Both containers are on the external `ai_network`; `ovms-server` exposes both the
`ovms-server` and `ovms` aliases, so the modular default `http://ovms:8001` and the compose value
`http://ovms-server:8001` both resolve.

**P2-5. Repository readiness.** The working tree carries a large uncommitted refactor
(44 files changed, +938 / −3395) and the entire modular gateway package
(`services/ai-gateway/research_ai_gateway/`) is **untracked**. CI (`.github/workflows/docker.yml`)
would build and push the modular image to GHCR on `main`, but nothing has been pushed, so no
GHCR image is available. A cutover should not proceed from an untracked, uncommitted source tree —
resolving this requires the user to lift the no-commit constraint.

---

## 6. Cutover runbook

### Phase 0 — Freeze and capture (read-only)

```bash
docker inspect ai-gateway > /mnt/user/appdata/ovms/api/ai-gateway.legacy-inspect.json
docker inspect ai-gateway --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E 'TOKENIZER_PATH|REGISTRY_PATH|CONFIG_PATH|CATALOG|PINNED'
sha256sum /mnt/user/appdata/ovms/api/main.py
cp -p /mnt/user/appdata/ovms/runtime/config.json \
      /mnt/user/appdata/ovms/runtime/config.json.bak-cutover-$(date +%Y%m%d-%H%M%S)
docker inspect ovms-server --format '{{.RestartCount}} {{.State.StartedAt}}'   # record baseline
```

### Phase 1 — Close P0-1 … P0-4 in the repository

Port or retire `genai_v3`; align `config/models.yaml` with the production registry (or drop the
`/config` mount); pin the six drifted settings into the deploy environment; add
`HF_ENDPOINT=https://hf-mirror.com`; make the deploy path `--no-deps` and correct the README.

### Phase 2 — Build and stage the image

Build the thin image from §5 on the host, tag it (e.g. `ovms-api-modular:20260916`), and confirm
it starts and serves `/health` **without** being wired to production traffic yet.

### Phase 3 — Bounded pre-cutover validation

The modular gateway writes to the **shared** `/ovms-config/config.json`, so a full dry run would
disturb production residency. Limit this phase to:

- read-only checks against the modular container: `GET /health`, `GET /v1/models`,
  `GET /v1/models/{id}` — compare the model id list byte-for-byte with the legacy response;
- one controlled **zero-resident cold-load test** with a client timeout **above** the ≈46.7 s
  per-model cold-load budget (the earlier 12 s probe was too short and produced a false negative),
  reporting load time separately from steady-state inference latency.

### Phase 4 — Cutover

```bash
# 1. normalise the runtime config for the alias scheme (see P1-3)
# 2. start the modular container on the same name / port / mounts / network
# 3. do NOT touch ovms-server
```

### Phase 5 — Post-cutover verification

| check | expectation |
|---|---|
| `GET /v1/models` | identical model id set to the pre-cutover baseline |
| `POST /v1/embeddings` `bge-m3` | HTTP 200, 1024-d vector |
| `POST /v1/embeddings` `qwen3-embedding-0.6b-int8` | HTTP 200 (**fails today — P0-1**) |
| `POST /v1/embeddings` `arctic-embed-m-v2-int8` | HTTP 200 |
| `POST /v1/rerank` (4 probe pairs) | Biology 0.99336 > Eiffel 3.0e-05; France/chemistry/ML pairs correct; all scores in `[0,1]` |
| `GET /health`, `GET /v1/broker/metrics` | 200 |
| ai-gateway log | no traceback |
| `ovms-server` RestartCount / StartedAt | **unchanged** |
| OVMS load/unload events | only those triggered by the cold-load test; no reload API call |

### Phase 6 — Rollback

```bash
# stop the modular container, restore the legacy bind-mounted file, restart on the legacy image
sha256sum /mnt/user/appdata/ovms/api/main.py   # expect 525513dafa56da8d92f7b5036156d16296ca70a745e291549f82358d723ca557
docker restart ai-gateway
```

Rollback touches only `ai-gateway`. The reranker rollback artifact
`main.py.bak-hotfix-qwen3rerank-20260916-004146` remains valid and is one step further back.

---

## 7. Open questions

1. **`genai_v3`** — port the backend, or retire `qwen3-embedding-0.6b-int8`?
2. **`/v1/chat/completions`** — confirm no client depends on it before removing it.
3. **Registry source of truth** — production JSON, or a mirrored YAML under `config/`?
4. **Repository state** — the modular package is untracked and the tree carries a large
   uncommitted refactor; committing is currently forbidden. Which constraint should be lifted
   first, and on what branch?
5. **External rerank clients** — the hotfix changed `score` from arbitrary logits (≈14.8–16.2) to
   a probability in `[0,1]`. Any client applying an absolute threshold (`score > 10`) will now
   never pass. This is independent of the cutover but should be surveyed in the same window.

> **SURVEYED 2026-09-16 — no in-repo or in-cluster consumer applies an absolute rerank threshold.
> Value-safe.**
>
> | scope | finding |
> |---|---|
> | repo, executable code | the only rerank-score hits are documentation and historical acceptance JSON. `packages/contracts/src/contracts/models.py` declares `relevance_score: Optional[float]` but **nothing ever reads or compares it** — a pure passthrough field. |
> | repo, threshold patterns | a sweep for `threshold` / `min_score` / `cutoff` / bare `14.` comparisons across `services/ai-gateway/`, `scripts/`, `tests/` returns one hit: a docstring about **batching length buckets**, unrelated to rerank scores. |
> | historical artifacts | `official_models_test/*.json` record legacy scores of `13.4921875` / `12.5234375`; these are immutable evidence files that already carry a corrigendum, not live consumers. |
> | **external client (real one)** | `research-memory-gateway` (ghcr, ports 18787/18788) is the actual rerank caller. `data/web_config.yaml` sets `rerank.enabled: true`, `base_url: http://192.168.22.102:28001/v1`, `model: qwen-reranker`, `endpoint_path: /rerank`. Its `RerankClient` (`/app/src/research_memory_gateway/retrieval.py:252`) and `_extract_score_item` read `relevance_score`/`score`/`relevance`, cast to float, and **never apply an absolute threshold** — they use ordering only, surfacing the value as `match_reason=f"…+rerank:score={score:.4f}"`. Its `score <= 0` / `score > 0` comparisons belong to the vector-cosine and claim-match paths, not rerank. |
>
> Live confirmation of the new semantics through the modular stack:
> `{"query":"what is biology?","documents":["biology is the study of life","the stock market fell today"]}`
> → `0.9932884707176769` vs `4.3941713681527055e-05` — correct ordering, both in `[0,1]`.

### Decisions taken 2026-09-16 (owner)

| # | question | decision |
|---|---|---|
| 1 | `genai_v3` port or retire? | **Port.** `qwen3-embedding-0.6b-int8` is not retired; production model/API parity is preserved for this cutover. |
| 2 | `/v1/chat/completions`? | **Not ported** as a modular feature. No consumer found, so no shim is needed; deletion from the legacy surface is a later, separate cleanup. |
| 3 | Registry source of truth? | **`config/models.yaml`** is the single authoritative logical registry. Before cutover it must be an exact mirror of the production model set (qwen3 embedding / arctic / reranker / BGE). YAML + `model_registry.json` dual authority is explicitly **not** maintained long term. |
| 4 | Repository state? | A dedicated branch `cutover/modular-ai-gateway`. **Local commits allowed, push forbidden.** The proposed file list must be published before the branch is created. `git reset --hard` and `git clean` are forbidden. |
| 5 | External rerank clients? | Surveyed — value-safe (see above). |

### Proposed commit file list — `cutover/modular-ai-gateway`

**Not yet created.** Published here for review; the branch is created only after sign-off.

*Included — the P0 parity work:*

```
config/models.yaml                                        (new)  P0-2 single-authority registry
config/models.example.yaml                                (mod)  P0-2 kept byte-identical
deploy/docker-compose.production.yml                      (new)  P0-4/P0-6 ai-gateway-only deploy path
deploy/docker-compose.yml                                 (mod)  P0-4 do-not-run-on-prod banner + baseline env
deploy/env.example                                        (mod)  P0-4 documented production baseline
services/ai-gateway/Dockerfile.cutover                    (new)  P0-3 pillow on the proven base
services/ai-gateway/pyproject.toml                        (mod)  deps
services/ai-gateway/research_ai_gateway/**                (new)  the modular package (genai_v3 port)
services/ai-gateway/app/**                                (del)  superseded by the package rename
services/ai-gateway/tests/test_genai_embeddings.py        (new)  P0-1 unit + live OVMS tests
services/ai-gateway/tests/test_registry_parity.py         (new)  P0-2 registry lock
services/ai-gateway/tests/test_gateway.py                 (mod)  DINO tests inject a local registry
services/ai-gateway/tests/conftest.py                     (mod)
services/ai-gateway/README.md                             (new)
tests/test_deployment_parity.py                           (new)  P0-4/P0-6 deployment lock
tests/integration/test_pipeline.py                        (mod)
scripts/cutover_normalize_runtime_config.py               (new)  P0-5 alias normalisation tool
docs/LEGACY_TO_MODULAR_GATEWAY_CUTOVER_PLAN_2026-09-16.md (new)  this document
```

*Excluded — scratch / cache / ignored (must never be committed):*

```
.tmp_stage/  .tmp_wheels/  .tmp_official_audit/  .workbuddy-ai/
acceptance_results.json  __pycache__/  .ruff_cache/  .pytest_cache/
legacy/            (already ignored via .gitignore)
```

*Excluded — unrelated in-flight work that happens to share the working tree. These belong on
their own branch/commit and must not ride along:*

```
official_models_test/**            (DINOv3 acceptance artifacts)
docs/OFFICIAL_DINOV3_*             docs/ROADMAP_MULTIMODAL_EMBEDDING_COMPAT.md
exported_models/model_catalog.json scripts/convert_models.py  scripts/smoke_test_models.py
tests/test_model_conversion.py     tests/acceptance_test_runner.py  tests/test_dynamic_lifecycle.py
config/providers.yaml              config/providers.example.yaml
services/research-media/**         (app/ -> research_media/ rename, separate service)
README.md  pyproject.toml          (mixed edits; split before committing)
```

> Note: `git add -A` would sweep in `.tmp_*`, `.workbuddy-ai/` and `acceptance_results.json` —
> only `/legacy` is currently ignored. The list above must be staged by explicit path.
