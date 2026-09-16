# OVMS 2026.3.1 Migration and Rollback Plan

Scope: how OVMS would be promoted from the validated 2026.1 digest to 2026.3.1,
and how that promotion is reversed.

**Nothing in this document is authorised to execute.** It exists so that the
promotion, if it is ever approved, is a checklist rather than an improvisation.
Until an explicit authorisation is given, production stays exactly as it is
today.

---

## 1. Production state that must not change

| Item | Pinned value |
| --- | --- |
| OVMS image | `openvino/model_server@sha256:22f92a1a5ad6784384e47296c43bde239887a623e75de45b0e1802659de416a6` |
| OVMS version | OVMS 2026.1.0.72cc06244 / OpenVINO 2026.1.0 / OpenVINO GenAI 2026.1 |
| OVMS image ID | `sha256:08ad73dfd651b3535c1069371aac082eff1ab96101219326c641e1dd4f81e13b` |
| Gateway image | `research-ai-runtime-gateway:local-5567-hotfix` |
| Gateway commit | `5567dbb8190d8315877389a92c5157f23a8a7c2a` |
| Gateway protocol | `OVMS_PROTOCOL=tfs` |
| Long text | `POOLED_MAX_TOTAL_TOKENS=32768`, `POOLED_CHUNK_TOKENS=1024`, `POOLED_CHUNK_OVERLAP=128`, `POOLED_CHUNK_BATCH_SIZE=1` |

Two production facts shape the whole plan:

* `openvino/model_server:latest-gpu` has already drifted to 2026.3.1, so the
  production container must never be pulled, recreated or restarted by a
  convenience command. Only the gateway is deployed, with `--no-deps`.
* OVMS polls `config.json` once per second by default, so any write to the
  production runtime directory is an immediate production model load/unload.
  The production runtime directory is not a scratch space.

---

## 2. Preconditions for promotion review

All of these must be true and evidenced before promotion is even discussed:

1. `docs/OVMS_2026_1_VS_2026_3_ACCEPTANCE.md` shows **PASS** for every verdict
   row on 2026.3.1 — compatibility, correctness, long-text liveness,
   concurrency, reranker, zero-resident, DINO, multimodal.
2. Gate 1 (CPU/protocol/functional) and Gate 2 (VM integration) are both green.
3. Gate 3 ran on the real UHD730 with a **pinned 2026.3.1 digest**, and the
   liveness gate passed after every long-text round.
4. Failure counters are all zero: gateway tracebacks, OVMS abnormal restarts,
   `Bad file descriptor`, unhandled 5xx.
5. The 2026.1 side was re-measured on the same acceptance stack, so the
   comparison is like-for-like rather than against a different day's numbers.
6. The exact 2026.3.1 image digest to promote is recorded, together with the
   command that produced it.
7. A rollback rehearsal has been performed at least once.

No model may be skipped. A single failing model is a failed gate, not a caveat.

---

## 3. Promotion outline (requires separate authorisation)

### 3.1 Gateway first, protocol explicitly pinned

The gateway change is independently safe: it defaults to `tfs`, so deploying the
new gateway against the existing 2026.1 OVMS is a no-op behaviourally.

```
OVMS_PROTOCOL=tfs \
docker compose -f deploy/docker-compose.production.yml up -d --no-deps ai-gateway
```

Verify before touching OVMS at all:

* `GET /health` reports `ovms_protocol_configured == ovms_protocol_effective == "tfs"`.
* A long-text embedding and a short-text embedding both return HTTP 200.
* The reranker still returns a 0..1 relevance probability.

### 3.2 OVMS upgrade

Only after 3.1 is verified, and only with an explicitly authorised window:

1. Record the current container identity and the rollback image reference.
2. Stop the gateway first so no inference lands mid-swap.
3. Replace the OVMS container using the **pinned digest**, never `latest-gpu`.
4. Confirm the version actually running (`/v1/config` or `/v2/models` metadata),
   not the tag that was requested.
5. Capture the 2026.3.1 capability responses listed in the acceptance report,
   §5.1.
6. Start the gateway with `OVMS_PROTOCOL=kserve`.
7. Re-run the Gate 3 checks against production, including the liveness gate.

### 3.3 What must not be done during promotion

* Do not run a bare `docker compose up -d` on `deploy/docker-compose.yml`. That
  file's `ovms` service does not match the running production container, so it
  would recreate OVMS.
* Do not set `OVMS_PROTOCOL=auto` in production.
* Do not point any acceptance container at `/mnt/user/appdata/ovms/runtime`.
* Do not delete or overwrite the existing rollback backup.
* Do not `git reset --hard`, `git clean`, `docker system prune` or
  `docker image prune -a`.

---

## 4. Rollback plan

Rollback is two independent switches. Either one alone restores a working
system, which is why they are separated.

### 4.1 Fast rollback — gateway only (seconds)

If 2026.3.1 is running but inference is unhealthy, revert the gateway's protocol
without touching OVMS:

```
OVMS_PROTOCOL=tfs \
docker compose -f deploy/docker-compose.production.yml up -d --no-deps ai-gateway
```

This only works while OVMS still serves the Classic Model REST API. On a genuine
2026.3.1 it does not, so this switch is for gateway-side regressions only.

### 4.2 Full rollback — gateway image + OVMS digest

1. Stop the gateway.
2. Restore the OVMS container to the validated digest:

```
openvino/model_server@sha256:22f92a1a5ad6784384e47296c43bde239887a623e75de45b0e1802659de416a6
```

3. Restore the gateway image to `research-ai-runtime-gateway:local-5567-hotfix`
   (commit `5567dbb8190d8315877389a92c5157f23a8a7c2a`).
4. Deploy with `--no-deps` so OVMS is not recreated a second time.
5. Confirm the running OVMS image ID is
   `sha256:08ad73dfd651b3535c1069371aac082eff1ab96101219326c641e1dd4f81e13b`.
6. Confirm `OVMS_PROTOCOL=tfs`.
7. Re-run: single long-text embedding, 2×4591-token concurrency, immediate
   short-text liveness, reranker ordering, zero-resident unload.

### 4.3 What rollback must not do

* It must not restore an OVMS runtime config from an acceptance directory.
* It must not restore models or tokenizers.
* It must not use `latest-gpu` as the "known good" image. The known good image
  is a digest.

---

## 5. Version pinning policy

| Environment | Rule |
| --- | --- |
| Production | Pin the validated **digest**. Tags are informational only. |
| Acceptance | Pin the validated **digest**; the compose file refuses to start without one. |
| Development / CI | A validated version tag is acceptable. `latest-gpu` is never acceptable. |

Rationale: `latest-gpu` moved from 2026.1 to 2026.3.1 on its own and silently
removed the Classic Model REST API. Any deployment that follows a moving tag has
an unpinned, unversioned compatibility dependency.

---

## 6. Why `auto` is not a production option

`OVMS_PROTOCOL=auto` exists for development, acceptance and CI. It is
deliberately not a production default:

* It resolves the protocol from live capability probes, so a backend change can
  change the wire protocol without a gateway deploy.
* Its tie-break ladder depends on real server behaviour that is documented but
  not yet re-validated against a pinned 2026.3.1 image (acceptance report, §5.1).
* When it cannot decide, it fails closed with HTTP 502. That is the correct
  behaviour, but it is an outage, and it is not something production should be
  able to reach by configuration drift.

Production pins `tfs` today, and would pin `kserve` after a promotion. Explicit
is boring; boring is what an inference path should be.

---

## 7. Interface stability guarantee

The adapter layer keeps two contracts stable across the protocol change:

* `ovms_predict(model_name, payload, timeout)` — unchanged signature.
* Canonical payload `{"instances": [...]}` and canonical response
  `{"predictions": [...]}` — unchanged shapes.

Consequently `inference/embeddings.py`, `inference/rerank.py`,
`inference/dino.py` and `inference/multimodal.py` did not change, and neither did
the long-text chunking path. A future protocol (KServe v3, GenAI-native) is
expected to be added as another adapter rather than another branch inside the
business modules.
