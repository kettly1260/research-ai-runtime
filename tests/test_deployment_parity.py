"""Deployment parity guards for the legacy -> modular cutover.

Two invariants are locked here:

1. **OVMS is untouchable.** The production deploy file must contain the gateway
   and nothing else, so no supported command can recreate `ovms-server`.
2. **Environment parity.** The production settings are the measured legacy
   baseline, not the code defaults. Several keys differ between the two, and a
   silent fallback to a code default would change runtime behaviour.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_COMPOSE = REPO_ROOT / "deploy" / "docker-compose.production.yml"
FULL_COMPOSE = REPO_ROOT / "deploy" / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / "deploy" / "env.example"
DOCKER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker.yml"
TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"

# Measured production baseline of the legacy gateway. These values are the
# parity reference; they must appear explicitly in the production deploy file.
PRODUCTION_BASELINE = {
    "MODEL_IDLE_UNLOAD_SECONDS": "1200",
    "RERANK_MAX_DOCS": "64",
    "RERANK_MAX_LENGTH": "128",
    "RERANK_BATCH_SIZE": "1",
    "BGE_BATCH_SIZE": "8",
    "HF_ENDPOINT": "https://hf-mirror.com",
}

# Keys whose code default differs from the production baseline. If one of these
# is dropped from the deploy file the gateway would silently change behaviour.
DRIFT_PRONE_KEYS = {
    "MODEL_IDLE_UNLOAD_SECONDS",
    "RERANK_MAX_DOCS",
    "RERANK_MAX_LENGTH",
    "RERANK_BATCH_SIZE",
    "BGE_BATCH_SIZE",
    "HF_ENDPOINT",
}

# The subset that is read by `research_ai_gateway.config` AND whose code default
# genuinely differs from the production baseline.
DIVERGENT_FROM_CODE_DEFAULT = {
    "RERANK_MAX_DOCS",
    "RERANK_MAX_LENGTH",
    "RERANK_BATCH_SIZE",
    "BGE_BATCH_SIZE",
}


def _load(path):
    assert path.exists(), f"missing deploy file: {path}"
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _env_map(service):
    env = {}
    for entry in service.get("environment", []) or []:
        key, _, value = str(entry).partition("=")
        env[key] = value
    return env


def test_production_compose_declares_only_the_gateway():
    compose = _load(PRODUCTION_COMPOSE)
    assert set(compose["services"]) == {"ai-gateway"}, (
        "the production deploy file must not declare an ovms service, otherwise a "
        "supported command could recreate the running ovms-server"
    )


def test_production_compose_never_names_ovms_container():
    text = PRODUCTION_COMPOSE.read_text(encoding="utf-8")
    assert "container_name: ovms-server" not in text
    assert "/dev/dri" not in text
    assert "privileged: true" not in text


def test_production_compose_pins_the_production_baseline():
    env = _env_map(_load(PRODUCTION_COMPOSE)["services"]["ai-gateway"])
    for key, expected in PRODUCTION_BASELINE.items():
        actual = env.get(key)
        assert actual is not None, f"{key} missing from the production deploy file"
        # Values may be written as ${KEY:-default}; the default is what matters.
        assert expected in actual, f"{key} does not pin the production value {expected}: {actual}"


def test_production_compose_sets_every_drift_prone_key():
    env = _env_map(_load(PRODUCTION_COMPOSE)["services"]["ai-gateway"])
    missing = DRIFT_PRONE_KEYS - set(env)
    assert not missing, f"drift-prone keys not pinned: {sorted(missing)}"


def test_production_compose_uses_polling_not_the_reload_api():
    env = _env_map(_load(PRODUCTION_COMPOSE)["services"]["ai-gateway"])
    assert env["OVMS_CONFIG_UPDATE_MODE"] == "poll"
    # The global reload endpoint is what produced the END tombstones.
    text = PRODUCTION_COMPOSE.read_text(encoding="utf-8")
    assert "OVMS_CONFIG_RELOAD_URL" not in text


def test_production_compose_mounts_the_registry_authority():
    service = _load(PRODUCTION_COMPOSE)["services"]["ai-gateway"]
    mounts = service["volumes"]
    assert any(m.endswith(":/config:ro") for m in mounts), mounts
    assert any(":/ovms-config:rw" in m for m in mounts), mounts
    env = _env_map(service)
    assert env["MODELS_CONFIG_PATH"] == "/config/models.yaml"


def test_production_compose_joins_the_existing_external_network():
    compose = _load(PRODUCTION_COMPOSE)
    assert compose["networks"]["ai_network"]["external"] is True


def test_full_stack_compose_carries_an_explicit_production_warning():
    text = FULL_COMPOSE.read_text(encoding="utf-8")
    assert "DO NOT RUN THIS FILE ON THE PRODUCTION UNRAID HOST" in text
    assert "docker-compose.production.yml" in text


def test_full_stack_compose_does_not_use_the_stale_idle_ttl():
    env = _env_map(_load(FULL_COMPOSE)["services"]["ai-gateway"])
    assert "600" not in env["MODEL_IDLE_UNLOAD_SECONDS"], env["MODEL_IDLE_UNLOAD_SECONDS"]


def test_env_example_documents_the_baseline():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for key, expected in PRODUCTION_BASELINE.items():
        assert f"{key}={expected}" in text, f"env.example must document {key}={expected}"
    assert "OVMS_CONFIG_UPDATE_MODE=poll" in text


def test_cutover_dockerfile_adds_pillow_on_the_proven_base():
    dockerfile = REPO_ROOT / "services" / "ai-gateway" / "Dockerfile.cutover"
    assert dockerfile.exists()
    text = dockerfile.read_text(encoding="utf-8")
    assert "ARG CUTOVER_BASE_IMAGE=ghcr.io/kettly1260/research-ai-runtime-gateway-base:20260915" in text
    assert "FROM ${CUTOVER_BASE_IMAGE}" in text
    assert "pillow" in text
    assert "research_ai_gateway.main:app" in text


def test_cutover_base_rebuilds_the_frozen_qualified_runtime():
    dockerfile = REPO_ROOT / "services" / "ai-gateway" / "Dockerfile.cutover-base"
    freeze = REPO_ROOT / "services" / "ai-gateway" / "requirements.cutover-base.txt"
    assert dockerfile.exists()
    assert freeze.exists()
    text = dockerfile.read_text(encoding="utf-8")
    frozen = freeze.read_text(encoding="utf-8")
    assert "ARG PYTHON_BASE_IMAGE=python:3.11.15" in text
    assert "FROM ${PYTHON_BASE_IMAGE}" in text
    assert "requirements.cutover-base.txt" in text
    assert "python -m pip check" in text
    for expected in (
        "fastapi==0.135.3",
        "transformers==5.5.4",
        "tokenizers==0.22.2",
        "numpy==2.4.4",
        "uvicorn==0.44.0",
        "httpx==0.28.1",
        "pydantic==2.13.0",
        "PyYAML==6.0.3",
    ):
        assert expected in frozen
    assert "org.opencontainers.image.source=\"https://github.com/kettly1260/research-ai-runtime\"" in text


def test_cutover_branch_runs_ci_and_builds_a_sha_pinned_ghcr_candidate():
    docker = DOCKER_WORKFLOW.read_text(encoding="utf-8")
    tests = TEST_WORKFLOW.read_text(encoding="utf-8")

    assert "cutover-modular-ai-gateway" in docker
    assert "workflow_dispatch" in docker
    assert "build-gateway-cutover-base" in docker
    assert "build-gateway-cutover" in docker
    assert "needs: build-gateway-cutover-base" in docker
    assert "services/ai-gateway/Dockerfile.cutover-base" in docker
    assert "research-ai-runtime-gateway-base" in docker
    assert "services/ai-gateway/Dockerfile.cutover" in docker
    assert "CUTOVER_BASE_IMAGE=${{ env.REGISTRY }}/${{ env.IMAGE_GATEWAY_BASE }}:20260915" in docker
    assert "type=raw,value=cutover-${{ github.sha }}" in docker
    assert "cutover-modular-ai-gateway" in tests


def test_cutover_dockerfile_does_not_copy_into_the_shadowed_app_dir():
    """`/app` is bind-mounted at runtime, so baking sources there is a no-op."""
    dockerfile = REPO_ROOT / "services" / "ai-gateway" / "Dockerfile.cutover"
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("COPY"):
            assert "/app" not in stripped, (
                f"COPY into the bind-mounted /app is shadowed at runtime: {stripped}"
            )


@pytest.mark.parametrize("key", sorted(DIVERGENT_FROM_CODE_DEFAULT))
def test_code_default_would_break_parity(key):
    """Each pinned key genuinely diverges from its code default.

    This is what makes pinning load-bearing: if the deploy file omitted the key,
    the gateway would silently fall back to a different value. `HF_ENDPOINT` is
    excluded because it is consumed by huggingface_hub, not by gateway config.
    """
    from research_ai_gateway import config as cfg

    code_default = str(getattr(cfg, key))
    assert code_default != PRODUCTION_BASELINE[key], (
        f"{key}: code default ({code_default}) equals the baseline, so this guard is stale"
    )


def test_idle_unload_seconds_is_pinned_even_though_defaults_agree():
    """The code default matches production, but the full-stack compose once said 600.

    Pinning it explicitly is what stops a compose default from changing residency.
    """
    env = _env_map(_load(PRODUCTION_COMPOSE)["services"]["ai-gateway"])
    assert "MODEL_IDLE_UNLOAD_SECONDS" in env
    assert PRODUCTION_BASELINE["MODEL_IDLE_UNLOAD_SECONDS"] in env["MODEL_IDLE_UNLOAD_SECONDS"]
