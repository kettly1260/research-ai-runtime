import pytest
import os
import sys
import time
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("packages/contracts/src"))

import importlib.util

def load_service_module(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module

# Load AI Gateway broker
broker_path = os.path.abspath("services/ai-gateway/app/broker.py")
# To allow broker.py to import its relative dependencies, ensure ai-gateway is temporarily in sys.path
sys.path.insert(0, os.path.abspath("services/ai-gateway"))
for k in list(sys.modules.keys()):
    if k == "app" or k.startswith("app."):
        del sys.modules[k]

from app.broker import DEVICE_BROKER, DeviceBroker
from app import broker as ai_gateway_broker_mod

# Now load research-media
sys.path.pop(0)
for k in list(sys.modules.keys()):
    if k == "app" or k.startswith("app."):
        del sys.modules[k]
sys.path.insert(0, os.path.abspath("services/research-media"))

from app.main import app as media_app
from app.parser.manager import PARSER_MANAGER
from app.media.store import LanceMediaStore, MEDIA_STORE
from app.media.ingest import MEDIA_INGESTOR
from app.media.search import MEDIA_SEARCH
from contracts import ParseRequest, ParsedDocument, DocumentFigure


@pytest.fixture
def media_client():
    return TestClient(media_app)


def test_zero_resident_idle_unload():
    """Verify zero-resident model policy: idle models are unloaded after TTL."""
    DEVICE_BROKER.model_loaded["test-model-1"] = True
    DEVICE_BROKER.model_last_used["test-model-1"] = time.time() - 2000

    with patch.object(DEVICE_BROKER, "_set_model_enabled", return_value=True) as mock_unload, \
         patch.object(DEVICE_BROKER, "_sync_model_loaded_state"):
        DEVICE_BROKER._evict_idle_models()
        assert not DEVICE_BROKER.model_loaded.get("test-model-1", False)
        mock_unload.assert_called_with("test-model-1", False)


def test_device_broker_cpu_fallback():
    """Verify Device Broker falls back to CPU if GPU loading fails."""
    with patch.object(ai_gateway_broker_mod, "is_model_available", return_value=False), \
         patch.object(DEVICE_BROKER, "_sync_model_loaded_state"), \
         patch.object(DEVICE_BROKER, "_evict_idle_models"), \
         patch.object(DEVICE_BROKER, "_evict_to_capacity"):

        calls = []
        def mock_set_enabled(name, enabled, target_device="GPU"):
            calls.append(target_device)
            return target_device == "CPU"

        with patch.object(DEVICE_BROKER, "_set_model_enabled", side_effect=mock_set_enabled):
            device_used = DEVICE_BROKER.ensure_model_available("bge-m3-i8", preferred_device="GPU")
            assert device_used == "CPU"
            assert "GPU" in calls
            assert "CPU" in calls


def test_parser_privacy_hard_boundary():
    """Verify that private documents strictly forbid remote cloud parsers."""
    req = ParseRequest(privacy="private", needs=["pdf", "layout", "figures"])
    candidates = PARSER_MANAGER.select_candidates(req)

    assert len(candidates) > 0
    for cand in candidates:
        assert cand.definition.location == "local", f"Security violation: Private document exposed to {cand.name} ({cand.definition.location})"


def test_parser_quota_exhaustion_backoff():
    """Verify that a provider returning 429 quota exhaustion enters cooldown."""
    req = ParseRequest(privacy="public", needs=["pdf", "layout"])
    candidates = PARSER_MANAGER.select_candidates(req)
    target_prov = candidates[0]

    target_prov.record_failure(Exception("HTTP 429 Quota Exceeded"), quota_exhausted=True)

    assert not target_prov.is_available()
    assert target_prov.status.quota_state == "exhausted"
    assert target_prov.status.cooldown_until is not None

    updated_candidates = PARSER_MANAGER.select_candidates(req)
    assert target_prov not in updated_candidates


def test_full_pipeline_parse_ingest_search(tmp_path, media_client):
    """Full integration flow: Parse document -> Extract figures -> Ingest -> Search."""
    temp_store = LanceMediaStore(data_dir=str(tmp_path / "integration_lance"))

    with patch.object(MEDIA_INGESTOR, "store", temp_store), \
         patch.object(MEDIA_SEARCH, "store", temp_store):

        with patch("app.media.ingest.GatewayClient.get_image_embedding", new_callable=AsyncMock) as mock_img_emb, \
             patch("app.media.ingest.GatewayClient.get_dino_embedding", new_callable=AsyncMock) as mock_dino_emb, \
             patch("app.media.search.GatewayClient.get_text_embedding", new_callable=AsyncMock) as mock_txt_emb:

            mock_img_emb.return_value = [[0.05] * 512]
            mock_dino_emb.return_value = [[0.08] * 384]
            mock_txt_emb.return_value = [[0.05] * 512]

            doc = ParsedDocument(
                document_id="nature_2026_paper",
                markdown="# Novel Superconductor Discovery\nFigure 1 shows magnetic levitation.",
                pages=[],
                blocks=[],
                figures=[
                    DocumentFigure(
                        figure_id="fig_superconduct",
                        page=2,
                        image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        caption="Figure 1: Meissner effect and magnetic levitation",
                        figure_type="chart",
                    )
                ],
                tables=[],
            )

            ingest_res = media_client.post("/v1/media/ingest", json={
                "document": doc.model_dump(),
                "source_uri": "https://doi.org/10.1038/nature12345",
            })
            assert ingest_res.status_code == 200
            assert ingest_res.json()["ingested_count"] == 1

            search_res = media_client.post("/v1/media/search", json={
                "mode": "text_to_image",
                "query_text": "Meissner effect and magnetic levitation",
                "top_k": 3,
            })
            assert search_res.status_code == 200
            s_data = search_res.json()
            assert s_data["count"] == 1
            assert s_data["results"][0]["source_native_id"] == "fig_superconduct"
            assert "Meissner" in s_data["results"][0]["caption"]
