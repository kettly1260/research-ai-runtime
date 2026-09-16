import pytest
import os
import sys
import time
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("packages/contracts/src"))

from research_ai_gateway.broker import DEVICE_BROKER
import research_ai_gateway.broker as ai_gateway_broker_mod

from research_media.main import app as media_app
from research_media.parser.manager import PARSER_MANAGER
from research_media.media.store import LanceMediaStore
from research_media.media.ingest import MEDIA_INGESTOR
from research_media.media.search import MEDIA_SEARCH
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

        with patch("research_media.media.ingest.GatewayClient.get_image_embedding", new_callable=AsyncMock) as mock_img_emb, \
             patch("research_media.media.ingest.GatewayClient.get_dino_embedding", new_callable=AsyncMock) as mock_dino_emb, \
             patch("research_media.media.search.GatewayClient.get_text_embedding", new_callable=AsyncMock) as mock_txt_emb:

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


def test_pdf_bytes_parse_and_ingest_flow(tmp_path, media_client):
    """End-to-end integration: Raw PDF bytes -> Parser -> Structured Document -> LanceDB Ingest."""
    import base64
    import asyncio

    # Minimal valid PDF header and trailer bytes
    pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000114 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"
    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    req = ParseRequest(
        file_content_base64=b64_pdf,
        mime_type="application/pdf",
        needs=["pdf", "figures"]
    )

    temp_store = LanceMediaStore(data_dir=str(tmp_path / "pdf_flow_lance"))

    with patch.object(MEDIA_INGESTOR, "store", temp_store):
        async def _run():
            # Mock driver execution producing structured document from PDF
            candidates = PARSER_MANAGER.select_candidates(req)
            assert len(candidates) > 0

            target_driver = candidates[0]
            with patch.object(PARSER_MANAGER.lifecycle, "prepare_provider", new_callable=AsyncMock) as mock_prep, \
                 patch.object(target_driver, "parse", new_callable=AsyncMock) as mock_parse:
                mock_prep.return_value = True
                mock_parse.return_value = ParsedDocument(
                    document_id="doc_from_pdf_bytes",
                    markdown="# PDF Report Title\n\nContent body from PDF.",
                    figures=[
                        DocumentFigure(
                            figure_id="pdf_fig_1",
                            page=1,
                            caption="Figure 1: PDF extracted diagram",
                            image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        )
                    ]
                )

                parsed_doc = await PARSER_MANAGER.parse(req)
                assert parsed_doc.document_id == "doc_from_pdf_bytes"
                assert len(parsed_doc.figures) == 1
                assert parsed_doc.provider_info.status == "success"

                # Ingest to LanceDB
                with patch("research_media.media.ingest.GatewayClient.get_image_embedding", new_callable=AsyncMock) as mock_img, \
                     patch("research_media.media.ingest.GatewayClient.get_dino_embedding", new_callable=AsyncMock) as mock_dino:
                    mock_img.return_value = [[0.1] * 512]
                    mock_dino.return_value = [[0.2] * 384]

                    result = await MEDIA_INGESTOR.ingest_document(parsed_doc, source_uri="file://local.pdf")
                    assert result["ingested_count"] == 1
                    assert temp_store.count() == 1

        asyncio.run(_run())
