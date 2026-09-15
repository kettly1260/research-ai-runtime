import pytest
import os
import sys
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("packages/contracts/src"))
sys.path.insert(0, os.path.abspath("services/research-media"))

from app.main import app
from app.parser.manager import PARSER_MANAGER
from app.media.store import LanceMediaStore
from contracts import ParseRequest, ParsedDocument, DocumentFigure, DocumentBlock


@pytest.fixture
def client():
    return TestClient(app)


def test_list_providers(client):
    res = client.get("/v1/providers")
    assert res.status_code == 200
    data = res.json()
    assert "providers" in data
    names = [p["name"] for p in data["providers"]]
    assert "mineru_cloud" in names


def test_privacy_routing():
    # Private request MUST NOT include remote providers
    req_private = ParseRequest(privacy="private", needs=["pdf", "layout"])
    candidates_private = PARSER_MANAGER.select_candidates(req_private)
    for c in candidates_private:
        assert c.config.location == "local", f"Private request routed to remote provider: {c.name}"

    # Public request can include remote providers
    req_public = ParseRequest(privacy="public", needs=["pdf", "layout"])
    candidates_public = PARSER_MANAGER.select_candidates(req_public)
    remote_names = [c.name for c in candidates_public if c.config.location == "remote"]
    assert len(remote_names) > 0


def test_capability_routing():
    # Need formula -> openvino_ocr does not support formula, so it should be excluded
    req = ParseRequest(privacy="public", needs=["formula"])
    candidates = PARSER_MANAGER.select_candidates(req)
    cand_names = [c.name for c in candidates]
    assert "openvino_ocr" not in cand_names


def test_parser_fallback_flow():
    import asyncio

    async def _run():
        req = ParseRequest(privacy="public", needs=["pdf", "layout"])
        candidates = PARSER_MANAGER.select_candidates(req)
        assert len(candidates) >= 2

        first = candidates[0]
        second = candidates[1]

        with patch.object(first, "parse", side_effect=RuntimeError("Cloud API Timeout")):
            dummy_doc = ParsedDocument(
                document_id="doc-123",
                markdown="# Parsed successfully by fallback",
                pages=[],
                blocks=[],
                figures=[],
                tables=[],
            )
            with patch.object(second, "parse", return_value=dummy_doc):
                doc = await PARSER_MANAGER.parse(req)
                assert doc.document_id == "doc-123"
                assert doc.provider_info is not None
                assert doc.provider_info.status == "fallback"
                assert doc.provider_info.provider_name == second.name
                assert "Cloud API Timeout" in str(doc.provider_info.fallback_reason)
                assert first.name in doc.provider_info.attempted_providers

    asyncio.run(_run())


def test_media_ingest_and_search(tmp_path, client):
    # Test LanceDB store with temporary directory
    temp_store = LanceMediaStore(data_dir=str(tmp_path / "test_media.lance"))

    doc = ParsedDocument(
        document_id="paper-001",
        markdown="Sample paper text",
        pages=[],
        blocks=[],
        figures=[
            DocumentFigure(
                figure_id="fig1",
                page=1,
                image="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                caption="Figure 1: XRD spectrum",
            )
        ],
        tables=[],
    )

    from app.media import MEDIA_INGESTOR, MEDIA_SEARCH

    with patch.object(MEDIA_INGESTOR, "store", temp_store), \
         patch.object(MEDIA_SEARCH, "store", temp_store):

        with patch("app.media.ingest.GatewayClient.get_image_embedding", new_callable=AsyncMock) as mock_img_emb, \
             patch("app.media.ingest.GatewayClient.get_dino_embedding", new_callable=AsyncMock) as mock_dino_emb:

            mock_img_emb.return_value = [[0.1] * 512]
            mock_dino_emb.return_value = [[0.2] * 384]

            # Ingest
            ingest_payload = {
                "document": doc.model_dump(),
                "source_uri": "paper-001",
            }
            res = client.post("/v1/media/ingest", json=ingest_payload)
            assert res.status_code == 200
            ingest_res = res.json()
            assert ingest_res["ingested_count"] == 1
            assert ingest_res["skipped_duplicates"] == 0

            # Ingest same again -> should be deduplicated (skipped_duplicates = 1)
            res2 = client.post("/v1/media/ingest", json=ingest_payload)
            assert res2.status_code == 200
            assert res2.json()["skipped_duplicates"] == 1

        # Search Text to Image
        with patch("app.media.search.GatewayClient.get_text_embedding", new_callable=AsyncMock) as mock_txt_emb:
            mock_txt_emb.return_value = [[0.1] * 512]
            search_payload = {
                "mode": "text_to_image",
                "query_text": "XRD spectrum",
                "top_k": 5,
            }
            res_search = client.post("/v1/media/search", json=search_payload)
            assert res_search.status_code == 200
            s_data = res_search.json()
            assert s_data["mode"] == "text_to_image"
            assert s_data["count"] == 1
            assert s_data["results"][0]["caption"] == "Figure 1: XRD spectrum"

        # Search Image to Image (DINO)
        with patch("app.media.search.GatewayClient.get_dino_embedding", new_callable=AsyncMock) as mock_dino_search:
            mock_dino_search.return_value = [[0.2] * 384]
            search_img_payload = {
                "mode": "image_to_image",
                "query_image": "dummy_img_b64",
                "top_k": 5,
            }
            res_dino = client.post("/v1/media/search", json=search_img_payload)
            assert res_dino.status_code == 200
            d_data = res_dino.json()
            assert d_data["mode"] == "image_to_image"
            assert d_data["count"] == 1
