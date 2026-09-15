import asyncio
import os
import sys
import tempfile
import yaml
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
import httpx

sys.path.insert(0, os.path.abspath("packages/contracts/src"))
for mod_name in list(sys.modules.keys()):
    if mod_name == "app" or mod_name.startswith("app."):
        del sys.modules[mod_name]
sys.path.insert(0, os.path.abspath("services/research-media"))

from app.main import app
from app.parser.drivers.http import GenericHttpDriver
from app.parser.models import (
    ProviderDefinition,
    RequestTemplate,
    RequestFileConfig,
    AsyncPollingConfig,
    AuthConfig,
    ResponseMappingConfig,
    LifecycleConfig,
)
from app.parser.registry import ProviderRegistry, PROVIDER_REGISTRY
from app.parser.manager import PARSER_MANAGER
from app.media.store import LanceMediaStore
from contracts import ParseRequest, ParsedDocument, DocumentFigure, DocumentBlock


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# Test A: 同一个 GenericHttpDriver 实例化两个完全不同 endpoint/config (证明无需供应商代码)
# ==============================================================================
def test_a_generic_driver_multiple_instances():
    def_a = ProviderDefinition(
        name="custom_cloud_alpha",
        driver="http",
        location="remote",
        endpoint="https://api.alpha-parser.io/v1",
        capabilities=["pdf", "ocr"],
        request=RequestTemplate(method="POST", path="/doc", encoding="json"),
    )
    def_b = ProviderDefinition(
        name="custom_local_beta",
        driver="http",
        location="local",
        endpoint="http://127.0.0.1:9099",
        capabilities=["image", "figures"],
        request=RequestTemplate(method="PUT", path="/process", encoding="multipart"),
    )

    driver_a = GenericHttpDriver(def_a)
    driver_b = GenericHttpDriver(def_b)

    assert driver_a.name == "custom_cloud_alpha"
    assert driver_a.endpoint == "https://api.alpha-parser.io/v1"
    assert driver_a.satisfies_capabilities(["pdf"])
    assert not driver_a.satisfies_capabilities(["figures"])

    assert driver_b.name == "custom_local_beta"
    assert driver_b.endpoint == "http://127.0.0.1:9099"
    assert driver_b.satisfies_capabilities(["figures"])
    assert not driver_b.satisfies_capabilities(["pdf"])


# ==============================================================================
# Test B: 同步 API
# ==============================================================================
def test_b_sync_api():
    definition = ProviderDefinition(
        name="sync_parser",
        driver="http",
        endpoint="https://api.test-sync.com",
        request=RequestTemplate(method="POST", path="/v1/parse", encoding="json"),
        response=ResponseMappingConfig(markdown_path="result.markdown"),
    )
    driver = GenericHttpDriver(definition)

    sync_response_payload = {
        "status": 0,
        "result": {
            "markdown": "# Synchronous Parse Result\nExtracted smoothly.",
        }
    }

    mock_resp = httpx.Response(200, json=sync_response_payload, request=httpx.Request("POST", "https://api.test-sync.com/v1/parse"))

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_resp

            req = ParseRequest(file_content_base64="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
            doc = await driver.parse(req)

            assert doc.markdown == "# Synchronous Parse Result\nExtracted smoothly."
            assert doc.raw_provider_result == sync_response_payload

    asyncio.run(_run())


# ==============================================================================
# Test C: submit + poll 异步 API
# ==============================================================================
def test_c_async_submit_and_poll():
    definition = ProviderDefinition(
        name="async_parser",
        driver="http",
        endpoint="https://api.test-async.com",
        request=RequestTemplate(method="POST", path="/tasks", encoding="json"),
        async_config=AsyncPollingConfig(
            enabled=True,
            job_id_path="data.task_id",
            poll_path="/tasks/{job_id}/status",
            poll_method="GET",
            status_path="data.state",
            success_values=["completed", "success"],
            failure_values=["failed"],
            result_path="data.result",
            poll_interval_seconds=0.01,
            max_poll_seconds=5.0,
        ),
        response=ResponseMappingConfig(markdown_path="markdown_body"),
    )
    driver = GenericHttpDriver(definition)

    # 1. Submit response
    submit_resp = httpx.Response(200, json={"code": 0, "data": {"task_id": "job-999"}}, request=httpx.Request("POST", "https://api.test-async.com/tasks"))
    # 2. Poll 1: running
    poll_1 = httpx.Response(200, json={"data": {"state": "running"}}, request=httpx.Request("GET", "https://api.test-async.com/tasks/job-999/status"))
    # 3. Poll 2: completed
    poll_2 = httpx.Response(200, json={
        "data": {
            "state": "completed",
            "result": {
                "markdown_body": "# Asynchronously Polled Content\nSuccess!",
            }
        }
    }, request=httpx.Request("GET", "https://api.test-async.com/tasks/job-999/status"))

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [submit_resp, poll_1, poll_2]

            req = ParseRequest(file_url="https://arxiv.org/pdf/2401.00001.pdf")
            doc = await driver.parse(req)

            assert doc.markdown == "# Asynchronously Polled Content\nSuccess!"
            assert mock_req.call_count == 3

    asyncio.run(_run())


# ==============================================================================
# Test D: multipart file upload
# ==============================================================================
def test_d_multipart_file_upload(tmp_path):
    dummy_pdf = tmp_path / "sample.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 test binary data")

    definition = ProviderDefinition(
        name="multipart_parser",
        driver="http",
        endpoint="https://api.test-multipart.com",
        request=RequestTemplate(
            method="POST",
            path="/upload",
            encoding="multipart",
            file=RequestFileConfig(mode="multipart", field="document_file"),
            fields={"extract_tables": True},
        ),
        response=ResponseMappingConfig(markdown_path="extracted_text"),
    )
    driver = GenericHttpDriver(definition)

    mock_resp = httpx.Response(200, json={"extracted_text": "Multipart PDF extracted"}, request=httpx.Request("POST", "https://api.test-multipart.com/upload"))

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_resp

            req = ParseRequest(file_path=str(dummy_pdf))
            doc = await driver.parse(req)

            assert doc.markdown == "Multipart PDF extracted"
            # Verify call kwargs included 'files' and 'data'
            kwargs = mock_req.call_args[1]
            assert "document_file" in kwargs["files"]
            filename, content, mime = kwargs["files"]["document_file"]
            assert filename == "sample.pdf"
            assert content == b"%PDF-1.4 test binary data"
            assert kwargs["data"]["extract_tables"] is True

    asyncio.run(_run())


# ==============================================================================
# Test E: base64 JSON upload
# ==============================================================================
def test_e_base64_json_upload():
    definition = ProviderDefinition(
        name="base64_json_parser",
        driver="http",
        endpoint="https://api.test-base64.com",
        request=RequestTemplate(
            method="POST",
            path="/extract_base64",
            encoding="json",
            file=RequestFileConfig(mode="base64", field="image_b64"),
            fields={"lang": "en"},
        ),
        response=ResponseMappingConfig(markdown_path="text_content"),
    )
    driver = GenericHttpDriver(definition)

    mock_resp = httpx.Response(200, json={"text_content": "Base64 image parsed"}, request=httpx.Request("POST", "https://api.test-base64.com/extract_base64"))

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_resp

            raw_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            req = ParseRequest(file_content_base64=raw_b64)
            doc = await driver.parse(req)

            assert doc.markdown == "Base64 image parsed"
            json_payload = mock_req.call_args[1]["json"]
            assert json_payload["image_b64"] == raw_b64
            assert json_payload["lang"] == "en"

    asyncio.run(_run())


# ==============================================================================
# Test F: Bearer 和 API-Key 两种 auth
# ==============================================================================
def test_f_bearer_and_apikey_auth():
    # 1. Bearer auth via token_env
    os.environ["MOCK_BEARER_TOKEN"] = "secret-token-xyz"
    def_bearer = ProviderDefinition(
        name="bearer_parser",
        driver="http",
        endpoint="https://api.bearer.com",
        auth=AuthConfig(type="bearer", token_env="MOCK_BEARER_TOKEN"),
    )
    driver_bearer = GenericHttpDriver(def_bearer)
    headers, params = driver_bearer._build_auth()
    assert headers["Authorization"] == "Bearer secret-token-xyz"

    # 2. Custom header / API-Key auth
    os.environ["MOCK_CUSTOM_KEY"] = "my-custom-apikey-123"
    def_header = ProviderDefinition(
        name="header_parser",
        driver="http",
        endpoint="https://api.header.com",
        auth=AuthConfig(type="header", header_name="X-API-KEY", token_env="MOCK_CUSTOM_KEY"),
    )
    driver_header = GenericHttpDriver(def_header)
    headers2, params2 = driver_header._build_auth()
    assert headers2["X-API-KEY"] == "my-custom-apikey-123"


# ==============================================================================
# Test G: Response Mapping via JMESPath
# ==============================================================================
def test_g_declarative_response_mapping():
    definition = ProviderDefinition(
        name="mapping_test_parser",
        driver="http",
        endpoint="https://api.mapping-test.com",
        response=ResponseMappingConfig(
            markdown_path="payload.document.full_text",
            blocks_path="payload.document.elements",
            block_type_path="category",
            block_content_path="body",
            block_page_path="page_no",
            figures_path="payload.media.figures",
            figure_id_path="fig_code",
            figure_caption_path="legend",
            figure_image_path="base64_data",
            tables_path="payload.media.tables",
            table_id_path="tab_code",
            table_markdown_path="grid_markdown",
        ),
    )
    driver = GenericHttpDriver(definition)

    raw_response = {
        "status": "ok",
        "payload": {
            "document": {
                "full_text": "# Paper Title\nIntroduction paragraph.",
                "elements": [
                    {"category": "heading", "body": "# Paper Title", "page_no": 1},
                    {"category": "text", "body": "Introduction paragraph.", "page_no": 1},
                ]
            },
            "media": {
                "figures": [
                    {"fig_code": "F1", "legend": "Figure 1: Architecture", "base64_data": "dummy_img_str", "page": 2}
                ],
                "tables": [
                    {"tab_code": "T1", "grid_markdown": "| Col1 | Col2 |\n|---|---|\n| A | B |", "page": 3}
                ]
            }
        }
    }

    mock_resp = httpx.Response(200, json=raw_response, request=httpx.Request("POST", "https://api.mapping-test.com"))

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_resp
            doc = await driver.parse(ParseRequest(file_url="https://test.com/paper.pdf"))

            assert doc.markdown == "# Paper Title\nIntroduction paragraph."
            assert len(doc.blocks) == 2
            assert doc.blocks[0].type == "heading"
            assert doc.blocks[0].content == "# Paper Title"
            assert len(doc.figures) == 1
            assert doc.figures[0].figure_id == "F1"
            assert doc.figures[0].caption == "Figure 1: Architecture"
            assert len(doc.tables) == 1
            assert doc.tables[0].table_id == "T1"
            assert "| Col1 | Col2 |" in doc.tables[0].markdown

    asyncio.run(_run())


# ==============================================================================
# Test H: 新增 future_parser 只修改 YAML，不修改任何 Python 文件！
# ==============================================================================
def test_h_future_parser_yaml_only(tmp_path):
    """
    Simulates introducing a completely new provider 'future_parser' exclusively through
    YAML configuration without modifying a single line of Python code.
    """
    yaml_content = """
providers:
  future_parser:
    driver: http
    location: remote
    endpoint: https://api.future-ai-labs.org/v2
    enabled: true
    priority: 95
    capabilities:
      - pdf
      - layout
      - figures
    auth:
      type: header
      header_name: X-Future-Token
      token: test-future-token-2030
    request:
      method: POST
      path: /submit_doc
      encoding: json
      fields:
        ai_pipeline: "super-resolution-ocr"
    async:
      enabled: true
      job_id_path: ticket.id
      poll_path: /ticket/{job_id}/state
      status_path: ticket.status
      success_values: ["finished"]
      result_path: ticket.final_output
      poll_interval_seconds: 0.01
    response:
      markdown_path: article_markdown
      figures_path: illustrations
      figure_caption_path: description
      figure_image_path: b64_img
"""
    cfg_file = tmp_path / "future_providers.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    # Load fresh registry from YAML ONLY
    registry = ProviderRegistry(config_path=str(cfg_file))
    future_driver = registry.get_driver("future_parser")

    assert future_driver is not None
    assert future_driver.name == "future_parser"
    assert future_driver.satisfies_capabilities(["pdf", "figures"])

    # Simulate future provider API responses
    submit_resp = httpx.Response(200, json={"ticket": {"id": "future-ticket-777"}}, request=httpx.Request("POST", "https://api.future-ai-labs.org/v2/submit_doc"))
    poll_resp = httpx.Response(200, json={
        "ticket": {
            "status": "finished",
            "final_output": {
                "article_markdown": "# Future AI Analysis (2030)\nZero Python changes needed!",
                "illustrations": [
                    {"description": "Quantum circuit diagram", "b64_img": "future_img_payload"}
                ]
            }
        }
    }, request=httpx.Request("GET", "https://api.future-ai-labs.org/v2/ticket/future-ticket-777/state"))

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [submit_resp, poll_resp]

            req = ParseRequest(file_url="https://future.org/quantum.pdf", needs=["pdf", "figures"])
            doc = await future_driver.parse(req)

            assert doc.markdown == "# Future AI Analysis (2030)\nZero Python changes needed!"
            assert len(doc.figures) == 1
            assert doc.figures[0].caption == "Quantum circuit diagram"
            assert doc.figures[0].image == "future_img_payload"

    asyncio.run(_run())


# ==============================================================================
# General Media Service & LanceDB Tests
# ==============================================================================
def test_list_providers(client):
    res = client.get("/v1/providers")
    assert res.status_code == 200
    data = res.json()
    assert "providers" in data
    # Providers from config/providers.example.yaml
    names = [p["name"] for p in data["providers"]]
    assert "mineru_cloud" in names


def test_privacy_routing():
    req_private = ParseRequest(privacy="private", needs=["pdf", "layout"])
    candidates_private = PARSER_MANAGER.select_candidates(req_private)
    for c in candidates_private:
        assert c.definition.location == "local", f"Security violation: Private request routed to remote: {c.name}"


def test_capability_routing():
    # Need formula -> generic_local_ocr has only [image, ocr], so it must be excluded
    req = ParseRequest(privacy="public", needs=["formula"])
    candidates = PARSER_MANAGER.select_candidates(req)
    cand_names = [c.name for c in candidates]
    assert "generic_local_ocr" not in cand_names


def test_media_ingest_and_search(tmp_path, client):
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

            # Ingest same again -> deduplication test
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
