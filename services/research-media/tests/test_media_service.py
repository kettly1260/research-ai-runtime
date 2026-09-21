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
sys.path.insert(0, os.path.abspath("services/research-media"))

from research_media.main import app
from research_media.parser.drivers.http import GenericHttpDriver
from research_media.parser.models import (
    ProviderDefinition,
    RequestTemplate,
    RequestFileConfig,
    AsyncPollingConfig,
    AuthConfig,
    ResponseMappingConfig,
    LifecycleConfig,
)
from research_media.parser.registry import ProviderRegistry, PROVIDER_REGISTRY
from research_media.parser.manager import ParserManager, PARSER_MANAGER
from research_media.media.store import LanceMediaStore
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


def test_parser_config_fail_fast_missing_file():
    """Verify that when PARSER_CONFIG_PATH points to a non-existent file, registry raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ProviderRegistry(config_path="/non/existent/path/providers_missing.yaml")


def test_parser_config_fail_fast_invalid_yaml(tmp_path):
    """Verify that invalid YAML syntax triggers fail-fast."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("providers:\n  broken: [this is unclosed", encoding="utf-8")
    with pytest.raises(RuntimeError):
        ProviderRegistry(config_path=str(bad_yaml))


def test_parser_config_fail_fast_empty_providers(tmp_path):
    """Verify that 0 providers in user-specified config triggers fail-fast ValueError."""
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("providers: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ProviderRegistry(config_path=str(empty_yaml))


def test_declarative_workflow_driver_file_upload_flow(tmp_path):
    """Test multi-step HTTP workflow driver:
    Step 1: POST to /file-urls/batch -> extracts upload_url & batch_id
    Step 2: PUT raw binary file to ${upload_url}
    Step 3: Poll /extract-results/batch/${batch_id} -> done
    Step 4: Normalize response
    """
    dummy_file = tmp_path / "paper.pdf"
    dummy_file.write_bytes(b"%PDF-1.5 test content")

    yaml_content = f"""
providers:
  workflow_mineru:
    driver: http
    location: remote
    endpoint: https://mineru.net/api/v4
    enabled: true
    priority: 100
    auth:
      type: bearer
      token: test-token-123
    workflow:
      - name: apply_upload_url
        type: http
        condition: "file_path != null"
        method: POST
        path: /file-urls/batch
        encoding: json
        body:
          files:
            - name: "${{filename}}"
              data_id: "${{document_id}}"
        exports:
          batch_id: "data.batch_id"
          upload_url: "data.file_urls[0]"
      - name: upload_binary
        type: http
        condition: "file_path != null"
        method: PUT
        url: "${{upload_url}}"
        encoding: binary_file
      - name: poll_result
        type: poll
        method: GET
        path: "/extract-results/batch/${{batch_id}}"
        status_path: "data.extract_result[0].state"
        success_values: ["done"]
        poll_interval_seconds: 0.01
        max_poll_seconds: 5.0
        exports:
          full_zip_url: "data.extract_result[0].full_zip_url"
          extract_result: "data.extract_result[0]"
    response:
      markdown_path: "extract_result.markdown || full_zip_url"
"""
    cfg_file = tmp_path / "workflow_providers.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    registry = ProviderRegistry(config_path=str(cfg_file))
    driver = registry.get_driver("workflow_mineru")
    assert driver is not None

    apply_resp = httpx.Response(200, json={
        "code": 0,
        "data": {
            "batch_id": "batch-abc-123",
            "file_urls": ["https://oss.mineru.net/upload/batch-abc-123/paper.pdf"]
        }
    }, request=httpx.Request("POST", "https://mineru.net/api/v4/file-urls/batch"))

    upload_resp = httpx.Response(200, text="OK", request=httpx.Request("PUT", "https://oss.mineru.net/upload/batch-abc-123/paper.pdf"))

    poll_resp = httpx.Response(200, json={
        "code": 0,
        "data": {
            "batch_id": "batch-abc-123",
            "extract_result": [{
                "state": "done",
                "full_zip_url": "https://cdn.mineru.net/results/batch-abc-123.zip",
                "markdown": "# Extracted Paper Content via Multi-step Workflow"
            }]
        }
    }, request=httpx.Request("GET", "https://mineru.net/api/v4/extract-results/batch/batch-abc-123"))

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [apply_resp, upload_resp, poll_resp]

            req = ParseRequest(file_path=str(dummy_file))
            doc = await driver.parse(req)

            assert doc.markdown == "# Extracted Paper Content via Multi-step Workflow"
            assert mock_req.call_count == 3
            # Check step 2 PUT call uploaded raw binary content
            put_call = mock_req.call_args_list[1]
            assert put_call.args[0] == "PUT"
            assert put_call.args[1] == "https://oss.mineru.net/upload/batch-abc-123/paper.pdf"
            assert put_call.kwargs.get("content") == b"%PDF-1.5 test content"

    asyncio.run(_run())


def test_declarative_workflow_driver_url_flow(tmp_path):
    """Test workflow driver in URL extract mode: branches to URL submission step."""
    yaml_content = """
providers:
  workflow_mineru_url:
    driver: http
    location: remote
    endpoint: https://mineru.net/api/v4
    enabled: true
    priority: 100
    workflow:
      - name: apply_upload_url
        type: http
        condition: "file_path != null"
        method: POST
        path: /file-urls/batch
      - name: submit_url
        type: http
        condition: "file_url != null && file_path == null"
        method: POST
        path: /extract/task/batch
        encoding: json
        body:
          files:
            - url: "${file_url}"
        exports:
          batch_id: "data.batch_id"
      - name: poll_result
        type: poll
        method: GET
        path: "/extract-results/batch/${batch_id}"
        status_path: "data.status"
        success_values: ["done"]
        poll_interval_seconds: 0.01
        exports:
          result_md: "data.markdown"
    response:
      markdown_path: "result_md"
"""
    cfg_file = tmp_path / "url_providers.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    registry = ProviderRegistry(config_path=str(cfg_file))
    driver = registry.get_driver("workflow_mineru_url")

    submit_resp = httpx.Response(200, json={
        "data": {"batch_id": "batch-url-999"}
    }, request=httpx.Request("POST", "https://mineru.net/api/v4/extract/task/batch"))

    poll_resp = httpx.Response(200, json={
        "data": {"status": "done", "markdown": "# URL Mode Extracted Document"}
    }, request=httpx.Request("GET", "https://mineru.net/api/v4/extract-results/batch/batch-url-999"))

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = [submit_resp, poll_resp]

            req = ParseRequest(file_url="https://arxiv.org/pdf/2401.00001.pdf")
            doc = await driver.parse(req)

            assert doc.markdown == "# URL Mode Extracted Document"
            assert mock_req.call_count == 2

    asyncio.run(_run())


def test_generic_command_driver(tmp_path):
    """Test GenericCommandDriver executing external command protocol."""
    from research_media.parser.drivers.command import GenericCommandDriver
    from research_media.parser.models import ProviderDefinition, ResponseMappingConfig

    test_file = tmp_path / "test.txt"
    test_file.write_text("sample content", encoding="utf-8")

    definition = ProviderDefinition(
        name="test_cmd_driver",
        driver="command",
        command=sys.executable,
        args=["-c", "import sys, json; print(json.dumps({'text': 'extracted OCR text from ' + sys.argv[1]}))", "${file_path}"],
        output_format="json",
        response=ResponseMappingConfig(markdown_path="text"),
    )

    driver = GenericCommandDriver(definition)
    assert driver.is_available()

    async def _run():
        req = ParseRequest(file_path=str(test_file))
        doc = await driver.parse(req)
        assert "extracted OCR text from" in doc.markdown
        assert str(test_file) in doc.markdown

    asyncio.run(_run())


# ==============================================================================
# General Media Service & LanceDB Tests
# ==============================================================================
def test_list_providers(client):
    res = client.get("/v1/providers")
    assert res.status_code == 200
    data = res.json()
    assert "providers" in data
    # Providers from the current unified config/providers.example.yaml
    names = [p["name"] for p in data["providers"]]
    assert "mineru" in names
    assert "paddleocr" in names


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


def test_parser_manager_primary_success_returns_immediately():
    """Verify that when the first (primary) provider succeeds:
    - ProviderExecutionInfo.status is 'success' (matching frozen contracts)
    - Returns immediately
    - Does NOT invoke the fallback provider
    """
    async def _test():
        mock_driver1 = MagicMock()
        mock_driver1.name = "primary_prov"
        mock_driver1.is_available.return_value = True
        mock_driver1.satisfies_capabilities.return_value = True
        mock_driver1.definition.location = "remote"
        mock_driver1.definition.priority = 100
        mock_driver1.definition.lifecycle = LifecycleConfig()
        mock_driver1.parse = AsyncMock(return_value=ParsedDocument(
            document_id="doc-1",
            markdown="primary parsed content",
        ))

        mock_driver2 = MagicMock()
        mock_driver2.name = "fallback_prov"
        mock_driver2.is_available.return_value = True
        mock_driver2.satisfies_capabilities.return_value = True
        mock_driver2.definition.location = "remote"
        mock_driver2.definition.priority = 50
        mock_driver2.definition.lifecycle = LifecycleConfig()
        mock_driver2.parse = AsyncMock()

        mock_reg = MagicMock()
        mock_reg.list_drivers.return_value = [mock_driver1, mock_driver2]

        pm = ParserManager(registry=mock_reg)
        req = ParseRequest(privacy="public", needs=["pdf"])
        doc = await pm.parse(req)

        assert doc.markdown == "primary parsed content"
        assert doc.provider_info is not None
        assert doc.provider_info.provider_name == "primary_prov"
        assert doc.provider_info.status == "success"
        assert doc.provider_info.fallback_reason is None
        assert doc.provider_info.attempted_providers == ["primary_prov"]
        mock_driver2.parse.assert_not_called()

    asyncio.run(_test())


def test_parser_manager_fallback_on_first_failure():
    """Verify that when primary provider fails, it gracefully falls back to the second:
    - Status is 'fallback'
    - attempted_providers includes both
    - fallback_reason contains the first error
    """
    async def _test():
        mock_driver1 = MagicMock()
        mock_driver1.name = "primary_prov"
        mock_driver1.is_available.return_value = True
        mock_driver1.satisfies_capabilities.return_value = True
        mock_driver1.definition.location = "remote"
        mock_driver1.definition.priority = 100
        mock_driver1.definition.lifecycle = LifecycleConfig()
        mock_driver1.parse = AsyncMock(side_effect=RuntimeError("Connection refused by upstream"))

        mock_driver2 = MagicMock()
        mock_driver2.name = "fallback_prov"
        mock_driver2.is_available.return_value = True
        mock_driver2.satisfies_capabilities.return_value = True
        mock_driver2.definition.location = "remote"
        mock_driver2.definition.priority = 50
        mock_driver2.definition.lifecycle = LifecycleConfig()
        mock_driver2.parse = AsyncMock(return_value=ParsedDocument(
            document_id="doc-2",
            markdown="fallback parsed content",
        ))

        mock_reg = MagicMock()
        mock_reg.list_drivers.return_value = [mock_driver1, mock_driver2]

        pm = ParserManager(registry=mock_reg)
        req = ParseRequest(privacy="public", needs=["pdf"])
        doc = await pm.parse(req)

        assert doc.markdown == "fallback parsed content"
        assert doc.provider_info.provider_name == "fallback_prov"
        assert doc.provider_info.status == "fallback"
        assert "Connection refused" in (doc.provider_info.fallback_reason or "")
        assert doc.provider_info.attempted_providers == ["primary_prov", "fallback_prov"]

    asyncio.run(_test())


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

    from research_media.media import MEDIA_INGESTOR, MEDIA_SEARCH

    with patch.object(MEDIA_INGESTOR, "store", temp_store), \
         patch.object(MEDIA_SEARCH, "store", temp_store):

        with patch("research_media.media.ingest.GatewayClient.get_image_embedding", new_callable=AsyncMock) as mock_img_emb, \
             patch("research_media.media.ingest.GatewayClient.get_dino_embedding", new_callable=AsyncMock) as mock_dino_emb:

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
        with patch("research_media.media.search.GatewayClient.get_text_embedding", new_callable=AsyncMock) as mock_txt_emb:
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
        with patch("research_media.media.search.GatewayClient.get_dino_embedding", new_callable=AsyncMock) as mock_dino_search:
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


def test_disabled_provider_never_available():
    """Verify enabled: false is strictly respected across all lifecycle modes."""
    from research_media.parser.models import ProviderDefinition, LifecycleConfig
    from research_media.parser.drivers.http import GenericHttpDriver

    # 1. on_demand with enabled: false
    def_on_demand = ProviderDefinition(
        name="test_on_demand_disabled",
        type="generic_http",
        enabled=False,
        lifecycle=LifecycleConfig(mode="on_demand", resource="some_res")
    )
    driver_on_demand = GenericHttpDriver(def_on_demand)
    assert driver_on_demand.is_available() is False

    # 2. model_on_demand with enabled: false
    def_model_on_demand = ProviderDefinition(
        name="test_model_disabled",
        type="generic_http",
        enabled=False,
        lifecycle=LifecycleConfig(mode="model_on_demand", resource="some_model")
    )
    driver_model_on_demand = GenericHttpDriver(def_model_on_demand)
    assert driver_model_on_demand.is_available() is False


def test_generic_http_zip_unpack():
    """Verify GenericHttpDriver downloads and unpacks MinerU full_zip_url to ParsedDocument."""
    import zipfile
    import io
    import json
    from research_media.parser.models import ProviderDefinition, ResponseMappingConfig
    from research_media.parser.drivers.http import GenericHttpDriver

    # Create MinerU official result ZIP using exact official schema:
    # 'image_caption': ["Fig. 1 ..."] (list[str])
    # 'table_caption': ["Table 1 ..."] (list[str])
    # 'table_body': "<html>...</html>"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("full.md", "# Superconducting Analysis\n\nHigh-temperature sample results.")
        content_list = [
            {"type": "text", "text": "Superconducting Analysis", "page_idx": 1},
            {"type": "image", "image_caption": ["Figure 1: Critical field in high-Tc cuprates"], "page_idx": 1, "img_path": "images/fig1.png"},
            {"type": "table", "table_caption": ["Table 1: Transition temperatures and pressure"], "table_body": "<html><tr><td>T</td><td>K</td></tr></html>", "page_idx": 2},
        ]
        zf.writestr("content_list.json", json.dumps(content_list))
        zf.writestr("images/fig1.png", b"\x89PNG\r\n\x1a\nfakeimage")

    fake_zip_bytes = zip_buf.getvalue()

    p_def = ProviderDefinition(
        name="test_mineru_zip",
        type="generic_http",
        enabled=True,
        endpoint="https://mock.mineru.net",
        response=ResponseMappingConfig(
            markdown_path="markdown",
            pages_path="pages",
            figures_path="figures",
            tables_path="tables",
        )
    )
    driver = GenericHttpDriver(p_def)

    class MockZipResponse:
        status_code = 200

        async def aiter_bytes(self, chunk_size=65536):
            yield fake_zip_bytes

    class MockClient:
        def stream(self, method, url, timeout=None):
            from contextlib import asynccontextmanager
            @asynccontextmanager
            async def _stream():
                yield MockZipResponse()
            return _stream()

    async def _run():
        extracted = await driver._unpack_zip_archive(MockClient(), "https://mock.mineru.net/download/result.zip")
        assert "Superconducting Analysis" in extracted["markdown"]
        assert len(extracted["pages"]) >= 1
        assert len(extracted["figures"]) == 1
        # Check that list[str] was cleanly converted to a validated string
        assert isinstance(extracted["figures"][0]["caption"], str)
        assert extracted["figures"][0]["caption"] == "Figure 1: Critical field in high-Tc cuprates"
        assert len(extracted["tables"]) == 1
        assert isinstance(extracted["tables"][0]["caption"], str)
        assert extracted["tables"][0]["caption"] == "Table 1: Transition temperatures and pressure"

        # Verify that normalize_response builds a valid ParsedDocument without Pydantic ValidationError
        from research_media.parser.normalization import normalize_response
        doc = normalize_response(extracted, p_def.response)
        assert doc.markdown == "# Superconducting Analysis\n\nHigh-temperature sample results."
        assert len(doc.figures) == 1
        assert doc.figures[0].caption == "Figure 1: Critical field in high-Tc cuprates"
        assert len(doc.tables) == 1
        assert doc.tables[0].caption == "Table 1: Transition temperatures and pressure"

    asyncio.run(_run())


def test_generic_http_zip_unpack_rejects_malformed_content_list():
    """Malformed MinerU content_list.json must fail instead of silently dropping figures/tables."""
    import io
    import zipfile
    from research_media.parser.models import ProviderDefinition
    from research_media.parser.drivers.http import GenericHttpDriver

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("full.md", "# Valid markdown")
        zf.writestr("content_list.json", "{not-valid-json")
    fake_zip_bytes = zip_buf.getvalue()

    driver = GenericHttpDriver(
        ProviderDefinition(
            name="test_mineru_bad_zip",
            type="generic_http",
            enabled=True,
            endpoint="https://mock.mineru.net",
        )
    )

    class MockZipResponse:
        status_code = 200

        async def aiter_bytes(self, chunk_size=65536):
            yield fake_zip_bytes

    class MockClient:
        def stream(self, method, url, timeout=None):
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def _stream():
                yield MockZipResponse()

            return _stream()

    async def _run():
        with pytest.raises(ValueError, match="Malformed content_list.json"):
            await driver._unpack_zip_archive(
                MockClient(), "https://mock.mineru.net/download/result.zip"
            )

    asyncio.run(_run())


def test_command_driver_timeout():
    """Verify GenericCommandDriver strictly enforces process execution timeout."""
    import sys
    from research_media.parser.models import ProviderDefinition
    from research_media.parser.drivers.command import GenericCommandDriver
    from contracts import ParseRequest

    # Create a command driver that sleeps longer than timeout
    p_def = ProviderDefinition(
        name="hanging_command",
        type="command",
        enabled=True,
        command=sys.executable,
        args=["-c", "import time; time.sleep(5)"],
        timeout=0.5,
    )
    driver = GenericCommandDriver(p_def)
    req = ParseRequest(file_content_base64="aGVsbG8=", mime_type="application/pdf")

    async def _run():
        with pytest.raises(TimeoutError) as exc_info:
            await driver.parse(req)
        assert "timed out after 0.5s" in str(exc_info.value)

    asyncio.run(_run())


def test_provider_registry_hot_reload_is_atomic(tmp_path):
    cfg = tmp_path / "providers.yaml"
    cfg.write_text(
        "providers:\n  parser_a:\n    driver: http\n    location: remote\n"
        "    endpoint: https://parser.example\n    model: model-v1\n    enabled: true\n",
        encoding="utf-8",
    )
    registry = ProviderRegistry(config_path=str(cfg))
    assert registry.get_driver("parser_a").definition.model == "model-v1"

    cfg.write_text(
        "providers:\n  parser_a:\n    driver: http\n    location: remote\n"
        "    endpoint: https://parser.example\n    model: model-v2-longer\n"
        "    api_mode: precision\n    options: {language: ch}\n    enabled: true\n",
        encoding="utf-8",
    )
    driver = registry.get_driver("parser_a")
    assert driver.definition.model == "model-v2-longer"
    assert driver.definition.options["language"] == "ch"
    assert registry.last_reload_error is None

    cfg.write_text("providers:\n  broken: [", encoding="utf-8")
    stale = registry.get_driver("parser_a")
    assert stale is not None
    assert stale.definition.model == "model-v2-longer"
    assert registry.last_reload_error

    cfg.write_text(
        "providers:\n  parser_a:\n    driver: http\n    location: remote\n"
        "    endpoint: https://parser.example\n    model: model-v3-even-longer\n    enabled: true\n",
        encoding="utf-8",
    )
    assert registry.get_driver("parser_a").definition.model == "model-v3-even-longer"
    assert registry.last_reload_error is None


def test_workflow_model_options_and_presigned_auth_boundary(tmp_path):
    dummy_file = tmp_path / "paper.pdf"
    dummy_file.write_bytes(b"%PDF test")
    definition = ProviderDefinition(
        name="generic_precision",
        driver="http",
        location="remote",
        endpoint="https://parser.example",
        model="vlm",
        api_mode="precision",
        options={"language": "ch", "enable_table": True},
        auth=AuthConfig(type="bearer", token="top-secret"),
        workflow=[
            {
                "name": "apply",
                "type": "http",
                "method": "POST",
                "path": "/apply",
                "body": {
                    "model": "${model}",
                    "language": "${options.language}",
                    "enable_table": "${options.enable_table}",
                },
                "exports": {"upload_url": "data.url"},
            },
            {
                "name": "upload",
                "type": "http",
                "method": "PUT",
                "transport": "requests",
                "url": "${upload_url}",
                "encoding": "binary_file",
                "use_auth": False,
                "response_format": "text",
            },
        ],
        response=ResponseMappingConfig(markdown_path="text"),
    )
    driver = GenericHttpDriver(definition)
    apply_resp = httpx.Response(
        200,
        json={"data": {"url": "https://storage.example/presigned"}},
        request=httpx.Request("POST", "https://parser.example/apply"),
    )
    upload_resp = MagicMock()
    upload_resp.status_code = 200
    upload_resp.text = "uploaded"
    upload_resp.content = b"uploaded"

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req, \
             patch("research_media.parser.drivers.http.requests.request") as mock_sync:
            mock_req.side_effect = [apply_resp]
            mock_sync.return_value = upload_resp
            doc = await driver.parse(ParseRequest(file_path=str(dummy_file)))
            assert doc.markdown == "uploaded"
            submit_call = mock_req.call_args_list[0]
            assert submit_call.kwargs["json"] == {
                "model": "vlm",
                "language": "ch",
                "enable_table": True,
            }
            assert submit_call.kwargs["headers"]["Authorization"] == "Bearer top-secret"
            upload_call = mock_sync.call_args
            assert upload_call.args[:2] == (
                "PUT",
                "https://storage.example/presigned",
            )
            assert "Authorization" not in upload_call.kwargs["headers"]
            assert upload_call.kwargs["data"] == b"%PDF test"
            assert "top-secret" not in str(doc.raw_provider_result)
            assert "api_key" not in doc.raw_provider_result
            assert "token" not in doc.raw_provider_result

    asyncio.run(_run())


def test_paddle_style_workflow_markdown_jsonl_and_no_auth_fetch():
    definition = ProviderDefinition(
        name="paddle_style",
        driver="http",
        location="remote",
        endpoint="https://paddle.example",
        model="PP-StructureV3",
        options={"useDocUnwarping": False},
        auth=AuthConfig(type="bearer", token="paddle-secret"),
        workflow=[
            {
                "name": "submit",
                "type": "http",
                "method": "POST",
                "path": "/jobs",
                "encoding": "json",
                "body": {
                    "fileUrl": "${file_url}",
                    "model": "${model}",
                    "optionalPayload": "${options}",
                },
                "exports": {"job_id": "data.jobId"},
            },
            {
                "name": "poll",
                "type": "poll",
                "method": "GET",
                "path": "/jobs/${job_id}",
                "status_path": "data.state",
                "success_values": ["done"],
                "exports": {
                    "jsonl_url": "data.resultUrl.jsonUrl",
                },
                "poll_interval_seconds": 0.01,
            },
            {
                "name": "jsonl",
                "type": "http",
                "method": "GET",
                "transport": "requests",
                "url": "${jsonl_url}",
                "use_auth": False,
                "response_format": "jsonl",
                "retry_statuses": [403, 404],
                "retry_attempts": 3,
                "retry_delay_seconds": 0,
            },
        ],
        response=ResponseMappingConfig(
            markdown_path="jsonl[].result.layoutParsingResults[].markdown.text",
            blocks_path="jsonl[].result.layoutParsingResults[]",
            block_content_path="markdown.text || content || text || res",
        ),
    )
    driver = GenericHttpDriver(definition)
    responses = [
        httpx.Response(
            200,
            json={"data": {"jobId": "job-1"}},
            request=httpx.Request("POST", "https://paddle.example/jobs"),
        ),
        httpx.Response(
            200,
            json={
                "data": {
                    "state": "done",
                    "resultUrl": {
                        "jsonUrl": "https://bos.example/result.jsonl",
                    },
                }
            },
            request=httpx.Request("GET", "https://paddle.example/jobs/job-1"),
        ),
    ]
    denied_resp = MagicMock()
    denied_resp.status_code = 403
    denied_resp.text = '{"code":"AccessDenied","message":"Access Denied."}'
    denied_resp.content = denied_resp.text.encode()
    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.text = (
        '{"result":{"layoutParsingResults":[{"markdown":{"text":"# Page 1"}}]}}\n'
        '{"result":{"layoutParsingResults":[{"markdown":{"text":"# Page 2"}}]}}\n'
    )
    success_resp.content = success_resp.text.encode()

    async def _run():
        with patch.object(httpx.AsyncClient, "request", new_callable=AsyncMock) as mock_req, \
             patch("research_media.parser.drivers.http.requests.request") as mock_sync:
            mock_req.side_effect = responses
            mock_sync.side_effect = [denied_resp, success_resp]
            doc = await driver.parse(ParseRequest(file_url="https://example.org/paper.pdf"))
            assert doc.markdown == "# Page 1\n\n# Page 2"
            assert len(doc.raw_provider_result["jsonl"]) == 2
            assert len(doc.blocks) == 2
            assert mock_req.call_args_list[0].kwargs["json"]["model"] == "PP-StructureV3"
            assert mock_req.call_args_list[0].kwargs["json"]["optionalPayload"] == {
                "useDocUnwarping": False
            }
            assert mock_sync.call_count == 2
            assert "Authorization" not in mock_sync.call_args_list[0].kwargs["headers"]
            assert "Authorization" not in mock_sync.call_args_list[1].kwargs["headers"]
            assert "paddle-secret" not in str(doc.raw_provider_result)
            assert "jsonl_url" not in doc.raw_provider_result

    asyncio.run(_run())
