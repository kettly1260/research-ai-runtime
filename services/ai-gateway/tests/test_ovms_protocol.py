"""Unit tests for the OVMS protocol adapter layer.

Covers the contract required by the 2026.1 / 2026.3 compatibility task:

* protocol selection (``tfs`` / ``kserve`` / ``auto`` / invalid)
* TFS serialisation stays a byte-identical pass-through
* KServe v2 tensor serialisation (``inputs[]`` with name/shape/datatype/data)
* response normalisation parity between ``predictions`` and ``outputs``
* unified error semantics (502 / 504) with protocol + model + status in detail
* auto-detection is fail-closed and cached
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from fastapi import HTTPException

import research_ai_gateway.ovms_client as ovms_client
import research_ai_gateway.ovms_protocol as protocol_mod
from research_ai_gateway.ovms_protocol import (
    protocol_diagnostics,
    reset_protocol_cache,
    resolve_adapter,
)
from research_ai_gateway.ovms_protocol.auto import probe_protocol
from research_ai_gateway.ovms_protocol.base import (
    HttpClient,
    ProtocolDetectionError,
    ProtocolPayloadError,
)
from research_ai_gateway.ovms_protocol.kserve import KserveAdapter
from research_ai_gateway.ovms_protocol.tensors import (
    instances_to_inputs,
    outputs_to_predictions,
    tensor_datatype,
)
from research_ai_gateway.ovms_protocol.tfs import TfsAdapter

GATEWAY_DIR = Path(__file__).resolve().parents[1]


class _NoopHttp(HttpClient):
    def get(self, path, timeout):
        return 200, {}

    def post(self, path, payload, timeout):
        return 200, {}


class _ScriptedHttp(HttpClient):
    """Records calls and replays a canned capability surface."""

    def __init__(self, get_map, post_result=(200, {})):
        self.get_map = get_map
        self.post_result = post_result
        self.calls = []

    def get(self, path, timeout):
        self.calls.append(("GET", path))
        return self.get_map.get(path, (404, None))

    def post(self, path, payload, timeout):
        self.calls.append(("POST", path))
        return self.post_result


class _FakeResponse:
    def __init__(self, status_code, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


@pytest.fixture(autouse=True)
def _isolate_protocol_cache():
    reset_protocol_cache()
    yield
    reset_protocol_cache()


# --------------------------------------------------------------------------- #
# Protocol selection
# --------------------------------------------------------------------------- #

def test_configured_tfs_selects_tfs_adapter(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    assert resolve_adapter(_NoopHttp()).name == "tfs"


def test_configured_kserve_selects_kserve_adapter(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")
    assert resolve_adapter(_NoopHttp()).name == "kserve"


def test_default_protocol_is_tfs_so_2026_1_behaviour_is_unchanged(monkeypatch):
    monkeypatch.delenv("OVMS_PROTOCOL", raising=False)
    env = {**os.environ}
    env.pop("OVMS_PROTOCOL", None)
    env["PYTHONPATH"] = str(GATEWAY_DIR)
    result = subprocess.run(
        [sys.executable, "-c", "import research_ai_gateway.config as c; print(c.OVMS_PROTOCOL)"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "tfs"


def test_invalid_protocol_fails_fast_at_import():
    env = {**os.environ, "OVMS_PROTOCOL": "grpc", "PYTHONPATH": str(GATEWAY_DIR)}
    result = subprocess.run(
        [sys.executable, "-c", "import research_ai_gateway.config"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "OVMS_PROTOCOL must be" in result.stderr


# --------------------------------------------------------------------------- #
# TFS serialisation
# --------------------------------------------------------------------------- #

def test_tfs_request_is_an_unchanged_passthrough():
    payload = {"instances": [{"input_ids": [1, 2], "attention_mask": [1, 1]}]}
    request = TfsAdapter().build_request("bge-m3-i8", payload)
    assert request.path == "/v1/models/bge-m3-i8:predict"
    assert request.payload is payload


def test_tfs_request_rejects_payload_without_instances():
    with pytest.raises(ProtocolPayloadError):
        TfsAdapter().build_request("m", {"inputs": []})


def test_tfs_response_is_returned_verbatim():
    body = {"model_name": "m", "predictions": [[0.1, 0.2]]}
    assert TfsAdapter().normalize_response("m", body, {"instances": [{}]}) is body


def test_tfs_response_rejects_non_object_body():
    with pytest.raises(ProtocolPayloadError):
        TfsAdapter().normalize_response("m", [1, 2, 3], {"instances": [{}]})


# --------------------------------------------------------------------------- #
# KServe v2 serialisation
# --------------------------------------------------------------------------- #

def test_kserve_request_builds_named_shaped_typed_inputs():
    payload = {
        "instances": [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 0]},
        ]
    }
    request = KserveAdapter().build_request("qwen3-embedding-0.6b-int4-pooled", payload)
    assert request.path == "/v2/models/qwen3-embedding-0.6b-int4-pooled/infer"
    assert request.payload == {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [2, 3],
                "datatype": "INT64",
                "data": [1, 2, 3, 4, 5, 6],
            },
            {
                "name": "attention_mask",
                "shape": [2, 3],
                "datatype": "INT64",
                "data": [1, 1, 1, 1, 1, 0],
            },
        ]
    }


def test_kserve_request_preserves_position_ids_for_the_reranker():
    """The reranker's real OVMS signature is input_ids/attention_mask/position_ids."""
    payload = {
        "instances": [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "position_ids": [0, 1]},
        ]
    }
    inputs = instances_to_inputs(payload["instances"])
    assert [item["name"] for item in inputs] == ["input_ids", "attention_mask", "position_ids"]
    assert all(item["shape"] == [1, 2] for item in inputs)


def test_kserve_request_carries_image_tensors_for_vision_models():
    """DINO / multimodal send pixel_values; nothing may be hardcoded to text."""
    payload = {"instances": [{"pixel_values": [[[0.5, -0.5], [0.25, 0.75]]]}]}
    inputs = instances_to_inputs(payload["instances"])
    assert inputs == [
        {
            "name": "pixel_values",
            "shape": [1, 1, 2, 2],
            "datatype": "FP32",
            "data": [0.5, -0.5, 0.25, 0.75],
        }
    ]


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1, 2, 3], "INT64"),
        ([1.0, 2.0], "FP32"),
        ([1, 2.5], "FP32"),
        ([True, False], "BOOL"),
    ],
)
def test_tensor_datatype_inference(values, expected):
    assert tensor_datatype(values) == expected


def test_tensor_datatype_rejects_unsupported_elements():
    with pytest.raises(ProtocolPayloadError):
        tensor_datatype([{"nested": 1}])
    with pytest.raises(ProtocolPayloadError):
        tensor_datatype([True, 1])


def test_kserve_rejects_ragged_instances_instead_of_guessing_padding():
    payload = {"instances": [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]}
    with pytest.raises(ProtocolPayloadError) as exc:
        KserveAdapter().build_request("m", payload)
    assert "ragged" in str(exc.value)


def test_kserve_rejects_instances_with_inconsistent_tensor_names():
    payload = {"instances": [{"input_ids": [1]}, {"attention_mask": [1]}]}
    with pytest.raises(ProtocolPayloadError):
        KserveAdapter().build_request("m", payload)


# --------------------------------------------------------------------------- #
# Response normalisation parity
# --------------------------------------------------------------------------- #

def test_single_output_normalises_like_tfs_predictions():
    payload = {"instances": [{}, {}]}
    body = {
        "model_name": "m",
        "outputs": [
            {"name": "embeddings", "shape": [2, 3], "datatype": "FP32", "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        ],
    }
    result = KserveAdapter().normalize_response("m", body, payload)
    assert result["predictions"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_three_dimensional_output_normalises_like_tfs_predictions():
    """Reranker logits come back as [batch, seq, vocab]."""
    payload = {"instances": [{}, {}]}
    body = {
        "model_name": "qwen-reranker",
        "outputs": [
            {
                "name": "logits",
                "shape": [2, 2, 2],
                "datatype": "FP32",
                "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        ],
    }
    result = KserveAdapter().normalize_response("qwen-reranker", body, payload)
    assert result["predictions"] == [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ]


def test_dynamic_negative_shape_is_resolved_from_the_data_length():
    payload = {"instances": [{}, {}]}
    body = {
        "model_name": "m",
        "outputs": [
            {"name": "out", "shape": [-1, 3], "datatype": "FP32", "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        ],
    }
    result = KserveAdapter().normalize_response("m", body, payload)
    assert result["predictions"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_multiple_outputs_become_one_dict_per_batch_row():
    payload = {"instances": [{}, {}]}
    body = {
        "model_name": "m",
        "outputs": [
            {"name": "text_embeds", "shape": [2, 2], "datatype": "FP32", "data": [1.0, 2.0, 3.0, 4.0]},
            {"name": "image_embeds", "shape": [2, 2], "datatype": "FP32", "data": [5.0, 6.0, 7.0, 8.0]},
        ],
    }
    result = KserveAdapter().normalize_response("m", body, payload)
    assert result["predictions"] == [
        {"text_embeds": [1.0, 2.0], "image_embeds": [5.0, 6.0]},
        {"text_embeds": [3.0, 4.0], "image_embeds": [7.0, 8.0]},
    ]


def test_missing_outputs_is_a_contract_violation():
    with pytest.raises(ProtocolPayloadError):
        outputs_to_predictions(None, 1)
    with pytest.raises(ProtocolPayloadError):
        outputs_to_predictions([], 1)


def test_output_without_data_is_a_contract_violation():
    with pytest.raises(ProtocolPayloadError):
        outputs_to_predictions([{"name": "out", "shape": [1, 2]}], 1)


def test_wrong_leading_dimension_is_a_contract_violation():
    with pytest.raises(ProtocolPayloadError) as exc:
        outputs_to_predictions(
            [{"name": "out", "shape": [3, 2], "datatype": "FP32", "data": [1.0] * 6}],
            2,
        )
    assert "leading dimension" in str(exc.value)


def test_data_length_disagreeing_with_shape_is_a_contract_violation():
    with pytest.raises(ProtocolPayloadError) as exc:
        outputs_to_predictions(
            [{"name": "out", "shape": [1, 4], "datatype": "FP32", "data": [1.0, 2.0]}],
            1,
        )
    assert "declares" in str(exc.value)


def test_non_numeric_output_datatype_is_a_contract_violation():
    with pytest.raises(ProtocolPayloadError) as exc:
        outputs_to_predictions(
            [{"name": "out", "shape": [1, 2], "datatype": "BYTES", "data": ["a", "b"]}],
            1,
        )
    assert "unsupported datatype" in str(exc.value)


def test_non_numeric_output_payload_is_a_contract_violation():
    with pytest.raises(ProtocolPayloadError):
        outputs_to_predictions(
            [{"name": "out", "shape": [1, 2], "data": [None, None]}],
            1,
        )


# --------------------------------------------------------------------------- #
# ovms_predict wiring: URL, body, error semantics
# --------------------------------------------------------------------------- #

def _instances(batch=1, length=2):
    return {"instances": [{"input_ids": list(range(length))} for _ in range(batch)]}


def test_ovms_predict_tfs_uses_v1_predict(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(200, payload={"predictions": [[1.0, 2.0]]})

    with patch.object(ovms_client.requests, "post", side_effect=fake_post):
        result = ovms_client.ovms_predict("bge-m3-i8__gpu", _instances())

    assert captured["url"].endswith("/v1/models/bge-m3-i8__gpu:predict")
    assert captured["json"] == _instances()
    assert result == {"predictions": [[1.0, 2.0]]}


def test_ovms_predict_kserve_uses_v2_infer(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(
            200,
            payload={
                "model_name": "bge-m3-i8",
                "outputs": [
                    {"name": "embeddings", "shape": [1, 2], "datatype": "FP32", "data": [0.5, 0.5]}
                ],
            },
        )

    with patch.object(ovms_client.requests, "post", side_effect=fake_post):
        result = ovms_client.ovms_predict("bge-m3-i8", _instances())

    assert captured["url"].endswith("/v2/models/bge-m3-i8/infer")
    assert captured["json"] == {
        "inputs": [{"name": "input_ids", "shape": [1, 2], "datatype": "INT64", "data": [0, 1]}]
    }
    assert result["predictions"] == [[0.5, 0.5]]


def test_both_protocols_yield_the_same_canonical_result(monkeypatch):
    """The whole point of the adapter layer: identical downstream semantics."""
    payload = _instances(batch=2, length=2)

    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    with patch.object(
        ovms_client.requests,
        "post",
        return_value=_FakeResponse(200, payload={"predictions": [[1.0, 2.0], [3.0, 4.0]]}),
    ):
        tfs_result = ovms_client.ovms_predict("m", payload)

    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")
    with patch.object(
        ovms_client.requests,
        "post",
        return_value=_FakeResponse(
            200,
            payload={
                "model_name": "m",
                "outputs": [
                    {"name": "out", "shape": [2, 2], "datatype": "FP32", "data": [1.0, 2.0, 3.0, 4.0]}
                ],
            },
        ),
    ):
        kserve_result = ovms_client.ovms_predict("m", payload)

    assert tfs_result["predictions"] == kserve_result["predictions"]


@pytest.mark.parametrize("status", [400, 404, 412, 500])
def test_backend_http_errors_map_to_502(monkeypatch, status):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    with patch.object(
        ovms_client.requests,
        "post",
        return_value=_FakeResponse(status, text="backend exploded"),
    ):
        with pytest.raises(HTTPException) as exc:
            ovms_client.ovms_predict("qwen-reranker__gpu", _instances(), timeout=7)

    assert exc.value.status_code == 502
    detail = exc.value.detail
    assert f"HTTP {status}" in detail
    assert "protocol=tfs" in detail
    assert "qwen-reranker__gpu" in detail
    assert "backend exploded" in detail


def test_backend_error_body_is_truncated(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    with patch.object(
        ovms_client.requests,
        "post",
        return_value=_FakeResponse(500, text="x" * 5000),
    ):
        with pytest.raises(HTTPException) as exc:
            ovms_client.ovms_predict("m", _instances())

    assert "..." in exc.value.detail
    assert len(exc.value.detail) < 1000


def test_backend_timeout_maps_to_504(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")
    with patch.object(
        ovms_client.requests,
        "post",
        side_effect=requests.exceptions.Timeout(),
    ):
        with pytest.raises(HTTPException) as exc:
            ovms_client.ovms_predict("m", _instances(), timeout=11)

    assert exc.value.status_code == 504
    assert "11s" in exc.value.detail
    assert "protocol=kserve" in exc.value.detail


def test_connection_error_maps_to_502(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    with patch.object(
        ovms_client.requests,
        "post",
        side_effect=requests.exceptions.ConnectionError("connection refused"),
    ):
        with pytest.raises(HTTPException) as exc:
            ovms_client.ovms_predict("m", _instances())

    assert exc.value.status_code == 502
    assert "protocol=tfs" in exc.value.detail


def test_malformed_json_maps_to_502(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    with patch.object(
        ovms_client.requests,
        "post",
        return_value=_FakeResponse(200, text="<html>not json</html>"),
    ):
        with pytest.raises(HTTPException) as exc:
            ovms_client.ovms_predict("m", _instances())

    assert exc.value.status_code == 502
    assert "invalid JSON" in exc.value.detail


def test_unrepresentable_payload_maps_to_502(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "kserve")
    with pytest.raises(HTTPException) as exc:
        ovms_client.ovms_predict(
            "m",
            {"instances": [{"input_ids": [1, 2, 3]}, {"input_ids": [4]}]},
        )

    assert exc.value.status_code == 502
    assert "cannot serialize" in exc.value.detail


# --------------------------------------------------------------------------- #
# auto detection
# --------------------------------------------------------------------------- #

def test_auto_prefers_tfs_when_only_the_tfs_api_answers():
    http = _ScriptedHttp(
        {
            "/v1/config": (200, {"model_config_list": []}),
            "/v2/health/ready": (404, None),
        }
    )
    decision = probe_protocol(http)
    assert decision.protocol == "tfs"
    assert decision.evidence["tfs_api_present"] is True
    assert decision.evidence["kserve_api_present"] is False


def test_auto_selects_kserve_when_the_tfs_config_api_is_gone():
    http = _ScriptedHttp(
        {
            "/v1/config": (404, None),
            "/v2/health/ready": (200, {"status": "ready"}),
        }
    )
    assert probe_protocol(http).protocol == "kserve"


def test_auto_detects_mediapipe_rejection_and_picks_kserve():
    """2026.3 routes /v1/models/{m}:predict to the MediaPipe graph handler."""
    http = _ScriptedHttp(
        {
            "/v1/config": (200, {"model_config_list": []}),
            "/v2/health/ready": (200, {}),
        },
        post_result=(412, "The file is not valid json - model field is missing in JSON body"),
    )
    decision = probe_protocol(http)
    assert decision.protocol == "kserve"
    assert decision.evidence["tfs_predict_probe_mediapipe_rejection"] is True


def test_auto_detects_mediapipe_graph_404_and_picks_kserve():
    http = _ScriptedHttp(
        {
            "/v1/config": (200, {"model_config_list": []}),
            "/v2/health/ready": (200, {}),
        },
        post_result=(404, "Mediapipe graph definition with requested name is not found"),
    )
    assert probe_protocol(http).protocol == "kserve"


def test_auto_keeps_tfs_when_both_apis_answer_normally():
    http = _ScriptedHttp(
        {
            "/v1/config": (200, {"model_config_list": []}),
            "/v2/health/ready": (200, {}),
        },
        post_result=(400, "Model with name __ovms_protocol_probe__ does not exist"),
    )
    assert probe_protocol(http).protocol == "tfs"


def test_auto_fails_closed_when_no_api_answers(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    http = _ScriptedHttp({})
    with pytest.raises(ProtocolDetectionError):
        probe_protocol(http)

    with pytest.raises(HTTPException) as exc:
        resolve_adapter(http)
    assert exc.value.status_code == 502
    assert "auto-detection failed" in exc.value.detail


def test_auto_detection_runs_once_and_is_cached(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    http = _ScriptedHttp(
        {
            "/v1/config": (404, None),
            "/v2/health/ready": (200, {}),
        }
    )
    assert resolve_adapter(http).name == "kserve"
    first_round = list(http.calls)
    assert resolve_adapter(http).name == "kserve"
    assert http.calls == first_round, "auto detection must not re-probe per inference"


def test_auto_detection_is_thread_safe_and_probes_once(monkeypatch):
    import threading

    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    http = _ScriptedHttp(
        {
            "/v1/config": (404, None),
            "/v2/health/ready": (200, {}),
        }
    )
    results = []

    def worker():
        results.append(resolve_adapter(http).name)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["kserve"] * 8
    assert sum(1 for method, _ in http.calls if method == "GET") == 2


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def test_diagnostics_report_configured_protocol(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "tfs")
    diagnostics = protocol_diagnostics()
    assert diagnostics["ovms_protocol_configured"] == "tfs"
    assert diagnostics["ovms_protocol_effective"] == "tfs"
    assert diagnostics["ovms_protocol_source"] == "configured"


def test_diagnostics_report_unresolved_auto_before_first_use(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    diagnostics = protocol_diagnostics()
    assert diagnostics["ovms_protocol_configured"] == "auto"
    assert diagnostics["ovms_protocol_effective"] == "unresolved"
    assert diagnostics["ovms_protocol_source"] == "auto_pending"


def test_diagnostics_report_effective_protocol_after_detection(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    resolve_adapter(
        _ScriptedHttp({"/v1/config": (404, None), "/v2/health/ready": (200, {})})
    )
    diagnostics = protocol_diagnostics()
    assert diagnostics["ovms_protocol_effective"] == "kserve"
    assert diagnostics["ovms_protocol_source"] == "auto_probe"


# --------------------------------------------------------------------------- #
# Protocol-aware model availability (DeviceBroker dependency)
# --------------------------------------------------------------------------- #

def test_tfs_availability_uses_model_version_status():
    adapter = TfsAdapter()
    assert adapter.model_available(
        "m", _ScriptedHttp({"/v1/models/m": (200, {"model_version_status": [{"state": "AVAILABLE"}]})})
    )
    assert not adapter.model_available(
        "m", _ScriptedHttp({"/v1/models/m": (200, {"model_version_status": [{"state": "LOADING"}]})})
    )
    assert not adapter.model_available("m", _ScriptedHttp({"/v1/models/m": (404, None)}))


def test_kserve_availability_uses_the_ready_endpoint():
    adapter = KserveAdapter()
    assert adapter.model_available("m", _ScriptedHttp({"/v2/models/m/ready": (200, {})}))
    assert not adapter.model_available("m", _ScriptedHttp({"/v2/models/m/ready": (503, {})}))


def test_kserve_availability_falls_back_to_model_metadata():
    adapter = KserveAdapter()
    http = _ScriptedHttp({"/v2/models/m/ready": (404, None), "/v2/models/m": (200, {})})
    assert adapter.model_available("m", http)


def test_is_model_available_returns_false_when_auto_detection_fails(monkeypatch):
    monkeypatch.setattr(protocol_mod, "OVMS_PROTOCOL", "auto")
    monkeypatch.setattr(
        ovms_client.HTTP_CLIENT,
        "get",
        lambda path, timeout: (None, None),
    )
    assert ovms_client.is_model_available("m") is False
