"""Tests for the Gate 2 / Gate 3 acceptance harness.

The harness in `scripts/ovms_acceptance_suite.py` is what turns raw
measurements into the PASS/FAIL verdict table that justifies (or blocks) an
OVMS 2026.3.1 promotion.  If its gate logic is wrong, a red run can be reported
as green -- so the logic is tested here, offline, against synthetic results.

Nothing in this module touches the network.
"""

from __future__ import annotations

import json
import requests
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import ovms_acceptance_suite as suite  # noqa: E402
import ovms_protocol_capability_probe as probe  # noqa: E402
import ovms_v3_embeddings_probe as v3  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _embedding(http: int = 200, dim: int = 1024, norm: float = 1.0) -> dict:
    return {
        "http": http,
        "seconds": 0.1,
        "dim": dim,
        "norm": norm,
        "prompt_tokens": 12,
        "error": None if http == 200 and dim else "backend error",
    }


@pytest.fixture
def green_results() -> dict:
    """A fully passing acceptance run, as the harness would emit it."""
    return {
        "label": "synthetic-green",
        "protocol": "kserve",
        "gateway_base": "http://127.0.0.1:28012",
        "ovms_base": "http://127.0.0.1:28342",
        "captured_at": "2026-09-17T00:00:00+00:00",
        "model_matrix": [
            {"model": "qwen3-embedding-0.6b-int4", "kind": "embedding", "required": True,
             "status": "PASS", **_embedding()},
            {"model": "qwen-reranker", "kind": "rerank", "required": True, "status": "PASS",
             "http": 200, "seconds": 0.4, "dim": None, "norm": None, "error": None},
            {"model": "DINO", "kind": "image_embedding", "required": False, "status": "N/A",
             "note": "no production OVMS deployment"},
        ],
        "long_text": [
            {
                "target_tokens": 4591,
                "actual_tokens": 4591,
                "single": _embedding(),
                "single_liveness": _embedding(),
                "concurrency": 2,
                "concurrent": [_embedding(), _embedding()],
                "wall_seconds": 21.11,
                "concurrent_liveness": _embedding(),
            }
        ],
        "reranker": {
            "http": 200, "seconds": 0.4, "scores": {}, "ordering_ok": True, "range_ok": True,
            "biology_score": 0.92, "eiffel_score": 0.03, "error": None,
        },
        "zero_resident": {
            "initial_loaded": [], "after_request": ["qwen3-embedding-0.6b-int4__gpu"],
            "unloaded": True, "unload_seconds": 300.0, "error": None,
        },
        "counters": {"http_5xx": 0, "exceptions": 0},
    }


# --------------------------------------------------------------------------- #
# Gate evaluation
# --------------------------------------------------------------------------- #

def test_all_green_run_passes_every_gate(green_results):
    gates = suite.evaluate_gates(green_results)
    assert gates == {
        "Compatibility": "PASS",
        "Correctness": "PASS",
        "Long-text liveness": "PASS",
        "Concurrency": "PASS",
        "Reranker": "PASS",
        "Zero-resident": "PASS",
        "Failure counters": "PASS",
    }


def test_absent_measurements_are_not_applicable_not_pass():
    """A section that never ran must not be reported as a pass."""
    gates = suite.evaluate_gates({})
    assert gates["Compatibility"] == "N/A"
    assert gates["Correctness"] == "N/A"
    assert gates["Long-text liveness"] == "N/A"
    assert gates["Concurrency"] == "N/A"
    assert gates["Reranker"] == "N/A"
    assert gates["Zero-resident"] == "N/A"
    # Counters are only meaningful once something ran, but zero failures on an
    # empty run is genuinely zero failures.
    assert gates["Failure counters"] == "PASS"


def test_required_model_failure_fails_compatibility(green_results):
    green_results["model_matrix"][0]["status"] = "FAIL"
    assert suite.evaluate_gates(green_results)["Compatibility"] == "FAIL"


def test_optional_model_does_not_affect_compatibility(green_results):
    """DINO has no production deployment, so its N/A row must not gate."""
    assert suite.evaluate_gates(green_results)["Compatibility"] == "PASS"
    dino = next(row for row in green_results["model_matrix"] if row["model"] == "DINO")
    assert dino["required"] is False
    assert dino["status"] == "N/A"


def test_norm_outside_tolerance_fails_correctness(green_results):
    green_results["long_text"][0]["single"] = _embedding(norm=0.5)
    assert suite.evaluate_gates(green_results)["Correctness"] == "FAIL"


def test_zero_dimension_fails_correctness(green_results):
    green_results["model_matrix"][0] = {
        "model": "qwen3-embedding-0.6b-int4", "kind": "embedding", "required": True,
        "status": "PASS", **_embedding(dim=0, norm=None),
    }
    assert suite.evaluate_gates(green_results)["Correctness"] == "FAIL"


def test_liveness_failure_fails_the_liveness_gate(green_results):
    green_results["long_text"][0]["concurrent_liveness"] = _embedding(http=502, dim=0, norm=None)
    gates = suite.evaluate_gates(green_results)
    assert gates["Long-text liveness"] == "FAIL"
    # The long round itself still succeeded, so concurrency is unaffected.
    assert gates["Concurrency"] == "PASS"


def test_partial_concurrency_fails_the_concurrency_gate(green_results):
    green_results["long_text"][0]["concurrent"] = [_embedding()]
    assert suite.evaluate_gates(green_results)["Concurrency"] == "FAIL"


def test_concurrency_backend_error_fails_the_concurrency_gate(green_results):
    green_results["long_text"][0]["concurrent"] = [_embedding(), _embedding(http=504, dim=0, norm=None)]
    assert suite.evaluate_gates(green_results)["Concurrency"] == "FAIL"


def test_reranker_ordering_is_required(green_results):
    green_results["reranker"]["ordering_ok"] = False
    assert suite.evaluate_gates(green_results)["Reranker"] == "FAIL"


def test_reranker_probability_range_is_required(green_results):
    green_results["reranker"]["range_ok"] = False
    assert suite.evaluate_gates(green_results)["Reranker"] == "FAIL"


def test_zero_resident_requires_observation_of_a_load(green_results):
    green_results["zero_resident"]["after_request"] = []
    assert suite.evaluate_gates(green_results)["Zero-resident"] == "FAIL"


def test_zero_resident_requires_observation_of_an_unload(green_results):
    green_results["zero_resident"]["unloaded"] = False
    assert suite.evaluate_gates(green_results)["Zero-resident"] == "FAIL"


def test_failure_counters_gate_on_both_5xx_and_exceptions(green_results):
    green_results["counters"]["http_5xx"] = 1
    assert suite.evaluate_gates(green_results)["Failure counters"] == "FAIL"
    green_results["counters"] = {"http_5xx": 0, "exceptions": 3}
    assert suite.evaluate_gates(green_results)["Failure counters"] == "FAIL"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def test_report_renders_the_verdict_table(green_results):
    report = suite.render_report(green_results)
    assert "## Acceptance run — synthetic-green" in report
    assert "| Compatibility | **PASS** |" in report
    assert "| Failure counters | **PASS** |" in report
    assert "### Long text" in report
    assert "### Reranker" in report
    assert "### Zero-resident" in report


def test_report_renders_missing_measurements_as_dashes(green_results):
    green_results["long_text"][0]["single"] = _embedding(http=None, dim=0, norm=None)
    green_results["long_text"][0]["actual_tokens"] = None
    report = suite.render_report(green_results)
    assert "| 4591 | - | - / 0.1s |" in report
    assert "None" not in report.split("### Capability")[0]


def test_report_embeds_the_capability_capture(green_results):
    green_results["capability"] = _capture()
    report = suite.render_report(green_results)
    assert "### Capability capture — ovms-2026.3.1" in report
    assert "would resolve to **kserve**" in report


# --------------------------------------------------------------------------- #
# Capability capture
# --------------------------------------------------------------------------- #

def _capture() -> dict:
    return {
        "label": "ovms-2026.3.1",
        "captured_at": "2026-09-17T00:00:00+00:00",
        "ovms_base": "http://127.0.0.1:28342",
        "endpoints": [
            {"method": "GET", "path": "/v2/health/ready", "status": 200, "response": {"status": "ok"},
             "error": None},
            {"method": "POST", "path": "/v1/models/qwen-reranker:predict", "status": None,
             "response": None, "error": "ConnectError: refused"},
        ],
        "auto_decision": {"protocol": "kserve", "source": "kserve_only",
                          "evidence": {"kserve_health_ready_status": 200}},
        "auto_decision_error": None,
        "inference_probe": None,
    }


def test_endpoint_plan_covers_every_auto_ladder_signal():
    plan = probe.build_endpoint_plan("qwen-reranker")
    paths = {entry["path"] for entry in plan}
    # auto signal 1, auto signal 2, auto signal 3 (the MediaPipe tie-breaker).
    assert "/v2/health/ready" in paths
    assert "/v1/config" in paths
    assert "/v1/models/qwen-reranker:predict" in paths
    assert "/v2/models/qwen-reranker/infer" in paths
    assert "/v2/models/qwen-reranker/ready" in paths
    assert "/v3/embeddings" in paths


def test_endpoint_plan_without_a_model_only_probes_server_level_surfaces():
    plan = probe.build_endpoint_plan(None)
    assert len(plan) == 5
    assert all("{model}" not in entry["path"] for entry in plan)


def test_capability_markdown_reports_transport_errors_not_blank_rows():
    markdown = probe.render_markdown(_capture())
    assert "transport error" in markdown
    assert "ConnectError: refused" in markdown


def test_capability_markdown_reports_a_fail_closed_auto_decision():
    capture = _capture()
    capture["auto_decision"] = None
    capture["auto_decision_error"] = "neither the TFS config API nor the KServe v2 health API responded"
    markdown = probe.render_markdown(capture)
    assert "failed closed" in markdown
    assert "would resolve to" not in markdown


def test_capability_markdown_is_json_serialisable_for_the_artefact():
    capture = _capture()
    capture["label"] = None
    # render_markdown falls back to the base URL when no label was given.
    assert "### Capability capture — http://127.0.0.1:28342" in probe.render_markdown(capture)
    json.dumps(capture)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

def test_proxy_environment_is_ignored_by_default():
    """Loopback acceptance runs must not be routed through an ambient proxy."""
    suite.set_env_proxy(False)
    assert suite._session().trust_env is False
    probe.set_env_proxy(False)
    assert probe._session().trust_env is False


def test_env_proxy_can_be_opted_into_and_back_out_of():
    try:
        suite.set_env_proxy(True)
        assert suite._session().trust_env is True
        suite.set_env_proxy(False)
        assert suite._session().trust_env is False
    finally:
        suite.set_env_proxy(False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"data": [{"embedding": [0.1, 0.2]}]}, [0.1, 0.2]),
        ({"data": []}, []),
        ({"data": "nope"}, []),
        ("not a dict", []),
        (None, []),
    ],
)
def test_first_vector_extraction(payload, expected):
    assert suite._first_vector(payload) == expected


def test_vector_stats_reports_dimension_and_l2_norm():
    assert suite._vector_stats([0.0, 0.0]) == (2, 0.0)
    assert suite._vector_stats([3.0, 4.0]) == (2, 5.0)
    assert suite._vector_stats([]) == (0, None)


def test_error_summaries_are_truncated():
    assert len(suite._summarise_error({"detail": "x" * 500})) == 300
    assert suite._summarise_error({"detail": "short"}) == "short"
    assert suite._summarise_error(None) is None


def test_calibrate_text_converges_on_the_requested_token_count(monkeypatch):
    """The harness measures tokens via `usage.prompt_tokens`, it does not guess."""
    calls = []

    def fake_embed(gateway_base, model, text, timeout, counters):
        calls.append(len(text))
        return {"http": 200, "seconds": 0.0, "dim": 1024, "norm": 1.0,
                "prompt_tokens": len(text), "error": None}

    monkeypatch.setattr(suite, "embed", fake_embed)
    text, actual = suite.calibrate_text("http://gw", "m", 4096, 10.0, suite.Counters())
    assert actual == pytest.approx(4096, rel=0.05)
    assert len(text) == actual
    assert len(calls) <= 3


def test_calibrate_text_gives_up_when_the_gateway_is_unreachable(monkeypatch):
    def dead_embed(gateway_base, model, text, timeout, counters):
        counters.note_exception("POST", f"{gateway_base}/v1/embeddings", ConnectionError("refused"))
        return {"http": None, "seconds": 0.0, "dim": 0, "norm": None,
                "prompt_tokens": None, "error": "ConnectError"}

    monkeypatch.setattr(suite, "embed", dead_embed)
    counters = suite.Counters()
    text, actual = suite.calibrate_text("http://gw", "m", 4096, 10.0, counters)
    assert actual is None
    assert text
    assert counters.exceptions == 1
    # A bare count is not actionable; the failure must name its target.
    assert counters.log() == [
        {
            "method": "POST",
            "url": "http://gw/v1/embeddings",
            "type": "ConnectionError",
            "message": "refused",
        }
    ]


def test_counters_track_5xx_and_transport_failures():
    counters = suite.Counters()
    counters.note_status(200)
    counters.note_status(502)
    counters.note_status(None)
    counters.note_exception("GET", "http://gw/v1/broker/metrics", TimeoutError("timed out"))
    assert counters.as_dict() == {"http_5xx": 1, "exceptions": 1}
    assert counters.log()[0]["method"] == "GET"
    assert counters.log()[0]["type"] == "TimeoutError"


# --------------------------------------------------------------------------- #
# Keep-alive resets must not be reported as product failures
# --------------------------------------------------------------------------- #


class _ResetThenOkSession:
    """A session whose first GET raises the exact reset uvicorn produces."""

    def __init__(self, failures: int = 1):
        self.remaining = failures
        self.calls = 0
        self.trust_env = True

    def get(self, url, timeout=None):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise requests.exceptions.ConnectionError(
                "('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))"
            )
        return _FakeResponse(200, {"loaded_models": {}})

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        raise requests.exceptions.ConnectionError("('Connection aborted.', RemoteDisconnected(...))")


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_get_retries_a_reset_keep_alive_connection_and_does_not_fail_the_gate(monkeypatch):
    session = _ResetThenOkSession(failures=1)
    monkeypatch.setattr(suite, "_SESSION", session)
    counters = suite.Counters()

    status, payload, _ = suite._timed_get("http://gw/v1/broker/metrics", 5.0, counters)

    assert status == 200
    assert payload == {"loaded_models": {}}
    assert counters.exceptions == 0, "a retried keep-alive reset is not a product failure"
    assert counters.as_dict()["exceptions"] == 0
    assert [entry["method"] for entry in counters.retries()] == ["GET"]
    assert counters.retries()[0]["type"] == "ConnectionError"
    assert session.calls == 2


def test_get_still_fails_when_the_server_is_really_unreachable(monkeypatch):
    """The retry must not be able to mask a genuine outage."""
    session = _ResetThenOkSession(failures=99)
    monkeypatch.setattr(suite, "_SESSION", session)
    counters = suite.Counters()

    status, payload, _ = suite._timed_get("http://gw/v1/broker/metrics", 5.0, counters)

    assert status is None
    assert "ConnectionError" in str(payload)
    assert counters.exceptions == 1
    # The attempt is logged so the retry is visible, and the final outcome is
    # counted, so the gate still fails.
    assert [entry["type"] for entry in counters.retries()] == ["ConnectionError"]
    assert session.calls == 2


def test_post_is_never_retried(monkeypatch):
    """A POST that may already have been processed must not be duplicated."""
    session = _ResetThenOkSession(failures=99)
    monkeypatch.setattr(suite, "_SESSION", session)
    counters = suite.Counters()

    status, payload, _ = suite._timed_post("http://gw/v1/embeddings", {"model": "m"}, 5.0, counters)

    assert status is None
    assert counters.exceptions == 1
    assert counters.retries() == []
    assert session.calls == 1, "POST must be attempted exactly once"


# --------------------------------------------------------------------------- #
# Matrix definition
# --------------------------------------------------------------------------- #

def test_long_text_ladder_matches_the_task_book():
    assert suite.LONG_TEXT_LENGTHS == [256, 512, 1024, 4591]
    assert suite.LONG_TEXT_MODEL == "qwen3-embedding-0.6b-int4"


def test_matrix_requires_every_deployed_registry_model():
    required = [row["model"] for row in suite.MODEL_MATRIX if row["required"]]
    assert required == [
        "qwen3-embedding-0.6b-int4",
        "qwen3-embedding-0.6b-int8",
        "bge-m3-i8",
        "bge-m3",
        "arctic-embed-m-v2-int8",
        "qwen-reranker",
    ]


def test_matrix_marks_undeployed_models_optional_with_a_reason():
    optional = [row for row in suite.MODEL_MATRIX if not row["required"]]
    assert [row["model"] for row in optional] == ["DINO", "multimodal Classic IR"]
    assert all(row.get("note") for row in optional)


def test_reranker_fixture_matches_the_task_book_expectation():
    assert suite.RERANK_QUERY == "what is biology?"
    assert any("Biology is the study" in doc for doc in suite.RERANK_DOCS)
    assert any("Eiffel" in doc for doc in suite.RERANK_DOCS)


# --------------------------------------------------------------------------- #
# /v3/embeddings alias identification
# --------------------------------------------------------------------------- #


def test_v3_alias_resolves_through_the_registry_ovms_model_mapping():
    """The registry maps logical `qwen3-embedding-0.6b-int8` -> `qwen3-embedding-0.6b`.

    The runtime alias therefore shares no exact name with the logical id, which
    is exactly the case a naive exact-match lookup gets wrong.
    """
    aliases = ["bge-m3-i8__cpu", "qwen3-embedding-0.6b__cpu"]
    assert v3._pick_alias(aliases, "qwen3-embedding-0.6b-int8") == "qwen3-embedding-0.6b__cpu"


def test_v3_alias_prefers_the_alias_that_just_appeared():
    before = ["bge-m3-i8__cpu"]
    after = ["bge-m3-i8__cpu", "qwen3-embedding-0.6b__cpu"]
    picked = v3._pick_alias(after, "qwen3-embedding-0.6b-int8", exclude=tuple(before))
    assert picked == "qwen3-embedding-0.6b__cpu"


def test_v3_alias_prefers_gpu_when_both_devices_are_resident():
    aliases = ["qwen3-embedding-0.6b__cpu", "qwen3-embedding-0.6b__gpu"]
    assert v3._pick_alias(aliases, "qwen3-embedding-0.6b-int8") == "qwen3-embedding-0.6b__gpu"


def test_v3_alias_returns_none_for_an_unrelated_model():
    assert v3._pick_alias(["qwen-reranker__cpu"], "qwen3-embedding-0.6b-int8") is None


def test_v3_probe_refuses_to_run_without_the_inference_flag():
    """A successful POST /v3/embeddings is a real inference; opt-in is mandatory."""
    with pytest.raises(SystemExit):
        v3.run_probe(
            gateway_base="http://gw",
            ovms_base="http://ovms",
            gateway_model="qwen3-embedding-0.6b-int8",
            timeout=1.0,
            allow_inference_probes=False,
        )


def test_v3_cold_warm_reports_an_honest_error_when_the_alias_never_unloads(monkeypatch):
    monkeypatch.setattr(
        v3, "_wait_until_not_resident", lambda *a, **k: {"unloaded": False, "waited_seconds": None}
    )
    result = v3._gateway_cold_warm("http://gw", "m", "m__cpu", 1.0)
    assert result["cold"] is None
    assert result["warm"] is None
    assert "still resident" in result["error"]


def test_v3_cold_warm_measures_a_cold_then_a_warm_call(monkeypatch):
    monkeypatch.setattr(
        v3, "_wait_until_not_resident", lambda *a, **k: {"unloaded": True, "waited_seconds": 1.5}
    )
    calls = []

    def fake_post(url, payload, timeout):
        calls.append(url)
        return {
            "http": 200,
            "seconds": 0.5,
            "transport_error": None,
            "body": {"data": [{"embedding": [1.0, 0.0]}]},
        }

    monkeypatch.setattr(v3, "_post", fake_post)
    result = v3._gateway_cold_warm("http://gw", "m", "m__cpu", 1.0)

    assert result["cold"]["kind"] == "cold"
    assert result["cold"]["resident_before_call"] is False
    assert result["warm"]["kind"] == "warm"
    assert result["warm"]["resident_before_call"] is True
    assert result["cold"]["dim"] == 2
    assert result["cold"]["all_finite"] is True
    assert result["cold"]["norm"] == 1.0
    assert len(calls) == 2
