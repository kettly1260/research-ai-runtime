import json
import os
import requests
from fastapi import HTTPException
from .config import OVMS_BASE, OVMS_TIMEOUT, OVMS_CONFIG_PATH, OVMS_CONFIG_RELOAD_URL, OVMS_MODEL_CATALOG_PATH


def ovms_url(model_name: str) -> str:
    return f"{OVMS_BASE.rstrip('/')}/v1/models/{model_name}:predict"


def ovms_predict(model_name: str, payload: dict, timeout: int = OVMS_TIMEOUT) -> dict:
    try:
        response = requests.post(ovms_url(model_name), json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        resp = exc.response
        status = resp.status_code if resp is not None else "unknown"
        body = ""
        if resp is not None:
            body = (resp.text or "").strip()
            if len(body) > 400:
                body = body[:400] + "..."
        raise HTTPException(
            status_code=502,
            detail=f"backend returned HTTP {status} for model {model_name}; body={body}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail=f"backend timeout after {timeout}s"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"backend request failed: {exc}"
        ) from exc
    return response.json()


def read_model_config_file(path: str) -> dict:
    if not os.path.exists(path):
        return {"model_config_list": []}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid OVMS config payload at {path}")
    if "model_config_list" not in payload:
        payload["model_config_list"] = []
    return payload


def catalog_model_config(model_name: str) -> dict:
    payload = read_model_config_file(OVMS_MODEL_CATALOG_PATH)
    for item in payload.get("model_config_list", []):
        cfg = item.get("config", {}) if isinstance(item, dict) else {}
        if cfg.get("name") == model_name:
            return cfg
    raise KeyError(f"model not found in catalog: {model_name}")


def write_runtime_config(payload: dict) -> None:
    directory = os.path.dirname(OVMS_CONFIG_PATH) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{OVMS_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, OVMS_CONFIG_PATH)


def reload_ovms_config() -> bool:
    url = OVMS_CONFIG_RELOAD_URL.format(ovms_base=OVMS_BASE.rstrip("/"))
    try:
        response = requests.post(url, timeout=max(OVMS_TIMEOUT, 30))
        return response.status_code in (200, 201, 202, 204)
    except requests.RequestException:
        return False


def is_model_available(model_name: str) -> bool:
    url = f"{OVMS_BASE.rstrip('/')}/v1/models/{model_name}"
    try:
        response = requests.get(url, timeout=min(OVMS_TIMEOUT, 15))
        if response.status_code != 200:
            return False
        payload = response.json()
    except (requests.RequestException, ValueError):
        return False

    statuses = payload.get("model_version_status", [])
    if not isinstance(statuses, list) or len(statuses) == 0:
        return True

    for status in statuses:
        state = str(status.get("state", "")).upper()
        if state == "AVAILABLE":
            return True
    return False
