from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import HTTPException
from ..config import OVMS_TIMEOUT
from .. import ovms_client
from .multimodal import load_image_from_input, preprocess_image_tensor
from ..registry import MODEL_REGISTRY


# Standard DINO ImageNet preprocessor parameters
DINO_DEFAULT_PREPROCESSOR = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "resize": 224,
    "resample": "bicubic",
}


def run_dino_embedding(
    model_name: str,
    images: List[Any],
    model_cfg: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    """Encodes images via DINO vision transformer backbone (image-to-image only).

    Respects declared variant, input size, ImageNet preprocessing, and expected output dimension.
    """
    base_name = model_name.split("__")[0]
    cfg = model_cfg or MODEL_REGISTRY.get(model_name) or MODEL_REGISTRY.get(base_name) or {}

    prep_cfg = dict(DINO_DEFAULT_PREPROCESSOR)
    if "preprocessor" in cfg and isinstance(cfg["preprocessor"], dict):
        prep_cfg.update(cfg["preprocessor"])

    target_size = cfg.get("input_size", [3, 224, 224])
    resize_val = target_size[-1] if isinstance(target_size, list) else 224

    tensors = []
    for item in images:
        pil_img = load_image_from_input(item)
        tensor = preprocess_image_tensor(pil_img, preprocessor_cfg=prep_cfg, target_size=resize_val)
        tensors.append(tensor)

    batch = np.stack(tensors, axis=0)  # [B, 3, H, W]
    payload = {
        "instances": [
            {"pixel_values": batch[i].tolist()}
            for i in range(batch.shape[0])
        ]
    }
    response = ovms_client.ovms_predict(model_name, payload, timeout=OVMS_TIMEOUT)
    predictions = response.get("predictions", [])
    if not predictions:
        raise HTTPException(status_code=502, detail="empty DINO prediction")

    if isinstance(predictions[0], dict):
        key = "pooler_output" if "pooler_output" in predictions[0] else list(predictions[0].keys())[0]
        vectors = [row.get(key, []) for row in predictions]
    else:
        vectors = predictions

    out = []
    for v in vectors:
        vec = np.asarray(v, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        out.append(vec.tolist())
    return out
