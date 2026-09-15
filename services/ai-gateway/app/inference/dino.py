from typing import Any, List
import numpy as np
from fastapi import HTTPException
from ..config import OVMS_TIMEOUT
from .. import ovms_client
from .multimodal import load_image_from_input, preprocess_image_tensor


def run_dino_embedding(model_name: str, images: List[Any]) -> List[List[float]]:
    """Encodes images via DINO vision transformer backbone (image-to-image only)."""
    tensors = []
    for item in images:
        pil_img = load_image_from_input(item)
        # DINOv2 / DINOv3 standard resolution is typically 224x224
        tensor = preprocess_image_tensor(pil_img, target_size=224)
        tensors.append(tensor)

    batch = np.stack(tensors, axis=0)  # [B, 3, 224, 224]
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
