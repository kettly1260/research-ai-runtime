import base64
import io
from typing import Any, List, Tuple, Union
import numpy as np
from PIL import Image
from fastapi import HTTPException
from ..config import OVMS_TIMEOUT
from ..ovms_client import ovms_predict
from ..tokenizers import get_embedding_tokenizer


def load_image_from_input(image_input: Union[str, bytes]) -> Image.Image:
    """Loads a PIL Image from base64 string, data URI, or file path."""
    try:
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                _, b64data = image_input.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Try raw base64
            try:
                img_bytes = base64.b64decode(image_input)
                if len(img_bytes) > 32:
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                pass
            # Try file path
            return Image.open(image_input).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to load image: {exc}") from exc
    raise HTTPException(status_code=400, detail="unsupported image input format")


def preprocess_image_tensor(img: Image.Image, target_size: int = 224) -> np.ndarray:
    """Resizes and normalizes an image for vision backbone (NCHW format)."""
    img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def run_multimodal_image_embedding(model_name: str, images: List[Any]) -> List[List[float]]:
    """Encodes images via OVMS vision model."""
    tensors = []
    for item in images:
        pil_img = load_image_from_input(item)
        tensor = preprocess_image_tensor(pil_img)
        tensors.append(tensor)

    batch = np.stack(tensors, axis=0)  # [B, C, H, W]
    payload = {
        "instances": [
            {"pixel_values": batch[i].tolist()}
            for i in range(batch.shape[0])
        ]
    }
    response = ovms_predict(model_name, payload, timeout=OVMS_TIMEOUT)
    predictions = response.get("predictions", [])
    if not predictions:
        raise HTTPException(status_code=502, detail="empty multimodal vision prediction")

    if isinstance(predictions[0], dict):
        key = "image_embeds" if "image_embeds" in predictions[0] else list(predictions[0].keys())[0]
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


def run_multimodal_text_embedding(model_name: str, texts: List[str], tokenizer_path: str = None) -> Tuple[List[List[float]], int]:
    """Encodes text via OVMS multimodal text tower."""
    from typing import Tuple
    if not tokenizer_path:
        tokenizer_path = model_name
    tokenizer = get_embedding_tokenizer(tokenizer_path)
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    prompt_tokens = int(attention_mask.sum())

    payload = {
        "instances": [
            {
                "input_ids": input_ids[i].tolist(),
                "attention_mask": attention_mask[i].tolist(),
            }
            for i in range(input_ids.shape[0])
        ]
    }
    response = ovms_predict(model_name, payload, timeout=OVMS_TIMEOUT)
    predictions = response.get("predictions", [])
    if not predictions:
        raise HTTPException(status_code=502, detail="empty multimodal text prediction")

    if isinstance(predictions[0], dict):
        key = "text_embeds" if "text_embeds" in predictions[0] else list(predictions[0].keys())[0]
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
    return out, prompt_tokens
