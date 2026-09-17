from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from fastapi import HTTPException
from ..config import OVMS_TIMEOUT
from ..ovms_client import ovms_predict
from ..tokenizers import get_embedding_tokenizer
from ..registry import MODEL_REGISTRY


def load_image_from_input(image_input: Union[str, bytes, Image.Image]) -> Image.Image:
    """Loads a PIL Image from PIL Image, base64 string, data URI, or file path."""
    try:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                _, b64data = image_input.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            try:
                img_bytes = base64.b64decode(image_input)
                if len(img_bytes) > 32:
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                pass
            return Image.open(image_input).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to load image: {exc}") from exc
    raise HTTPException(status_code=400, detail="unsupported image input format")


def preprocess_image_tensor(
    img: Image.Image,
    preprocessor_cfg: Optional[Dict[str, Any]] = None,
    target_size: Optional[int] = None,
) -> np.ndarray:
    """Resizes and normalizes an image for vision backbone (NCHW format).

    Supports:
    - shortest edge resize with center crop (e.g. Jina-CLIP-v2: shortest edge to 512, then center crop to 512x512)
    - direct resize (e.g. DINO: 224x224)
    """
    cfg = preprocessor_cfg or {}
    resize_mode = cfg.get("resize_mode", "direct")
    target_dim = target_size or cfg.get("size") or cfg.get("resize") or 512
    if isinstance(target_dim, (list, tuple)):
        target_w, target_h = int(target_dim[-1]), int(target_dim[-2])
    else:
        target_w, target_h = int(target_dim), int(target_dim)

    resample_name = str(cfg.get("interpolation") or cfg.get("resample", "bicubic")).lower()
    resample_filter = Image.Resampling.BICUBIC if resample_name == "bicubic" else Image.Resampling.BILINEAR

    if resize_mode == "shortest":
        w, h = img.size
        short_edge = min(w, h)
        scale = target_w / max(1, short_edge)
        new_w, new_h = max(target_w, int(round(w * scale))), max(target_h, int(round(h * scale)))
        img = img.resize((new_w, new_h), resample_filter)
        # Center crop to target_w x target_h
        crop_size = cfg.get("crop_size", target_w)
        crop_w = crop_size if isinstance(crop_size, int) else crop_size[-1]
        crop_h = crop_size if isinstance(crop_size, int) else crop_size[-2]
        left = max(0, (new_w - crop_w) // 2)
        top = max(0, (new_h - crop_h) // 2)
        img = img.crop((left, top, left + crop_w, top + crop_h))
    else:
        img = img.resize((target_w, target_h), resample_filter)

    arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]

    # Default to CLIP mean/std if not specified in preprocessor config
    mean_val = cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
    std_val = cfg.get("std", [0.26862954, 0.26130258, 0.27577711])

    mean = np.array(mean_val, dtype=np.float32)
    std = np.array(std_val, dtype=np.float32)

    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def run_multimodal_image_embedding(
    model_name: str,
    images: List[Any],
    model_cfg: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    """Encodes images via OVMS vision model with declarative preprocessing and Matryoshka truncation."""
    base_name = model_name.split("__")[0]
    cfg = model_cfg or MODEL_REGISTRY.get(model_name) or MODEL_REGISTRY.get(base_name) or {}

    prep_cfg = cfg.get("preprocessor", {})
    truncate_dim = cfg.get("truncate_dimension")

    tensors = []
    for item in images:
        pil_img = load_image_from_input(item)
        tensor = preprocess_image_tensor(pil_img, preprocessor_cfg=prep_cfg)
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
        # Apply Matryoshka dimension truncation if declared in model config
        if truncate_dim and len(vec) > truncate_dim:
            vec = vec[:truncate_dim]
        # L2-normalize the (potentially truncated) embedding vector
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        out.append(vec.tolist())
    return out


def run_multimodal_text_embedding(
    model_name: str,
    texts: List[str],
    tokenizer_path: Optional[str] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[List[float]], int]:
    """Encodes text via OVMS multimodal text tower with matching Matryoshka truncation."""
    base_name = model_name.split("__")[0]
    cfg = model_cfg or MODEL_REGISTRY.get(model_name) or MODEL_REGISTRY.get(base_name) or {}

    tok_path = tokenizer_path or cfg.get("tokenizer_path") or model_name
    truncate_dim = cfg.get("truncate_dimension")

    tokenizer = get_embedding_tokenizer(tok_path)
    max_len = int(cfg.get("max_length", 8192))
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="np")
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
        # Apply identical Matryoshka dimension truncation for text
        if truncate_dim and len(vec) > truncate_dim:
            vec = vec[:truncate_dim]
        # L2-normalize
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        out.append(vec.tolist())
    return out, prompt_tokens
