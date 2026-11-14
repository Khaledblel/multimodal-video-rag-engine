"""SigLIP 2 embeddings of video frames and of short text queries, in one shared space.

Needs PyTorch, so it is imported only in the worker image: by ingest and by the embedder service.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from vrag.config import get_settings

BATCH = 16
TEXT_TOKENS = 64


def device() -> str:
    wanted = get_settings().embed_device
    if wanted != "auto":
        return wanted
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache
def _load() -> tuple:
    name = get_settings().frame_embed_model
    model = AutoModel.from_pretrained(name).eval().to(device())
    if model.device.type == "cuda":
        model = model.half()
    return model, AutoTokenizer.from_pretrained(name)


def _unit(output: object) -> np.ndarray:
    features = output if isinstance(output, torch.Tensor) else output.pooler_output
    return torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()


def _pixels(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0


def embed_frames(paths: list[Path]) -> np.ndarray:
    """One unit-length row per image file."""
    model, _ = _load()
    size = model.config.vision_config.image_size
    rows = []
    for start in range(0, len(paths), BATCH):
        batch = np.stack([_pixels(path, size) for path in paths[start : start + BATCH]])
        pixels = torch.from_numpy(batch).to(model.device, model.dtype)
        with torch.inference_mode():
            rows.append(_unit(model.get_image_features(pixel_values=pixels)))
    if not rows:
        return np.empty((0, get_settings().frame_embed_dim), dtype=np.float32)
    return np.concatenate(rows)


def embed_texts(texts: list[str]) -> np.ndarray:
    """One unit-length row per text, comparable with the rows of embed_frames."""
    model, tokenizer = _load()
    tokens = tokenizer(
        [text.lower() for text in texts],
        padding="max_length",
        max_length=TEXT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        return _unit(model.get_text_features(input_ids=tokens["input_ids"].to(model.device)))
