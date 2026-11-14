"""Query-time embeddings from the models that need PyTorch, served over HTTP so the API image stays light.

Run: uvicorn vrag.embedder:app --host 0.0.0.0 --port 8001
"""

import logging
import threading

from fastapi import FastAPI
from pydantic import BaseModel, Field

from vrag import __version__, vision
from vrag.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())

app = FastAPI(title="Multimodal Video RAG Engine: embedder", version=__version__)
_model_lock = threading.Lock()


class Texts(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class Vectors(BaseModel):
    model: str
    vectors: list[list[float]]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "frame_model": settings.frame_embed_model, "device": vision.device()}


@app.post("/embed/frame-text")
def frame_text(body: Texts) -> Vectors:
    """Embed text queries into the space of the frame vectors. The first call loads the model."""
    with _model_lock:
        vectors = vision.embed_texts(body.texts)
    return Vectors(model=settings.frame_embed_model, vectors=vectors.tolist())
