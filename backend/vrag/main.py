import json
import logging
import tempfile
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from vrag import __version__, answer, index, ollama
from vrag.config import get_settings
from vrag.schemas.answer import AnswerResponse
from vrag.schemas.search import SearchResponse
from vrag.schemas.timeline import SCHEMA_VERSION

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Multimodal Video RAG Engine", version=__version__)


async def _check_qdrant() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(settings.qdrant_url)
            response.raise_for_status()
            return {"ok": True, "version": response.json().get("version")}
    except (httpx.HTTPError, ValueError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _check_data_dir() -> dict:
    path = str(settings.data_dir)
    try:
        with tempfile.TemporaryFile(dir=settings.data_dir):
            pass
        return {"ok": True, "path": path}
    except OSError as exc:
        return {"ok": False, "path": path, "error": str(exc)}


@app.get("/health")
async def health() -> JSONResponse:
    qdrant = await _check_qdrant()
    data_dir = _check_data_dir()
    ok = qdrant["ok"] and data_dir["ok"]
    return JSONResponse(
        {
            "status": "ok" if ok else "degraded",
            "version": __version__,
            "timeline_schema_version": SCHEMA_VERSION,
            "qdrant": qdrant,
            "data_dir": data_dir,
        },
        status_code=200 if ok else 503,
    )


@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=1, max_length=500)],
    video_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 5,
    mode: Literal["fused", "text", "frame"] = "fused",
) -> SearchResponse:
    if mode == "frame" and not settings.frame_embed_enabled:
        raise HTTPException(status_code=400, detail="frame search is turned off (FRAME_EMBED_ENABLED=false)")
    try:
        hits = index.search(q, limit, video_id, mode)
    except httpx.HTTPError as exc:
        # Frame search embeds the query in the embedder service.
        raise HTTPException(status_code=503, detail=f"embedder unavailable: {exc}") from exc
    return SearchResponse(query=q, collection=index.collection_name(), mode=mode, hits=hits)

@app.get("/ask", response_model=None)
def ask(
    q: Annotated[str, Query(min_length=1, max_length=500)],
    video_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=10)] = 5,
    frames: Annotated[int, Query(ge=0, le=5)] = 3,
    mode: Literal["fused", "text", "frame"] = "fused",
    stream: bool = False,
) -> AnswerResponse | StreamingResponse:
    """Answer a question from the moments a search finds. `limit` is the number of hits, `frames` the frames per hit.

    With stream=true the reply is newline-delimited JSON: a "moments" event, "token" events, then "done" or "error".
    """
    if not settings.vlm_enabled:
        raise HTTPException(status_code=400, detail="answers are turned off (VLM_ENABLED=false)")
    if mode == "frame" and not settings.frame_embed_enabled:
        raise HTTPException(status_code=400, detail="frame search is turned off (FRAME_EMBED_ENABLED=false)")
    # Searching and building the prompt come before the reply starts, so their failures still get a status code.
    try:
        prepared = answer.prepare(q, limit, video_id, mode, frames)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"embedder unavailable: {exc}") from exc
    except ollama.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if stream:
        lines = (json.dumps(event, ensure_ascii=False) + "\n" for event in answer.generate(prepared))
        return StreamingResponse(lines, media_type="application/x-ndjson")
    try:
        return answer.collect(prepared)
    except ollama.OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
