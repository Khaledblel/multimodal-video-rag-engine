import io
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image
from starlette.requests import ClientDisconnect

from vrag import __version__, answer, index, jobs, ollama, planner
from vrag.artifacts import artifact_dir, list_videos
from vrag.config import get_settings
from vrag.schemas.answer import AnswerResponse
from vrag.schemas.jobs import Job
from vrag.schemas.search import SearchResponse
from vrag.schemas.timeline import SCHEMA_VERSION
from vrag.schemas.videos import VideoSummary

VIDEO_ID = r"^[0-9a-f]{16}$"

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
    plan: bool | None = None,
) -> SearchResponse:
    """Search every indexed video, or one. `plan` turns the query planner on or off for this request."""
    if mode == "frame" and not settings.frame_embed_enabled:
        raise HTTPException(status_code=400, detail="frame search is turned off (FRAME_EMBED_ENABLED=false)")
    # A planner that fails does not fail the search: the query then passes through as it is.
    made = planner.plan(q, video_id, plan)
    try:
        hits = index.search(made.text, limit, made.video_id, mode)
    except httpx.HTTPError as exc:
        # Frame search embeds the query in the embedder service.
        raise HTTPException(status_code=503, detail=f"embedder unavailable: {exc}") from exc
    except ollama.OllamaError as exc:
        # Text search embeds the query in Ollama.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchResponse(query=q, collection=index.collection_name(), mode=mode, plan=made, hits=hits)

@app.get("/ask", response_model=None)
def ask(
    q: Annotated[str, Query(min_length=1, max_length=500)],
    video_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=10)] = 5,
    frames: Annotated[int, Query(ge=0, le=5)] = 3,
    mode: Literal["fused", "text", "frame"] = "fused",
    plan: bool | None = None,
    stream: bool = False,
) -> AnswerResponse | StreamingResponse:
    """Answer a question from the moments a search finds. `limit` is the number of hits, `frames` the frames per hit.

    `plan` turns the query planner on or off for this request. With stream=true the reply is newline-delimited
    JSON: a "moments" event, "token" events, then "done" or "error".
    """
    if not settings.vlm_enabled:
        raise HTTPException(status_code=400, detail="answers are turned off (VLM_ENABLED=false)")
    if mode == "frame" and not settings.frame_embed_enabled:
        raise HTTPException(status_code=400, detail="frame search is turned off (FRAME_EMBED_ENABLED=false)")
    # Searching and building the prompt come before the reply starts, so their failures still get a status code.
    try:
        prepared = answer.prepare(q, limit, video_id, mode, frames, plan)
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


@app.get("/videos")
def videos() -> list[VideoSummary]:
    """Every ingested video, by name."""
    current = index.collection_name()
    return [
        VideoSummary(
            video_id=video.video_id,
            source_name=video.source_name,
            duration_s=video.duration_s,
            indexed=video.collection == current,
            playable=(artifact_dir(video.video_id) / "proxy.mp4").is_file(),
        )
        for video in list_videos()
    ]


def _safe_name(name: str) -> str:
    """A file name without folders or odd characters, so that an upload can only land in the media folder."""
    last = name.replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", last).lstrip(".")[-120:] or "video"


def _free_path(path: Path) -> Path:
    """The path itself, or with -1, -2 and so on before its extension when that name is taken."""
    candidate, n = path, 0
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
    return candidate


@app.put("/videos")
async def upload(request: Request, name: Annotated[str, Query(min_length=1, max_length=200)]) -> Job:
    """Store a video and queue its ingest. The request body is the file itself; `name` is its file name.

    The body is written to data/media/ as it arrives, so a large file takes no memory and no second copy.
    """
    media_dir = settings.data_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    target = _free_path(media_dir / _safe_name(name))
    part = target.with_name(target.name + ".part")
    size = 0
    try:
        with part.open("wb") as handle:
            async for chunk in request.stream():
                handle.write(chunk)
                size += len(chunk)
    except ClientDisconnect:
        part.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="the upload was cut off") from None
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise HTTPException(status_code=507, detail=f"the upload could not be stored: {exc}") from exc
    if not size:
        part.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="the upload is empty")
    part.replace(target)
    return jobs.create(target, name)


@app.get("/jobs")
def list_jobs() -> list[Job]:
    """The latest ingest jobs, newest first. The worker runs them one at a time."""
    return jobs.latest()


@app.get("/jobs/{job_id}")
def job(job_id: str) -> Job:
    found = jobs.load(job_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such job")
    return found


@app.get("/videos/{video_id}/video")
def video(video_id: Annotated[str, PathParam(pattern=VIDEO_ID)]) -> FileResponse:
    """The copy of a video that ingest makes for browsers. A player seeks in it with Range requests."""
    path = artifact_dir(video_id) / "proxy.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="this video has no copy for the browser")
    return FileResponse(path, media_type="video/mp4")


@app.get("/videos/{video_id}/frames/{frame_index}")
def frame(
    video_id: Annotated[str, PathParam(pattern=VIDEO_ID)],
    frame_index: Annotated[int, PathParam(ge=0)],
    w: Annotated[int, Query(ge=32, le=1920)] = 320,
) -> Response:
    """One of a video's frames as a JPEG, at most `w` pixels wide."""
    path = artifact_dir(video_id) / "frames" / f"{frame_index:06d}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such frame")
    with Image.open(path) as image:
        image.draft("RGB", (w, w))
        small = image.convert("RGB")
    small.thumbnail((w, 10 * w))
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=80)
    return Response(buffer.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
