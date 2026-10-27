import logging
import tempfile

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from vrag import __version__
from vrag.config import get_settings
from vrag.schemas.timeline import SCHEMA_VERSION

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())

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
