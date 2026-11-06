"""Response models of the search API."""

from pydantic import BaseModel


class Hit(BaseModel):
    video_id: str
    source_name: str | None = None
    window_id: int
    start: float
    end: float
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    collection: str
    hits: list[Hit]
