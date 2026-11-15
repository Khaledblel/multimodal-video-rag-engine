"""Response models of the search API."""

from pydantic import BaseModel


class Plan(BaseModel):
    """What a question becomes before the search. Later filters (speaker, emotion, sound) join as fields."""

    text: str
    video_id: str | None = None
    planned: bool = False
    error: str | None = None


class Hit(BaseModel):
    video_id: str
    source_name: str | None = None
    window_id: int
    start: float
    end: float
    text: str
    score: float
    text_rank: int | None = None
    frame_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    collection: str
    mode: str
    plan: Plan
    hits: list[Hit]
