"""Response models of the video list."""

from pydantic import BaseModel


class VideoSummary(BaseModel):
    video_id: str
    source_name: str | None = None
    duration_s: float
    indexed: bool
    playable: bool
