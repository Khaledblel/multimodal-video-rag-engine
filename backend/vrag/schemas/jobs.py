"""An ingest job: a video handed to the worker as a file in data/jobs/."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Job(BaseModel):
    id: str
    source_name: str
    path: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    stages: list[str] = []
    stage: str | None = None
    video_id: str | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
