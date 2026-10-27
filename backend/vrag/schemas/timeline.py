"""Versioned timeline contract shared by every pipeline stage; times are seconds matching HTML5 video.currentTime."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Final = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoInfo(_Model):
    video_id: str
    source_name: str | None = None
    sha256: str | None = None
    duration_s: float = Field(ge=0)
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    is_vfr: bool | None = None
    # Raw stream start times before normalising to the presentation timeline.
    video_start_s: float | None = None
    audio_start_s: float | None = None
    has_audio: bool = True


class _Span(_Model):
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def check_order(self) -> _Span:
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class Word(_Span):
    text: str
    prob: float | None = Field(default=None, ge=0, le=1)
    speaker_id: str | None = None


class Segment(_Span):
    id: int
    text: str
    words: list[Word] = Field(default_factory=list)
    speaker_id: str | None = None


class Frame(_Model):
    index: int = Field(ge=0)
    t: float = Field(ge=0)
    path: str  # relative to the video's artifact directory


class Speaker(_Model):
    id: str
    label: str | None = None


class StageRecord(_Model):
    version: str
    params: dict[str, Any] = Field(default_factory=dict)
    finished_at: datetime


class Timeline(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    video: VideoInfo
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    segments: list[Segment] = Field(default_factory=list)
    frames: list[Frame] = Field(default_factory=list)
    speakers: list[Speaker] = Field(default_factory=list)
