"""Eval set format: labeled queries whose answers are time spans in indexed videos."""

from typing import Literal

from pydantic import Field

from vrag.schemas.timeline import _Model, _Span


class Target(_Span):
    video_id: str


class EvalQuery(_Model):
    id: str
    category: Literal["speech", "visual", "speaker_emotion", "small_object", "multimodal"] = "speech"
    query: str = Field(min_length=1)
    targets: list[Target] = Field(min_length=1)


class EvalSet(_Model):
    tolerance_s: float = Field(default=2.0, ge=0)
    sources: dict[str, str] = Field(default_factory=dict)
    queries: list[EvalQuery] = Field(default_factory=list)
