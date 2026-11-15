"""Models of the answer API: the moments an answer is built from, and the answer itself."""

from pydantic import BaseModel

from vrag.schemas.search import Plan


class Line(BaseModel):
    """One transcript line of a moment."""

    start: float
    end: float
    text: str


class FrameRef(BaseModel):
    """A frame that was shown to the model."""

    index: int
    t: float


class Moment(BaseModel):
    """One or more search hits of a video that follow each other in time. An answer cites it as [n]."""

    n: int
    video_id: str
    source_name: str | None = None
    start: float
    end: float
    score: float
    window_ids: list[int]
    lines: list[Line] = []
    frames: list[FrameRef] = []


class AnswerStats(BaseModel):
    """What one answer cost. Token counts and the loaded_*, size and vram figures come from Ollama."""

    model: str
    hits: int
    moments: int
    frames: int
    frames_dropped: int  # frames left out so that the prompt fits in the context
    context: int
    prompt_tokens_estimate: int
    prompt_tokens: int | None = None
    answer_tokens: int | None = None
    done_reason: str | None = None
    plan_s: float
    search_s: float
    first_token_s: float | None = None
    total_s: float
    load_s: float | None = None
    loaded_context: int | None = None
    size_bytes: int | None = None
    vram_bytes: int | None = None


class AnswerResponse(BaseModel):
    query: str
    mode: str
    plan: Plan
    answer: str
    citations: list[int]
    moments: list[Moment]
    stats: AnswerStats | None = None
