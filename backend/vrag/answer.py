"""Answer a question from the moments a search finds: their transcript and frames go to the VL model, which cites them.

Usage: python -m vrag.answer "QUESTION" [--video VIDEO_ID] [--limit 5] [--frames 3] [--mode fused]
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx
from PIL import Image
from pydantic import ValidationError

from vrag import index, ollama
from vrag.artifacts import artifact_dir, load_timeline
from vrag.config import get_settings
from vrag.schemas.answer import AnswerResponse, AnswerStats, FrameRef, Line, Moment
from vrag.schemas.search import Hit
from vrag.schemas.timeline import Frame, Timeline

log = logging.getLogger("vrag.answer")

# Two windows this close have no other window between them, so hits on both make one moment.
MERGE_GAP_S = index.WINDOW_GAP_S
ANSWER_TOKENS = 1024
THINKING_TOKENS = 3072
PROMPT_MARGIN = 256
PIXELS_PER_TOKEN = 32
MESSAGE_PER_MOMENT = True

SYSTEM = """\
You answer questions about videos from the moments a search has found in them.

Each moment is given with its number, its video, its time range, its frames (still images, with the time of \
each) and what is said in it, one line per sentence with the time it starts. \
The search is not perfect, so some moments may have nothing to do with the question. Ignore those.

Rules:
- Use only what the moments say and show. If they do not answer the question, say so instead of guessing.
- After each statement, cite the moment it rests on by its number in square brackets, like [1] or [2][3].
- To say when something happens, use the times given in the moment, like "at 0:51".
- Answer in the language of the question, in a few sentences."""

NOTHING_INDEXED = "There is nothing to answer from: no indexed video matches. Ingest a video first."


@dataclass
class Prepared:
    """A question with its evidence, ready to be sent to the model."""

    query: str
    mode: str
    moments: list[Moment]
    messages: list[dict]
    hits: int
    frames: int
    frames_dropped: int
    estimate: int
    max_tokens: int
    think: bool | None
    search_s: float


def clock(seconds: float) -> str:
    """75.4 becomes "1:15" and 3725 becomes "1:02:05": the form in which the model reads and writes times."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def text_tokens(text: str) -> int:
    """A rough count that errs on the high side in any script: one token per three bytes of UTF-8."""
    return len(text.encode()) // 3 + 1


def fit(width: int, height: int, side: int) -> tuple[int, int]:
    """The size an image is sent at: scaled down until its longer side is about `side`, in multiples of 32."""
    scale = min(1.0, side / max(width, height))
    return (
        max(PIXELS_PER_TOKEN, round(width * scale / PIXELS_PER_TOKEN) * PIXELS_PER_TOKEN),
        max(PIXELS_PER_TOKEN, round(height * scale / PIXELS_PER_TOKEN) * PIXELS_PER_TOKEN),
    )


def image_tokens(size: tuple[int, int]) -> int:
    """What an image of this size should cost the model. tools/vlm_check compares it with Ollama's count."""
    return (size[0] // PIXELS_PER_TOKEN) * (size[1] // PIXELS_PER_TOKEN) + 2


def as_jpeg(image: Image.Image) -> str:
    """Base64 JPEG, the form in which Ollama takes an image."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _frame_image(path: Path, side: int) -> tuple[str, int]:
    """A frame scaled down for the model, and the tokens it should cost."""
    with Image.open(path) as image:
        size = fit(*image.size, side)
        image.draft("RGB", size)  # lets the JPEG decoder shrink the picture while it decodes
        scaled = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return as_jpeg(scaled), image_tokens(size)


def merge_hits(hits: list[Hit]) -> list[list[Hit]]:
    """Group the hits of one video that follow each other in time. Groups come in the order of their best hit."""
    groups: list[list[Hit]] = []
    for hit in sorted(hits, key=lambda hit: (hit.video_id, hit.start)):
        group = groups[-1] if groups else None
        if (
            group
            and group[0].video_id == hit.video_id
            and hit.start - max(member.end for member in group) <= MERGE_GAP_S
        ):
            group.append(hit)
        else:
            groups.append([hit])
    rank = {(hit.video_id, hit.window_id): position for position, hit in enumerate(hits)}
    groups.sort(key=lambda group: min(rank[hit.video_id, hit.window_id] for hit in group))
    return groups


@lru_cache(maxsize=8)
def _read_timeline(path: Path, modified_ns: int) -> Timeline | None:
    return load_timeline(path)


def _timeline(video_id: str) -> Timeline | None:
    """A video's timeline, read again only when the file has changed; None when it cannot be read."""
    path = artifact_dir(video_id) / "timeline.json"
    try:
        return _read_timeline(path, path.stat().st_mtime_ns)
    except (OSError, ValidationError):
        return None


def _spread(rows: list[int], wanted: int) -> list[int]:
    """Up to `wanted` of the rows, evenly spread: the middle of each of `wanted` equal parts."""
    if wanted >= len(rows):
        return rows
    return [rows[int((i + 0.5) * len(rows) / wanted)] for i in range(wanted)]


def _moment(n: int, group: list[Hit], frames_per_hit: int) -> tuple[Moment, list[Frame]]:
    """A moment from hits that follow each other, and the frames to show for it."""
    first = group[0]
    start, end = first.start, max(hit.end for hit in group)
    lines: list[Line] = []
    frames: list[Frame] = []
    timeline = _timeline(first.video_id)
    if timeline is not None:
        # A window is made of whole segments, so a moment's lines are the segments whose middle lies in it.
        lines = [
            Line(start=segment.start, end=segment.end, text=segment.text)
            for segment in timeline.segments
            if start - 0.01 <= (segment.start + segment.end) / 2 <= end + 0.01
        ]
        rows = index.span_frames(start, end, timeline.frames)
        # A moment merged from several hits gets the frames of each, spread over its whole span.
        frames = [timeline.frames[row] for row in _spread(rows, frames_per_hit * len(group))]
    if not lines:
        # No segments to show (the artifacts are gone, or have changed since indexing): use the indexed text.
        lines = [Line(start=hit.start, end=hit.end, text=hit.text) for hit in group if hit.text]
    moment = Moment(
        n=n,
        video_id=first.video_id,
        source_name=first.source_name,
        start=start,
        end=end,
        score=max(hit.score for hit in group),
        window_ids=[hit.window_id for hit in group],
        lines=lines,
    )
    return moment, frames


def _describe(moment: Moment, first: int | None = None) -> str:
    """A moment as the model reads it.

    Its frames are the images right above the text, or, when every image goes in one message, the images
    numbered from `first`.
    """
    parts = [
        f"Moment [{moment.n}]",
        f"Video: {moment.source_name or moment.video_id}",
        f"Time: {clock(moment.start)} to {clock(moment.end)}",
    ]
    if moment.frames:
        last = (first or 1) + len(moment.frames) - 1
        if first is None:
            which = "the images above, in order"
        elif last == first:
            which = f"image {first} above"
        else:
            which = f"images {first} to {last} above, in order"
        parts.append(f"Frames ({which}): {', '.join(clock(frame.t) for frame in moment.frames)}")
    if moment.lines:
        parts.append("Said:")
        parts += [f"{clock(line.start)}  {line.text}" for line in moment.lines]
    else:
        parts.append("Nothing is said in this moment.")
    return "\n".join(parts)


def prepare(
    query: str,
    limit: int = 5,
    video_id: str | None = None,
    mode: str = "fused",
    frames_per_hit: int = 3,
) -> Prepared:
    """Search, merge the hits into moments, and build a prompt that fits in the context size."""
    settings = get_settings()
    model = ollama.show(settings.vlm_model)
    if model is None:
        raise ollama.OllamaError(
            f"{settings.vlm_model} is not downloaded; run: "
            "docker compose exec worker python -m vrag.tools.vlm_check --pull"
        )
    think = settings.vlm_think if model.thinks else None
    if frames_per_hit and not model.vision:
        log.warning("%s takes no images: answering from the transcript alone", settings.vlm_model)
        frames_per_hit = 0
    max_tokens = ANSWER_TOKENS + (THINKING_TOKENS if think else 0)

    started = time.perf_counter()
    hits = index.search(query, limit, video_id, mode)
    search_s = time.perf_counter() - started

    question = f"Question: {query}"
    room = settings.vlm_num_ctx - max_tokens - PROMPT_MARGIN
    used = text_tokens(SYSTEM) + text_tokens(question)

    moments: list[Moment] = []
    shown: list[list[Frame]] = []
    groups = merge_hits(hits)
    for n, group in enumerate(groups, start=1):
        moment, frames = _moment(n, group, frames_per_hit)
        cost = text_tokens(_describe(moment))
        if moments and used + cost > room:
            log.warning("context full: answering from %d of %d moments", len(moments), len(groups))
            break
        used += cost
        moments.append(moment)
        shown.append(frames)

    images: list[list[str]] = []
    dropped = 0
    for moment, frames in zip(moments, shown):
        encoded: list[str] = []
        for frame in frames:
            if dropped:
                dropped += 1
                continue
            try:
                data, cost = _frame_image(artifact_dir(moment.video_id) / frame.path, settings.vlm_frame_side)
            except OSError as exc:
                log.warning("frame %s of %s cannot be read: %s", frame.path, moment.video_id, exc)
                continue
            if used + cost > room:
                dropped += 1
                continue
            used += cost
            encoded.append(data)
            moment.frames.append(FrameRef(index=frame.index, t=frame.t))
        images.append(encoded)

    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    if MESSAGE_PER_MOMENT:
        for moment, encoded in zip(moments, images):
            message: dict = {"role": "user", "content": _describe(moment)}
            if encoded:
                message["images"] = encoded
            messages.append(message)
        messages.append({"role": "user", "content": question})
    else:
        texts, first = [], 1
        for moment, encoded in zip(moments, images):
            texts.append(_describe(moment, first))
            first += len(encoded)
        message = {"role": "user", "content": "\n\n".join([*texts, question])}
        if first > 1:
            message["images"] = [data for encoded in images for data in encoded]
        messages.append(message)

    return Prepared(
        query=query,
        mode=mode,
        moments=moments,
        messages=messages,
        hits=len(hits),
        frames=sum(len(encoded) for encoded in images),
        frames_dropped=dropped,
        estimate=used,
        max_tokens=max_tokens,
        think=think,
        search_s=search_s,
    )


def cited(answer: str, moments: int) -> list[int]:
    """The moment numbers an answer cites, in order of first use. Numbers that match no moment are left out."""
    found: list[int] = []
    for group in re.findall(r"\[(?:moments?\s*)?([\d,\s]+)\]", answer, flags=re.IGNORECASE):
        for number in re.findall(r"\d+", group):
            if 1 <= int(number) <= moments and int(number) not in found:
                found.append(int(number))
    return found


def _seconds(nanoseconds: int | None) -> float | None:
    return None if nanoseconds is None else round(nanoseconds / 1e9, 3)


def summary(stats: AnswerStats) -> str:
    """One line on what an answer cost: what went in, what came out, how long it took, where the model sits."""
    frames = f"{stats.frames} frames"
    if stats.frames_dropped:
        frames += f" ({stats.frames_dropped} left out to fit the context)"
    first = "?" if stats.first_token_s is None else f"{stats.first_token_s:.2f}"
    prompt = "?" if stats.prompt_tokens is None else stats.prompt_tokens
    written = "?" if stats.answer_tokens is None else stats.answer_tokens
    parts = [
        f"{stats.hits} hits in {stats.moments} moments, {frames}",
        f"prompt {prompt} of {stats.context} tokens (estimated {stats.prompt_tokens_estimate})",
        f"answer {written} tokens (ended by: {stats.done_reason})",
        f"search {stats.search_s:.2f} s, first token {first} s, total {stats.total_s:.2f} s",
    ]
    if stats.size_bytes:
        on_gpu = (stats.vram_bytes or 0) / stats.size_bytes
        parts.append(
            f"{stats.model}: {stats.size_bytes / 2**30:.1f} GiB, {on_gpu:.0%} on the GPU, "
            f"context {stats.loaded_context}"
        )
    return "; ".join(parts)


def generate(prepared: Prepared) -> Iterator[dict]:
    """The events of one answer, in order: "moments", then "token" pieces, then "done" or "error".

    A model that thinks first also sends "thinking" pieces before its tokens.
    """
    settings = get_settings()
    yield {
        "type": "moments",
        "query": prepared.query,
        "mode": prepared.mode,
        "moments": [moment.model_dump() for moment in prepared.moments],
    }
    if not prepared.moments:
        yield {"type": "done", "answer": NOTHING_INDEXED, "citations": [], "stats": None}
        return

    started = time.perf_counter()
    first_token_s = None
    pieces: list[str] = []
    final: dict = {}
    try:
        for chunk in ollama.chat(prepared.messages, max_tokens=prepared.max_tokens, think=prepared.think):
            message = chunk.get("message") or {}
            if message.get("thinking"):
                yield {"type": "thinking", "text": message["thinking"]}
            if message.get("content"):
                if first_token_s is None:
                    first_token_s = time.perf_counter() - started
                pieces.append(message["content"])
                yield {"type": "token", "text": message["content"]}
            if chunk.get("done"):
                final = chunk
    except ollama.OllamaError as exc:
        log.warning("answer failed: %s", exc)
        yield {"type": "error", "detail": str(exc)}
        return
    total_s = time.perf_counter() - started

    try:
        loaded = ollama.loaded(settings.vlm_model) or {}
    except ollama.OllamaError:
        loaded = {}
    stats = AnswerStats(
        model=settings.vlm_model,
        hits=prepared.hits,
        moments=len(prepared.moments),
        frames=prepared.frames,
        frames_dropped=prepared.frames_dropped,
        context=settings.vlm_num_ctx,
        prompt_tokens_estimate=prepared.estimate,
        prompt_tokens=final.get("prompt_eval_count"),
        answer_tokens=final.get("eval_count"),
        done_reason=final.get("done_reason"),
        search_s=round(prepared.search_s, 3),
        first_token_s=None if first_token_s is None else round(first_token_s, 3),
        total_s=round(total_s, 3),
        load_s=_seconds(final.get("load_duration")),
        loaded_context=loaded.get("context_length"),
        size_bytes=loaded.get("size"),
        vram_bytes=loaded.get("size_vram"),
    )
    log.info("answer: %s", summary(stats))

    answer = "".join(pieces).strip()
    yield {
        "type": "done",
        "answer": answer,
        "citations": cited(answer, len(prepared.moments)),
        "stats": stats.model_dump(),
    }


def collect(prepared: Prepared) -> AnswerResponse:
    """Run an answer to its end, for a caller that does not stream."""
    for event in generate(prepared):
        if event["type"] == "error":
            raise ollama.OllamaError(event["detail"])
        if event["type"] == "done":
            return AnswerResponse(
                query=prepared.query,
                mode=prepared.mode,
                answer=event["answer"],
                citations=event["citations"],
                moments=prepared.moments,
                stats=event["stats"],
            )
    raise ollama.OllamaError("the model ended without an answer")


def _label(moment: Moment) -> str:
    return f"[{moment.n}] {moment.source_name or moment.video_id}  {clock(moment.start)} to {clock(moment.end)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--video", help="search only this video_id")
    parser.add_argument("--limit", type=int, default=5, help="search hits to answer from (default 5)")
    parser.add_argument("--frames", type=int, default=3, help="frames per hit (default 3; 0 sends none)")
    parser.add_argument("--mode", choices=index.MODES, default="fused")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.vlm_enabled:
        print("answers are turned off (VLM_ENABLED=false)", file=sys.stderr)
        return 1
    if args.mode == "frame" and not settings.frame_embed_enabled:
        print("frame search is turned off (FRAME_EMBED_ENABLED=false)", file=sys.stderr)
        return 1
    try:
        prepared = prepare(args.question, max(1, args.limit), args.video, args.mode, max(0, args.frames))
    except (ollama.OllamaError, httpx.HTTPError) as exc:
        # httpx errors come from the embedder, which embeds the question for the frame search.
        print(f"cannot answer: {exc}", file=sys.stderr)
        return 1

    for moment in prepared.moments:
        times = ", ".join(clock(frame.t) for frame in moment.frames) or "none"
        print(f"{_label(moment)}  frames: {times}")
    print()
    for event in generate(prepared):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        elif event["type"] == "error":
            print(f"\nfailed: {event['detail']}", file=sys.stderr)
            return 1
        elif event["type"] == "done":
            if not prepared.moments:
                print(event["answer"])
                return 1
            print("\n")
            for moment in prepared.moments:
                if moment.n in event["citations"]:
                    print(f"cited: {_label(moment)}")
            print(summary(AnswerStats(**event["stats"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
