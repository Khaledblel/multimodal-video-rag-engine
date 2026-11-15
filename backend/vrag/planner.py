"""The query planner: before a search, the VL model turns a question into a plan, written as JSON.

So far a plan holds what to search for and, when the question names one, the video to search in. The question
otherwise passes through word for word.

Usage: python -m vrag.planner "QUESTION"
"""

from __future__ import annotations

import argparse
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from vrag import index, ollama
from vrag.artifacts import VideoEntry, list_videos
from vrag.config import get_settings
from vrag.schemas.search import Plan

log = logging.getLogger("vrag.planner")

PLAN_TOKENS = 300
MAX_VIDEOS = 200
SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "video": {"type": "integer"}},
    "required": ["text", "video"],
}

SYSTEM = """\
You prepare a search over a collection of videos. You get the list of videos and a question. \
Reply with JSON only, in this form: {"text": "...", "video": 0}

- "text" is what to search for: the question, word for word, without the words that only say which video \
it is about. Do not answer the question, and do not add or change words.
- "video" is the number of the video the question is about. Use 0 unless the question points to one of the \
listed videos as a video: by its title, or by what kind of recording it is. A word that is only the topic of \
the question does not point to a video. When in doubt, use 0.

Example, with the videos "1. budget_meeting.mp4" and "2. cooking_pasta.mp4":
Question: in the cooking video, when do they add the garlic?
{"text": "when do they add the garlic?", "video": 2}
Question: what do they decide about the budget?
{"text": "what do they decide about the budget?", "video": 0}"""


class Reply(BaseModel):
    """What the model must write."""

    text: str = Field(min_length=1, max_length=1000)
    video: int = Field(ge=0)


def draft(question: str, names: list[str], think: bool | None = None) -> Reply:
    """Ask the model for a plan over videos with these names, numbered from 1, and validate what it writes."""
    videos = "\n".join(f"{number}. {name}" for number, name in enumerate(names, start=1))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Videos:\n{videos}\n\nQuestion: {question}"},
    ]
    return Reply.model_validate_json(ollama.chat_json(messages, SCHEMA, max_tokens=PLAN_TOKENS, think=think))


def candidates() -> list[VideoEntry]:
    """The videos a question may be narrowed to: those indexed in the collection that searches use."""
    name = index.collection_name()
    return [video for video in list_videos() if video.collection == name]


def _search_text(question: str, planned: str) -> str:
    """The text to search for: the planned one if it only leaves words of the question out, else the question.

    A text with the same words as the question is not used either, so that such a search is exactly the one
    without the planner.
    """
    asked = re.findall(r"\w+", question.lower())
    words = re.findall(r"\w+", planned.lower())
    if not words or words == asked or not set(words) <= set(asked):
        return question
    return planned.strip()


def plan(question: str, video_id: str | None = None, use: bool | None = None) -> Plan:
    """The plan for a question. `use` overrides PLANNER_ENABLED; a `video_id` from the caller is kept as given.

    When the planner is off, has nothing to decide or fails, the question passes through as it is.
    """
    settings = get_settings()
    through = Plan(text=question, video_id=video_id)
    if use is None:
        use = settings.planner_enabled
    if not use or not settings.vlm_enabled or video_id is not None:
        return through
    videos = candidates()
    if not 2 <= len(videos) <= MAX_VIDEOS:
        return through

    try:
        model = ollama.show(settings.vlm_model)
        if model is None:
            raise ollama.OllamaError(ollama.not_downloaded(settings.vlm_model))
        names = [video.source_name or video.video_id for video in videos]
        reply = draft(question, names, think=False if model.thinks else None)
    except (ollama.OllamaError, ValidationError) as exc:
        reason = " ".join(str(exc).split())
        log.warning("planner failed, searching with the question as it is: %s", reason)
        return through.model_copy(update={"error": reason})

    chosen = None
    if 1 <= reply.video <= len(videos):
        chosen = videos[reply.video - 1].video_id
    elif reply.video:
        log.warning("planner chose video %d of %d; searching every video", reply.video, len(videos))
    return Plan(text=_search_text(question, reply.text), video_id=chosen, planned=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args()

    videos = candidates()
    for number, video in enumerate(videos, start=1):
        print(f"{number}. {video.source_name or video.video_id}  ({video.video_id})")
    if len(videos) < 2:
        print("nothing to plan: the planner chooses between videos, and fewer than two are indexed")
    result = plan(args.question, use=True)
    print(result.model_dump_json(indent=2))
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
