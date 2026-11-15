import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from vrag.config import get_settings
from vrag.schemas.timeline import Timeline


@dataclass(frozen=True)
class VideoEntry:
    video_id: str
    source_name: str | None
    duration_s: float
    collection: str | None


def artifact_dir(video_id: str) -> Path:
    return get_settings().data_dir / "artifacts" / video_id


def load_timeline(path: Path) -> Timeline | None:
    if not path.exists():
        return None
    return Timeline.model_validate_json(path.read_text())


def save_timeline(timeline: Timeline, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(timeline.model_dump_json(indent=2))
    os.replace(tmp, path)


@lru_cache(maxsize=1024)
def _entry(path: Path, modified_ns: int) -> VideoEntry | None:
    """The few facts of a timeline that a list of videos needs, read without validating the whole file."""
    try:
        data = json.loads(path.read_text())
        video = data["video"]
        indexed = (data.get("stages") or {}).get("index") or {}
        return VideoEntry(
            video_id=str(video["video_id"]),
            source_name=video.get("source_name"),
            duration_s=float(video.get("duration_s") or 0.0),
            collection=(indexed.get("params") or {}).get("collection"),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def list_videos() -> list[VideoEntry]:
    """Every ingested video, from the timelines on disk, sorted by name. A file is read again only when it changed."""
    entries = []
    for path in (get_settings().data_dir / "artifacts").glob("*/timeline.json"):
        try:
            entry = _entry(path, path.stat().st_mtime_ns)
        except OSError:
            continue
        if entry is not None:
            entries.append(entry)
    return sorted(entries, key=lambda entry: ((entry.source_name or "").lower(), entry.video_id))
