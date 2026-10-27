import os
from pathlib import Path

from vrag.config import get_settings
from vrag.schemas.timeline import Timeline


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
