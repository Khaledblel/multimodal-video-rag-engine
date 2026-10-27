"""Validate a timeline JSON file: python -m vrag.tools.check_timeline [path]."""

import sys
from pathlib import Path

from pydantic import ValidationError

from vrag.schemas.timeline import Timeline

DEFAULT_PATH = "samples/timeline.sample.json"


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
    try:
        timeline = Timeline.model_validate_json(path.read_text())
    except (OSError, ValidationError) as exc:
        print(f"INVALID {path}\n{exc}")
        return 1
    print(
        f"OK {path}: schema v{timeline.schema_version}, "
        f"{len(timeline.segments)} segments, {len(timeline.frames)} frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
