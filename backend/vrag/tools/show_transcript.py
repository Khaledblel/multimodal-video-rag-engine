"""Print a transcript with timestamps: python -m vrag.tools.show_transcript VIDEO_ID_OR_PATH [--words]."""

import argparse
from pathlib import Path

from vrag.artifacts import artifact_dir, load_timeline


def _ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", help="video_id or path to a timeline.json")
    parser.add_argument("--words", action="store_true", help="also print every word")
    args = parser.parse_args()

    path = Path(args.timeline)
    if not path.is_file():
        path = artifact_dir(args.timeline) / "timeline.json"
    timeline = load_timeline(path)
    if timeline is None:
        print(f"no timeline at {path}")
        return 1

    print(
        f"{timeline.video.source_name}: language={timeline.language or '-'}, "
        f"{len(timeline.segments)} segments"
    )
    for segment in timeline.segments:
        print(f"{_ts(segment.start)}-{_ts(segment.end)}  {segment.text}")
        if args.words:
            for word in segment.words:
                print(f"    {_ts(word.start)}-{_ts(word.end)}  {word.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
