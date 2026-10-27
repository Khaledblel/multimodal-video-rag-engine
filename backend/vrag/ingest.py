"""Ingest one video into data/artifacts/<video_id>/: timeline.json, audio.wav and frames/.

Usage: python -m vrag.ingest PATH [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from vrag import media
from vrag.artifacts import artifact_dir, load_timeline, save_timeline
from vrag.config import get_settings
from vrag.schemas.timeline import StageRecord, Timeline

log = logging.getLogger("vrag.ingest")

STAGE_VERSIONS = {"probe": "1", "audio": "1", "frames": "1"}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _record(timeline: Timeline, stage: str, **params: object) -> None:
    timeline.stages[stage] = StageRecord(
        version=STAGE_VERSIONS[stage],
        params=params,
        finished_at=datetime.now(timezone.utc),
    )


def _is_done(timeline: Timeline, stage: str) -> bool:
    record = timeline.stages.get(stage)
    return record is not None and record.version == STAGE_VERSIONS[stage]


def ingest(src: Path, force: bool = False) -> Timeline:
    src = src.resolve()
    sha256 = _sha256(src)
    video_id = sha256[:16]
    out = artifact_dir(video_id)
    out.mkdir(parents=True, exist_ok=True)
    timeline_path = out / "timeline.json"

    probe = media.probe(src, video_id, sha256)
    timeline = None if force else load_timeline(timeline_path)
    if timeline is None:
        timeline = Timeline(video=probe.info)
    timeline.video = probe.info
    _record(timeline, "probe")
    save_timeline(timeline, timeline_path)

    audio_path = out / "audio.wav"
    if probe.audio_index is None:
        log.info("audio: no audio stream")
        timeline.stages.pop("audio", None)
        audio_path.unlink(missing_ok=True)
    elif _is_done(timeline, "audio") and audio_path.exists():
        log.info("audio: up to date")
    else:
        log.info("audio: extracting")
        media.extract_audio(src, probe.audio_index, audio_path)
        _record(timeline, "audio", sample_rate=16000, channels=1)
        save_timeline(timeline, timeline_path)

    frames_dir = out / "frames"
    if _is_done(timeline, "frames") and len(list(frames_dir.glob("*.jpg"))) == len(timeline.frames) > 0:
        log.info("frames: up to date")
    else:
        log.info("frames: extracting")
        timeline.frames = media.extract_frames(src, probe.video_index, frames_dir)
        _record(timeline, "frames", rule="first frame of each whole second")

    save_timeline(timeline, timeline_path)
    return timeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--force", action="store_true", help="redo every stage")
    args = parser.parse_args()
    logging.basicConfig(level=get_settings().log_level.upper(), format="%(message)s")

    try:
        timeline = ingest(args.video, force=args.force)
    except (OSError, media.MediaError, ValidationError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    video = timeline.video
    print(
        f"video_id={video.video_id} duration={video.duration_s:.2f}s "
        f"frames={len(timeline.frames)} audio={'yes' if video.has_audio else 'no'}"
    )
    print(f"artifacts: {artifact_dir(video.video_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
