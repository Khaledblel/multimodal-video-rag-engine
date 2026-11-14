"""Ingest one video into data/artifacts/<video_id>/ (timeline.json, audio.wav, frames/) and the search index.

Usage: python -m vrag.ingest PATH [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from vrag import asr, index, media
from vrag.artifacts import artifact_dir, load_timeline, save_timeline
from vrag.config import get_settings
from vrag.schemas.timeline import StageRecord, Timeline

log = logging.getLogger("vrag.ingest")

STAGE_VERSIONS = {
    "probe": "1",
    "audio": "1",
    "frames": "1",
    "transcript": "1",
    "frame_vectors": "1",
    "index": "1",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _record(timeline: Timeline, stage: str, **params: object) -> None:
    timeline.stages[stage] = StageRecord(
        version=STAGE_VERSIONS[stage],
        params=params,
        finished_at=datetime.now(timezone.utc),
    )


def _is_done(timeline: Timeline, stage: str, **params: object) -> bool:
    record = timeline.stages.get(stage)
    return (
        record is not None
        and record.version == STAGE_VERSIONS[stage]
        and all(record.params.get(key) == value for key, value in params.items())
    )


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
        timeline.stages.pop("frame_vectors", None)  # vectors of the previous frames no longer apply
        save_timeline(timeline, timeline_path)

    settings = get_settings()
    language = settings.asr_language or None
    asr_params = {"model": settings.asr_model, "compute_type": settings.asr_compute_type, "language": language}
    if probe.audio_index is None:
        log.info("transcript: no audio stream")
        timeline.stages.pop("transcript", None)
        timeline.language, timeline.segments = None, []
    elif not settings.asr_enabled:
        log.info("transcript: disabled")
    elif _is_done(timeline, "transcript", **asr_params):
        log.info("transcript: up to date")
    else:
        log.info(
            "transcript: faster-whisper %s, %s on %s",
            settings.asr_model, settings.asr_compute_type, settings.asr_device,
        )
        timeline.language, timeline.segments = asr.transcribe(
            audio_path, settings.asr_model, settings.asr_device, settings.asr_compute_type, language
        )
        _record(timeline, "transcript", **asr_params)
        save_timeline(timeline, timeline_path)

    # One vector per frame, kept on disk so that re-indexing never has to embed the frames again.
    frame_vectors = None
    if not settings.frame_embed_enabled:
        log.info("frame vectors: disabled")
    else:
        vectors_path = out / "frame_vectors.npy"
        frame_params = {"model": settings.frame_embed_model}
        if _is_done(timeline, "frame_vectors", **frame_params) and vectors_path.exists():
            frame_vectors = np.load(vectors_path)
        if frame_vectors is not None and len(frame_vectors) == len(timeline.frames):
            log.info("frame vectors: up to date")
        else:
            from vrag import vision  # imported here so that PyTorch loads only when frames are embedded

            log.info(
                "frame vectors: %d frames with %s on %s",
                len(timeline.frames), settings.frame_embed_model, vision.device(),
            )
            frame_vectors = vision.embed_frames([out / frame.path for frame in timeline.frames])
            if frame_vectors.shape[1] != settings.frame_embed_dim:
                raise ValueError(
                    f"{settings.frame_embed_model} gives {frame_vectors.shape[1]}-d vectors, "
                    f"but FRAME_EMBED_DIM is {settings.frame_embed_dim}; set it to match"
                )
            frame_vectors = frame_vectors.astype(np.float16)
            tmp = vectors_path.with_name("frame_vectors.tmp.npy")
            np.save(tmp, frame_vectors)
            os.replace(tmp, vectors_path)
            _record(timeline, "frame_vectors", **frame_params)
            save_timeline(timeline, timeline_path)

    if not settings.index_enabled:
        log.info("index: disabled")
    else:
        # Stretches without speech have nothing to index but frames, so they get windows only with frame vectors.
        duration = timeline.video.duration_s if frame_vectors is not None else None
        windows = index.build_windows(timeline.segments, duration)
        index_params = {"model": settings.text_embed_model, "collection": index.collection_name()}
        if (
            _is_done(timeline, "index", **index_params)
            and timeline.windows == windows
            and index.count(video_id) == len(windows)
        ):
            log.info("index: up to date")
        else:
            log.info("index: %d windows into %s", len(windows), index_params["collection"])
            index.replace(timeline.video, windows, timeline.frames, frame_vectors)
            timeline.windows = windows
            _record(timeline, "index", **index_params)

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
        f"frames={len(timeline.frames)} audio={'yes' if video.has_audio else 'no'} "
        f"segments={len(timeline.segments)} language={timeline.language or '-'} "
        f"windows={len(timeline.windows)}"
    )
    print(f"artifacts: {artifact_dir(video.video_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
