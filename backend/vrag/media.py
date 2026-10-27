"""FFmpeg wrappers: probe, audio extraction and 1 FPS frame sampling on the container timeline."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vrag.schemas.timeline import Frame, VideoInfo

# First frame of each whole second by PTS; the 1 ms slack absorbs float error at boundaries.
_SELECT = (
    "select='gte(t+0.001,0)*(isnan(prev_selected_t)"
    "+gt(floor(t+0.001),floor(prev_selected_t+0.001)))'"
)
_SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*(\S+)")
# Local files only, so playlists or concat files cannot pull in URLs or other paths.
_LOCAL_ONLY = ["-protocol_whitelist", "file"]


class MediaError(RuntimeError):
    pass


@dataclass
class ProbeResult:
    info: VideoInfo
    video_index: int
    audio_index: int | None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"{cmd[0]} exited with {proc.returncode}:\n{tail}")
    return proc


def _num(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _rate(value: str | None) -> float | None:
    num, _, den = (value or "").partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def probe(src: Path, video_id: str, sha256: str) -> ProbeResult:
    proc = _run(
        [
            "ffprobe", "-v", "error", *_LOCAL_ONLY,
            "-print_format", "json", "-show_format", "-show_streams", str(src),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next(
        (
            s for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise MediaError(f"{src.name} has no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    avg, base = _rate(video.get("avg_frame_rate")), _rate(video.get("r_frame_rate"))
    info = VideoInfo(
        video_id=video_id,
        source_name=src.name,
        sha256=sha256,
        duration_s=_num(data.get("format", {}).get("duration")) or _num(video.get("duration")) or 0.0,
        width=video.get("width"),
        height=video.get("height"),
        fps=avg or base,
        is_vfr=abs(avg - base) / base > 0.01 if avg and base else None,
        video_start_s=_num(video.get("start_time")),
        audio_start_s=_num(audio.get("start_time")) if audio else None,
        has_audio=audio is not None,
    )
    return ProbeResult(info, video["index"], audio["index"] if audio else None)


def extract_audio(src: Path, audio_index: int, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    _run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y", *_LOCAL_ONLY, "-i", str(src),
            "-map", f"0:{audio_index}", "-vn",
            # Pads leading silence if audio starts after the container, keeping it on the video timeline.
            "-af", "aresample=async=1:first_pts=0",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(tmp),
        ]
    )
    tmp.replace(dst)


def extract_frames(src: Path, video_index: int, out_dir: Path) -> list[Frame]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    proc = _run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-y",
            *_LOCAL_ONLY, "-i", str(src),
            "-map", f"0:{video_index}", "-an", "-sn", "-dn",
            "-vf", f"{_SELECT},showinfo",
            "-fps_mode", "passthrough", "-q:v", "3", "-start_number", "0",
            str(out_dir / "%06d.jpg"),
        ]
    )
    times = [float(t) for _, t in _SHOWINFO.findall(proc.stderr)]
    written = len(list(out_dir.glob("*.jpg")))
    if len(times) != written:
        raise MediaError(f"showinfo reported {len(times)} frames but {written} files were written")
    return [
        Frame(index=i, t=max(0.0, round(t, 6)), path=f"frames/{i:06d}.jpg")
        for i, t in enumerate(times)
    ]
