"""Timestamp self-check: build a clip with flashes and beeps at known times, ingest it, verify alignment.

Usage: python -m vrag.tools.sync_check
"""

from __future__ import annotations

import array
import math
import subprocess
import wave
from pathlib import Path

from PIL import Image, ImageStat

from vrag.artifacts import artifact_dir
from vrag.config import get_settings
from vrag.ingest import ingest

EVENT_TIMES = (2.0, 6.0)  # flash and beep both start here on the container timeline
AUDIO_START_S = 0.5  # audio stream starts late, as in many real files
DURATION_S = 10


def make_clip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flash = "+".join(f"between(t,{t - 0.1:.2f},{t + 0.3:.2f})" for t in EVENT_TIMES)
    beep = "+".join(
        f"between(t,{t - AUDIO_START_S:.2f},{t - AUDIO_START_S + 0.4:.2f})" for t in EVENT_TIMES
    )
    video = (
        f"color=c=black:s=320x180:r=25:d={DURATION_S},"
        f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{flash}'"
    )
    audio = (
        f"sine=frequency=1000:sample_rate=48000:duration={DURATION_S - AUDIO_START_S},"
        f"volume='{beep}':eval=frame"
    )
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", video,
            "-itsoffset", str(AUDIO_START_S), "-f", "lavfi", "-i", audio,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le",
            str(path),
        ],
        check=True,
    )


def _ffprobe_times(path: Path) -> list[float]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    return sorted(float(token.strip(",")) for token in out.split())


def _expected_times(times: list[float]) -> list[float]:
    expected, last_slot = [], None
    for t in times:
        slot = math.floor(t + 0.001)
        if t + 0.001 >= 0 and (last_slot is None or slot > last_slot):
            expected.append(t)
            last_slot = slot
    return expected


def _brightness(path: Path) -> float:
    with Image.open(path) as img:
        return ImageStat.Stat(img.convert("L")).mean[0]


def _beep_onsets(path: Path) -> list[float]:
    with wave.open(str(path)) as wav:
        rate = wav.getframerate()
        samples = array.array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    step = rate // 100
    onsets, was_loud = [], False
    for i in range(0, len(samples) - step, step):
        window = samples[i : i + step]
        loud = math.sqrt(sum(s * s for s in window) / step) > 800
        if loud and not was_loud:
            onsets.append(i / rate)
        was_loud = loud
    return onsets


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def main() -> int:
    clip = get_settings().data_dir / "media" / "sync_test.mkv"
    print(subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
    make_clip(clip)
    timeline = ingest(clip, force=True)
    video = timeline.video
    out = artifact_dir(video.video_id)
    print(
        f"probe: duration={video.duration_s:.3f}s video_start={video.video_start_s} "
        f"audio_start={video.audio_start_s} vfr={video.is_vfr}"
    )

    got = [frame.t for frame in timeline.frames]
    want = _expected_times(_ffprobe_times(clip))
    bright = [math.floor(f.t + 0.001) for f in timeline.frames if _brightness(out / f.path) > 150]
    onsets = _beep_onsets(out / "audio.wav")

    results = [
        _report(
            "frame times match ffprobe",
            len(got) == len(want) and all(abs(a - b) < 1e-3 for a, b in zip(got, want)),
            f"{len(got)} frames, expected {len(want)}",
        ),
        _report("flash frames", bright == [int(t) for t in EVENT_TIMES], f"bright slots {bright}"),
        _report(
            "beep onsets",
            len(onsets) == len(EVENT_TIMES)
            and all(abs(o - e) < 0.05 for o, e in zip(onsets, EVENT_TIMES)),
            f"measured {[round(o, 3) for o in onsets]} s",
        ),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
