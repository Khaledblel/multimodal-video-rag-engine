"""Speech to text with faster-whisper; times are on the audio.wav timeline, which starts at container time 0."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Word as WhisperWord

from vrag.schemas.timeline import Segment, Word


def _word(word: WhisperWord) -> Word:
    return Word(
        start=word.start,
        end=max(word.start, word.end),
        text=word.word.strip(),
        prob=round(float(word.probability), 3),
    )


def _load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        pcm = wav.readframes(wav.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def transcribe(
    audio: Path, model_name: str, device: str, compute_type: str, language: str | None
) -> tuple[str | None, list[Segment]]:
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    raw, info = model.transcribe(
        _load(audio),
        language=language,
        log_progress=True,
        vad_filter=True,
        word_timestamps=True,
        # Earlier text is not fed back as a prompt, so one hallucination cannot cascade.
        condition_on_previous_text=False,
    )
    segments = [
        Segment(
            id=i,
            start=s.start,
            end=max(s.start, s.end),
            text=s.text.strip(),
            words=[_word(w) for w in s.words or []],
        )
        for i, s in enumerate(raw)
    ]
    return (info.language if segments else None), segments
