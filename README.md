# Multimodal Video RAG Engine

Ask questions about your videos in natural language. Speech transcripts and video frames are aligned on a single timeline and indexed in a vector database, so a question returns the matching moments and an answer grounded in what was said and shown.

> Early development: the API skeleton and video ingestion (audio, 1 FPS frames and a transcript with word timestamps) exist so far. Search and answering are being added step by step.

## Quick start

Requires Docker with Compose.

```bash
mkdir -p data/media
docker compose up -d --build
curl localhost:8000/health
```

Stop with `docker compose down`. To change the API port or log level, copy `.env.example` to `.env` and edit it.

## Ingest a video

Put a video in `data/media/`, then run:

```bash
docker compose exec worker python -m vrag.ingest /data/media/clip.mp4
```

Output goes to `data/artifacts/<video_id>/`: `timeline.json` (metadata, frame list and transcript), `audio.wav` (16 kHz mono) and `frames/` (the first frame of every second). Finished stages are skipped on re-runs; pass `--force` to redo them.

Speech is transcribed with word timestamps by [faster-whisper](https://github.com/SYSTRAN/faster-whisper), using the `small` model on the CPU by default; the model downloads on first use. The `ASR_*` settings in `.env` change the model, device, compute type or language (empty means auto-detect), and `ASR_ENABLED=false` skips transcription. To read a transcript: `docker compose exec worker python -m vrag.tools.show_transcript <video_id> --words`.

To check timestamp handling on a generated clip: `docker compose exec worker python -m vrag.tools.sync_check`.
