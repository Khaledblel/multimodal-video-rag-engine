# Multimodal Video RAG Engine

Ask questions about your videos in natural language. Speech transcripts and video frames are aligned on a single timeline and indexed in a vector database, so a question returns the matching moments and an answer grounded in what was said and shown.

> Early development: video ingestion (audio, 1 FPS frames and a transcript with word timestamps) and search over the transcripts exist so far. Frame search and answering are being added step by step.

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

## Search

Ingesting also indexes the transcript. It is grouped into windows of up to 20 seconds, embedded with `sentence-transformers/all-MiniLM-L6-v2` (English) through [fastembed](https://github.com/qdrant/fastembed), and stored in Qdrant as one point per window. `TEXT_EMBED_MODEL` picks another fastembed text model (each model gets its own collection; ingest the videos again to fill it), and `INDEX_ENABLED=false` skips indexing.

```bash
curl -G localhost:8000/search --data-urlencode 'q=when do they talk about the budget'
```

Each hit gives the video, the window's start and end in seconds, its text and a similarity score. Add `video_id=...` to search one video, or `limit=...` for more than 5 hits.

## Evaluate retrieval

`backend/eval/queries.json` holds labeled queries. Each has a `category` (`speech`, `visual`, `speaker_emotion`, `small_object` or `multimodal`; only speech is searchable so far) and one or more target time spans:

```json
{
  "tolerance_s": 2.0,
  "sources": {"0123456789abcdef": "where the clip comes from"},
  "queries": [
    {"id": "q01", "category": "speech", "query": "when do they talk about the budget",
     "targets": [{"video_id": "0123456789abcdef", "start": 42.0, "end": 51.5}]}
  ]
}
```

`docker compose exec worker python -m vrag.evaluate` prints Recall@1/5/10 and MRR per category and saves them in `data/eval/`. A hit counts when it overlaps a target, give or take `tolerance_s`. The file is built into the worker image, so rebuild after editing it (`docker compose up -d --build`) or pass another path, such as one under `/data`.
