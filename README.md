# Multimodal Video RAG Engine

Ask questions about your videos in natural language. Speech transcripts and video frames are aligned on a single timeline and indexed in a vector database, so a question returns the matching moments and an answer grounded in what was said and shown.

> Early development: video ingestion (audio, 1 FPS frames and a transcript with word timestamps), search over what is said and what is shown, and answers that cite the moments they rest on exist so far. More is being added step by step.

## Quick start

Requires Docker with Compose.

```bash
mkdir -p data/media
docker compose up -d --build
curl localhost:8000/health
```

Stop with `docker compose down`. To change the API port or log level, copy `.env.example` to `.env` and edit it.

With an NVIDIA GPU and the NVIDIA Container Toolkit on the host, add `COMPOSE_FILE=compose.yaml:compose.gpu.yaml` to `.env` before starting, so the containers that run models can use the GPU. `docker compose exec worker python -m vrag.tools.gpu_check` confirms it. Without a GPU everything still runs, more slowly. The worker image includes PyTorch and is several GB.

## Ingest a video

Put a video in `data/media/`, then run:

```bash
docker compose exec worker python -m vrag.ingest /data/media/clip.mp4
```

Output goes to `data/artifacts/<video_id>/`: `timeline.json` (metadata, frame list and transcript), `audio.wav` (16 kHz mono), `frames/` (the first frame of every second) and `frame_vectors.npy` (one embedding per frame). Finished stages are skipped on re-runs; pass `--force` to redo them.

Speech is transcribed with word timestamps by [faster-whisper](https://github.com/SYSTRAN/faster-whisper), using the `small` model on the CPU by default; the model downloads on first use. The `ASR_*` settings in `.env` change the model, device, compute type or language (empty means auto-detect), and `ASR_ENABLED=false` skips transcription. To read a transcript: `docker compose exec worker python -m vrag.tools.show_transcript <video_id> --words`.

To check timestamp handling on a generated clip: `docker compose exec worker python -m vrag.tools.sync_check`.

## Search

Ingesting also indexes the video for search, in windows of up to 20 seconds that together cover the whole video:

- **What is said:** each window's transcript is embedded with `sentence-transformers/all-MiniLM-L6-v2` (English) through [fastembed](https://github.com/qdrant/fastembed).
- **What is shown:** each frame is embedded with [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-384). A window keeps the vectors of its frames, and a query matches it through its best frame.

Both are stored in Qdrant as one point per window. A search runs both ways and fuses the two rankings.

```bash
curl -G localhost:8000/search --data-urlencode 'q=when do they talk about the budget'
curl -G localhost:8000/search --data-urlencode 'q=a red car at night' -d mode=frame
```

Each hit gives the video, the window's start and end in seconds, its text, a score, and its rank in the text search and in the frame search. Options: `mode=text` or `mode=frame` to search one way only, `video_id=...` to search one video, and `limit=...` for more than 5 hits.

Settings in `.env`:

- `TEXT_EMBED_MODEL` picks another fastembed text model.
- `FRAME_EMBED_MODEL` and `FRAME_EMBED_DIM` pick another fixed-resolution SigLIP checkpoint and its vector size.
- `FUSION_TEXT_WEIGHT` and `FUSION_FRAME_WEIGHT` weigh the two rankings in a fused search.
- `FRAME_EMBED_ENABLED=false` turns frame search off, and `INDEX_ENABLED=false` skips indexing.

Each combination of models gets its own collection; ingest the videos again to fill it.

## Ask a question

Answers are written by [Qwen3-VL](https://ollama.com/library/qwen3-vl) (8B, Instruct edition), which [Ollama](https://ollama.com) runs in its own container. Download the model once, about 6 GB; the same command checks that it sees images:

```bash
docker compose exec worker python -m vrag.tools.vlm_check --pull
```

Then ask:

```bash
curl -G localhost:8000/ask --data-urlencode 'q=what do they decide about the budget?'
```

The question is searched as above. Hits that follow each other in time are merged into one moment, and the model gets each moment's transcript, with the time of every line, and about three frames per hit. The response holds:

- `answer`: the text, which cites the moments it rests on as `[1]`, `[2]`;
- `moments`: each numbered moment with its video, its start and end in seconds, its transcript lines and the frames that were sent;
- `citations`: the numbers the answer used;
- `stats`: prompt and answer tokens, timings and the model's memory use.

Options: `video_id`, `mode` and `limit` (the hits to answer from, 5 by default) as for search, `frames=...` for the frames per hit (3 by default; 0 sends none), and `stream=true` to get the answer while it is written, as newline-delimited JSON events (`moments`, then `token` pieces, then `done`). From a terminal, `docker compose exec worker python -m vrag.answer "your question"` prints the answer as it arrives.

Settings in `.env`:

- `VLM_MODEL` picks another vision model from Ollama's library, such as `qwen3-vl:4b-instruct` for a smaller GPU.
- `VLM_NUM_CTX` is the context size in tokens (16384). Frames are left out when a prompt would not fit in it.
- `VLM_FRAME_SIDE` is the longer side, in pixels, that frames are scaled down to (768).
- `VLM_KEEP_ALIVE` is how long the model stays in memory after an answer (`5m`; `0` unloads it at once).
- `VLM_THINK=true` lets a Thinking edition, such as `qwen3-vl:8b`, reason before it answers.
- `OLLAMA_URL` points to another Ollama server, for example one that runs on the host (`http://host.docker.internal:11434` with Docker Desktop).
- `VLM_ENABLED=false` turns answers off.

## Evaluate retrieval

`backend/eval/queries.json` holds labeled queries. Each has a `category` (`speech`, `visual`, `speaker_emotion`, `small_object` or `multimodal`; speech and visual queries are searchable so far) and one or more target time spans:

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

`docker compose exec worker python -m vrag.evaluate` prints Recall@1/5/10 and MRR per category, once for each way of searching (`text`, `frame` and `fused`), and saves them in `data/eval/`. A hit counts when it overlaps a target, give or take `tolerance_s`. The file is built into the worker image, so rebuild after editing it (`docker compose up -d --build`) or pass another path, such as one under `/data`.

### Baseline results

The first measurement, with transcript search only (November 2026):

| Category | Queries | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| speech | 20 | 0.650 | 0.800 | 0.900 | 0.727 |

- **Data:** 5 clips, 17 minutes in total, with 20 labeled speech queries. Each query is searched across every indexed clip.
- **Setup:** the defaults: faster-whisper `small` transcripts (int8, CPU), windows of up to 20 seconds, `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dimensions, cosine similarity) and a tolerance of 2 seconds.
- **Reading:** the right moment was the top hit for 13 of the 20 queries, in the top 5 for 16 and in the top 10 for 18. Two queries had no correct hit in the top 10.
- **Limits:** this is a single run on a small set. Each query is worth 0.05 of recall, so only large differences between setups are meaningful, and only speech queries are covered so far.

This is good enough to build on for now. MiniLM is a small, English-only model, so if retrieval quality becomes the limit, the next options are a stronger embedding model such as [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), a re-ranker over the top hits, or both.
