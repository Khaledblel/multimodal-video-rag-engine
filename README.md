# Multimodal Video RAG Engine

Ask questions about your videos in natural language. Speech transcripts and video frames are aligned on a single timeline and indexed in a vector database, so a question returns the matching moments and an answer grounded in what was said and shown.

> Early development: only the API skeleton exists so far. Ingestion, search and answering are being added step by step.

## Quick start

Requires Docker with Compose.

```bash
mkdir -p data
docker compose up -d --build
curl localhost:8000/health
```

Stop with `docker compose down`. To change the API port or log level, copy `.env.example` to `.env` and edit it.
