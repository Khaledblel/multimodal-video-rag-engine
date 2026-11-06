"""Transcript windows embedded with a fastembed text model into Qdrant, one point per window."""

from __future__ import annotations

import re
import uuid
from functools import lru_cache

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from vrag.config import get_settings
from vrag.schemas.search import Hit
from vrag.schemas.timeline import Segment, VideoInfo, Window

WINDOW_MAX_S = 20.0
WINDOW_GAP_S = 2.0
UPSERT_BATCH = 64


def build_windows(segments: list[Segment]) -> list[Window]:
    groups: list[list[Segment]] = []
    for segment in segments:
        group = groups[-1] if groups else []
        if (
            group
            and segment.end - group[0].start <= WINDOW_MAX_S
            and segment.start - group[-1].end <= WINDOW_GAP_S
        ):
            group.append(segment)
        else:
            groups.append([segment])
    return [
        Window(
            id=i,
            start=group[0].start,
            end=max(segment.end for segment in group),
            text=" ".join(segment.text for segment in group),
        )
        for i, group in enumerate(groups)
    ]


@lru_cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(get_settings().text_embed_model)


@lru_cache
def _client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=30)


def collection_name() -> str:
    model = get_settings().text_embed_model
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return f"windows_{slug}_{TextEmbedding.get_embedding_size(model)}"


def _video_filter(video_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
    )


def count(video_id: str) -> int:
    client, name = _client(), collection_name()
    if not client.collection_exists(name):
        return 0
    return client.count(name, count_filter=_video_filter(video_id), exact=True).count


def replace(video: VideoInfo, windows: list[Window]) -> None:
    client, name = _client(), collection_name()
    if not client.collection_exists(name):
        size = TextEmbedding.get_embedding_size(get_settings().text_embed_model)
        client.create_collection(
            name, vectors_config={"text": models.VectorParams(size=size, distance=models.Distance.COSINE)}
        )
        client.create_payload_index(name, "video_id", models.PayloadSchemaType.KEYWORD)
        for field in ("start_time", "end_time"):
            client.create_payload_index(name, field, models.PayloadSchemaType.FLOAT)
    client.delete(name, points_selector=models.FilterSelector(filter=_video_filter(video.video_id)))
    if not windows:
        return

    vectors = _embedder().embed([window.text for window in windows])
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"vrag:{video.video_id}:{window.id}")),
            vector={"text": vector.tolist()},
            payload={
                "video_id": video.video_id,
                "source_name": video.source_name,
                "window_id": window.id,
                "start_time": window.start,
                "end_time": window.end,
                "text": window.text,
            },
        )
        for window, vector in zip(windows, vectors)
    ]
    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(name, points=points[i : i + UPSERT_BATCH])


def search(query: str, limit: int = 5, video_id: str | None = None) -> list[Hit]:
    client, name = _client(), collection_name()
    if not client.collection_exists(name):
        return []
    vector = next(iter(_embedder().query_embed(query)))
    response = client.query_points(
        name,
        query=vector.tolist(),
        using="text",
        query_filter=_video_filter(video_id) if video_id else None,
        limit=limit,
        with_payload=True,
    )
    return [
        Hit(
            video_id=point.payload["video_id"],
            source_name=point.payload.get("source_name"),
            window_id=point.payload["window_id"],
            start=point.payload["start_time"],
            end=point.payload["end_time"],
            text=point.payload["text"],
            score=point.score,
        )
        for point in response.points
    ]
