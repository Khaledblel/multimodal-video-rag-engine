"""Windows over a video as Qdrant points, each with a text vector and frame vectors, searched by either or both."""

from __future__ import annotations

import math
import re
import uuid
from functools import lru_cache

import httpx
import numpy as np
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from vrag.config import get_settings
from vrag.schemas.search import Hit
from vrag.schemas.timeline import Frame, Segment, VideoInfo, Window

WINDOW_MAX_S = 20.0
WINDOW_GAP_S = 2.0
FRAME_PAD_S = 1.0
UPSERT_BATCH = 16
RRF_K = 60
MODES = ("fused", "text", "frame")


def build_windows(segments: list[Segment], duration_s: float | None = None) -> list[Window]:
    """Group speech into windows; given the video's duration, also cover the stretches without speech."""
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
    spans = [
        (group[0].start, max(segment.end for segment in group), " ".join(segment.text for segment in group))
        for group in groups
    ]
    if duration_s is not None:
        # The edges alternate between where a stretch without speech starts and where it ends.
        edges = [0.0, *(edge for start, end, _ in spans for edge in (start, end)), duration_s]
        for gap_start, gap_end in zip(edges[::2], edges[1::2]):
            length = gap_end - gap_start
            if length >= WINDOW_GAP_S:
                pieces = math.ceil(length / WINDOW_MAX_S)
                spans += [
                    (gap_start + i * length / pieces, gap_start + (i + 1) * length / pieces, "")
                    for i in range(pieces)
                ]
        spans.sort()
    return [
        Window(id=i, start=round(start, 3), end=round(end, 3), text=text)
        for i, (start, end, text) in enumerate(spans)
    ]


def span_frames(start: float, end: float, frames: list[Frame]) -> list[int]:
    """Positions of the frames of a time span: those inside it, padded so that even a very short span has one."""
    rows = [i for i, frame in enumerate(frames) if start - FRAME_PAD_S <= frame.t < end + FRAME_PAD_S]
    if not rows and frames:
        middle = (start + end) / 2
        rows = [min(range(len(frames)), key=lambda i: abs(frames[i].t - middle))]
    return rows


@lru_cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(get_settings().text_embed_model)


@lru_cache
def _client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url, timeout=30)


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _text_size() -> int:
    return TextEmbedding.get_embedding_size(get_settings().text_embed_model)


def collection_name() -> str:
    """One collection per set of embedding models, so vectors from different models never mix."""
    settings = get_settings()
    name = f"windows_{_slug(settings.text_embed_model)}_{_text_size()}"
    if settings.frame_embed_enabled:
        name += f"_{_slug(settings.frame_embed_model)}_{settings.frame_embed_dim}"
    return name


def _video_filter(video_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
    )


def count(video_id: str) -> int:
    client, name = _client(), collection_name()
    if not client.collection_exists(name):
        return 0
    return client.count(name, count_filter=_video_filter(video_id), exact=True).count


def _ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        return
    settings = get_settings()
    vectors = {"text": models.VectorParams(size=_text_size(), distance=models.Distance.COSINE)}
    if settings.frame_embed_enabled:
        # One vector per frame; a query scores a window by its best-matching frame.
        vectors["frame"] = models.VectorParams(
            size=settings.frame_embed_dim,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
        )
    client.create_collection(name, vectors_config=vectors)
    client.create_payload_index(name, "video_id", models.PayloadSchemaType.KEYWORD)
    for field in ("start_time", "end_time"):
        client.create_payload_index(name, field, models.PayloadSchemaType.FLOAT)


def replace(
    video: VideoInfo,
    windows: list[Window],
    frames: list[Frame] | None = None,
    frame_vectors: np.ndarray | None = None,
) -> None:
    """Replace a video's points. frame_vectors has one row per frame, in the order of frames."""
    client, name = _client(), collection_name()
    _ensure_collection(client, name)
    client.delete(name, points_selector=models.FilterSelector(filter=_video_filter(video.video_id)))

    spoken = [window for window in windows if window.text]
    embedded = _embedder().embed([window.text for window in spoken]) if spoken else []
    text_vectors = {window.id: vector for window, vector in zip(spoken, embedded)}
    points = []
    for window in windows:
        # A point carries only the vectors its window has: a stretch without speech has no text vector.
        vector: dict[str, list] = {}
        if window.id in text_vectors:
            vector["text"] = text_vectors[window.id].tolist()
        if frame_vectors is not None and frames:
            rows = span_frames(window.start, window.end, frames)
            vector["frame"] = frame_vectors[rows].astype(np.float32).tolist()
        if not vector:
            continue
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"vrag:{video.video_id}:{window.id}")),
                vector=vector,
                payload={
                    "video_id": video.video_id,
                    "source_name": video.source_name,
                    "window_id": window.id,
                    "start_time": window.start,
                    "end_time": window.end,
                    "text": window.text,
                },
            )
        )
    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(name, points=points[i : i + UPSERT_BATCH])


def _frame_query(query: str) -> list[float]:
    response = httpx.post(
        f"{get_settings().embedder_url}/embed/frame-text",
        json={"texts": [query]},
        timeout=httpx.Timeout(10.0, read=300.0),
    )
    response.raise_for_status()
    return response.json()["vectors"][0]


def search(query: str, limit: int = 5, video_id: str | None = None, mode: str = "fused") -> list[Hit]:
    """Search by what is said ("text"), by what is shown ("frame"), or by both ("fused")."""
    settings = get_settings()
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if mode == "frame" and not settings.frame_embed_enabled:
        raise ValueError("frame search is turned off (FRAME_EMBED_ENABLED=false)")
    client, name = _client(), collection_name()
    if not client.collection_exists(name):
        return []

    queries: dict[str, list] = {}
    if mode != "frame":
        queries["text"] = next(iter(_embedder().query_embed(query))).tolist()
    if mode != "text" and settings.frame_embed_enabled:
        queries["frame"] = [_frame_query(query)]
    fusing = len(queries) > 1
    depth = max(4 * limit, 20) if fusing else limit
    weights = {"text": settings.fusion_text_weight, "frame": settings.fusion_frame_weight}

    points: dict[str, models.ScoredPoint] = {}
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, float] = {}
    for source, vector in queries.items():
        found = client.query_points(
            name,
            query=vector,
            using=source,
            query_filter=_video_filter(video_id) if video_id else None,
            limit=depth,
            with_payload=True,
        ).points
        for rank, point in enumerate(found, start=1):
            points[point.id] = point
            ranks.setdefault(point.id, {})[source] = rank
            if fusing:
                # Weighted reciprocal rank fusion: the two searches score on different scales, so only
                # their ranks are combined. A window found by both gets both contributions.
                scores[point.id] = scores.get(point.id, 0.0) + weights[source] / (RRF_K + rank)
            else:
                scores[point.id] = point.score
    best = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [
        Hit(
            video_id=points[i].payload["video_id"],
            source_name=points[i].payload.get("source_name"),
            window_id=points[i].payload["window_id"],
            start=points[i].payload["start_time"],
            end=points[i].payload["end_time"],
            text=points[i].payload["text"],
            score=scores[i],
            text_rank=ranks[i].get("text"),
            frame_rank=ranks[i].get("frame"),
        )
        for i in best
    ]
