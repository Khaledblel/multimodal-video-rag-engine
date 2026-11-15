"""Retrieval eval: Recall@K and MRR of search hits against labeled time spans, for each way of searching.

Usage: python -m vrag.evaluate [QUERIES] [--plan]   (default: eval/queries.json)

--plan sends every query through the query planner first, to compare with a run without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import ValidationError

from vrag import index, ollama, planner
from vrag.config import get_settings
from vrag.schemas.evalset import EvalSet, Target
from vrag.schemas.search import Hit, Plan

KS = (1, 5, 10)
MODES = ("text", "frame", "fused")


def _rank(hits: list[Hit], targets: list[Target], tolerance: float) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        for target in targets:
            if (
                hit.video_id == target.video_id
                and hit.start <= target.end + tolerance
                and hit.end >= target.start - tolerance
            ):
                return rank
    return None


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    found = [rank for rank in ranks if rank is not None]
    metrics = {f"R@{k}": sum(rank <= k for rank in found) / len(ranks) for k in KS}
    metrics["MRR"] = sum(1 / rank for rank in found) / len(ranks)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="?", type=Path, default=Path("eval/queries.json"))
    parser.add_argument("--plan", action="store_true", help="send every query through the query planner first")
    args = parser.parse_args()
    settings = get_settings()

    try:
        raw = args.queries.read_text()
        eval_set = EvalSet.model_validate_json(raw)
    except (OSError, ValidationError) as exc:
        print(f"INVALID {args.queries}\n{exc}")
        return 1
    if not eval_set.queries:
        print(f"no queries in {args.queries}")
        return 1
    for video_id in sorted({target.video_id for query in eval_set.queries for target in query.targets}):
        if index.count(video_id) == 0:
            print(f"warning: {video_id} has no indexed windows")

    modes = MODES if settings.frame_embed_enabled else ("text",)
    per_query = []
    # What the planner did over the whole set: how often it wrote the plan, changed the text, chose a video, failed.
    planned = {"planned": 0, "changed_text": 0, "chose_video": 0, "failed": 0}
    try:
        for query in eval_set.queries:
            plan = planner.plan(query.query, use=True) if args.plan else Plan(text=query.query)
            row = {"id": query.id, "category": query.category}
            if args.plan:
                row["plan"] = plan.model_dump()
                planned["planned"] += plan.planned
                planned["changed_text"] += plan.text != query.query
                planned["chose_video"] += plan.video_id is not None
                planned["failed"] += plan.error is not None
            for mode in modes:
                hits = index.search(plan.text, limit=max(KS), video_id=plan.video_id, mode=mode)
                row[mode] = _rank(hits, query.targets, eval_set.tolerance_s)
            per_query.append(row)
    except (ollama.OllamaError, httpx.HTTPError) as exc:
        # Text queries are embedded in Ollama, frame queries in the embedder service.
        print(f"evaluation stopped: {exc}")
        return 1

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for mode in modes:
        groups: dict[str, list[int | None]] = {}
        for row in per_query:
            groups.setdefault(row["category"], []).append(row[mode])
        groups["all"] = [row[mode] for row in per_query]
        metrics[mode] = {name: {"n": len(ranks), **_metrics(ranks)} for name, ranks in groups.items()}

    columns = [f"R@{k}" for k in KS] + ["MRR"]
    print(f"{index.collection_name()}: {len(per_query)} queries, tolerance {eval_set.tolerance_s} s")
    if args.plan:
        print(
            f"planner: wrote {planned['planned']} of {len(per_query)} plans, changed the text of "
            f"{planned['changed_text']}, chose a video for {planned['chose_video']}, failed on {planned['failed']}"
        )
    print(f"{'mode':<7}{'category':<16}{'n':>4}" + "".join(f"{column:>7}" for column in columns))
    for mode in modes:
        for name, row in metrics[mode].items():
            print(f"{mode:<7}{name:<16}{row['n']:>4}" + "".join(f"{row[column]:>7.3f}" for column in columns))
    for mode in modes:
        misses = [row["id"] for row in per_query if row[mode] is None]
        if misses:
            print(f"{mode}: not in top {max(KS)}: {', '.join(misses)}")

    finished = datetime.now(timezone.utc)
    out = settings.data_dir / "eval" / f"{finished:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "finished_at": finished.isoformat(),
        "queries_file": str(args.queries),
        "queries_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "collection": index.collection_name(),
        "text_model": settings.text_embed_model,
        "frame_model": settings.frame_embed_model if settings.frame_embed_enabled else None,
        "fusion": {
            "rrf_k": index.RRF_K,
            "text_weight": settings.fusion_text_weight,
            "frame_weight": settings.fusion_frame_weight,
        },
        "window_max_s": index.WINDOW_MAX_S,
        "window_gap_s": index.WINDOW_GAP_S,
        "tolerance_s": eval_set.tolerance_s,
        "planner": planned if args.plan else None,
        "modes": list(modes),
        "metrics": metrics,
        "per_query": per_query,
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
