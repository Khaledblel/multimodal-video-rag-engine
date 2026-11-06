"""Retrieval eval: Recall@K and MRR of search hits against labeled time spans.

Usage: python -m vrag.evaluate [QUERIES]   (default: eval/queries.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from vrag import index
from vrag.config import get_settings
from vrag.schemas.evalset import EvalSet, Target
from vrag.schemas.search import Hit

KS = (1, 5, 10)


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
    args = parser.parse_args()

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

    per_query = [
        {
            "id": query.id,
            "category": query.category,
            "rank": _rank(index.search(query.query, limit=max(KS)), query.targets, eval_set.tolerance_s),
        }
        for query in eval_set.queries
    ]
    groups: dict[str, list[int | None]] = {}
    for row in per_query:
        groups.setdefault(row["category"], []).append(row["rank"])
    groups["all"] = [row["rank"] for row in per_query]
    metrics = {name: {"n": len(ranks), **_metrics(ranks)} for name, ranks in groups.items()}

    columns = [f"R@{k}" for k in KS] + ["MRR"]
    print(f"{index.collection_name()}: {len(per_query)} queries, tolerance {eval_set.tolerance_s} s")
    print(f"{'category':<16}{'n':>4}" + "".join(f"{column:>7}" for column in columns))
    for name, row in metrics.items():
        print(f"{name:<16}{row['n']:>4}" + "".join(f"{row[column]:>7.3f}" for column in columns))
    misses = [row["id"] for row in per_query if row["rank"] is None]
    if misses:
        print(f"not in top {max(KS)}: {', '.join(misses)}")

    finished = datetime.now(timezone.utc)
    out = get_settings().data_dir / "eval" / f"{finished:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "finished_at": finished.isoformat(),
        "queries_file": str(args.queries),
        "queries_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "collection": index.collection_name(),
        "text_model": get_settings().text_embed_model,
        "window_max_s": index.WINDOW_MAX_S,
        "window_gap_s": index.WINDOW_GAP_S,
        "tolerance_s": eval_set.tolerance_s,
        "metrics": metrics,
        "per_query": per_query,
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
