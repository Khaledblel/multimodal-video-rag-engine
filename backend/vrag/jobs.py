"""Ingest jobs: the API queues a video as a file in data/jobs/, and the worker runs the jobs one at a time.

Run in the worker: python -m vrag.jobs
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from vrag.config import get_settings
from vrag.schemas.jobs import Job

log = logging.getLogger("vrag.jobs")

POLL_S = 1.0
JOB_ID = r"[0-9]{8}T[0-9]{6}-[0-9a-f]{6}"
STAGES = ("probe", "audio", "frames", "transcript", "frame_vectors", "index", "proxy")


def _dir() -> Path:
    return get_settings().data_dir / "jobs"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def save(job: Job) -> None:
    path = _dir() / f"{job.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(job.model_dump_json(indent=2))
    os.replace(tmp, path)


def load(job_id: str) -> Job | None:
    if not re.fullmatch(JOB_ID, job_id):
        return None
    try:
        return Job.model_validate_json((_dir() / f"{job_id}.json").read_text())
    except (OSError, ValidationError):
        return None


def latest(limit: int = 50) -> list[Job]:
    """The newest jobs first. An id starts with its time of creation, so file names sort by age."""
    paths = sorted(_dir().glob("*.json"), reverse=True)[:limit]
    return [job for job in (load(path.stem) for path in paths) if job is not None]


def create(video: Path, source_name: str) -> Job:
    """Queue the ingest of a video file."""
    now = _now()
    job = Job(
        id=f"{now:%Y%m%dT%H%M%S}-{secrets.token_hex(3)}",
        source_name=source_name,
        path=str(video),
        stages=list(STAGES),
        created_at=now,
    )
    save(job)
    return job


def update(job_id: str, **changes: object) -> Job | None:
    job = load(job_id)
    if job is None:
        return None
    job = job.model_copy(update=changes)
    save(job)
    return job


def _waiting() -> Job | None:
    """The oldest job that has not started."""
    for path in sorted(_dir().glob("*.json")):
        job = load(path.stem)
        if job is not None and job.status == "queued":
            return job
    return None


def _run(job: Job) -> None:
    log.info("job %s: ingesting %s", job.id, job.path)
    update(job.id, status="running", started_at=_now(), stage=None, error=None)
    code = subprocess.run([sys.executable, "-m", "vrag.ingest", job.path, "--job", job.id]).returncode
    if code == 0:
        update(job.id, status="done", stage=None, finished_at=_now())
        log.info("job %s: done", job.id)
        return
    ended = load(job.id)
    error = (ended.error if ended else None) or f"the ingest stopped with exit code {code}"
    update(job.id, status="failed", finished_at=_now(), error=error)
    log.warning("job %s: failed: %s", job.id, error)


def main() -> int:
    logging.basicConfig(level=get_settings().log_level.upper(), format="%(message)s")
    _dir().mkdir(parents=True, exist_ok=True)
    for job in latest(limit=10_000):
        if job.status == "running":
            update(job.id, status="queued", stage=None)
    log.info("waiting for ingest jobs in %s", _dir())
    while True:
        job = _waiting()
        if job is None:
            time.sleep(POLL_S)
        else:
            _run(job)


if __name__ == "__main__":
    raise SystemExit(main())
