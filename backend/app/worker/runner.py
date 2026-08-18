
from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable

from ..config import settings

_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_running = 0


class JobRejected(RuntimeError):
    pass


def _prune_locked() -> None:
    now = time.time()
    ttl = settings.inference_job_ttl_seconds
    stale = [jid for jid, j in _jobs.items()
             if j["status"] in ("succeeded", "failed")
             and now - j.get("finished_at", now) > ttl]
    for jid in stale:
        _jobs.pop(jid, None)


def submit(kind: str, fn: Callable[..., dict], **kwargs) -> str:
    global _running
    with _lock:
        _prune_locked()
        if _running >= settings.inference_max_concurrent:
            raise JobRejected(
                f"an inference job is already running "
                f"(limit {settings.inference_max_concurrent}). "
                "Poll the existing job or retry when it completes.")
        job_id = uuid.uuid4().hex[:16]
        _jobs[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 0,
            "message": "queued",
            "submitted_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        _running += 1

    def _progress(pct: int, message: str) -> None:
        with _lock:
            job = _jobs.get(job_id)
            if job is not None and job["status"] == "running":
                job["progress"] = max(0, min(100, int(pct)))
                job["message"] = message

    def _target() -> None:
        global _running
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = time.time()
            _jobs[job_id]["message"] = "starting"
        try:
            result = fn(progress=_progress, **kwargs)
            with _lock:
                _jobs[job_id].update(
                    status="succeeded", progress=100, message="complete",
                    result=result, finished_at=time.time())
        except Exception as exc:                               # noqa: BLE001
            tb = traceback.format_exc(limit=8)
            with _lock:
                _jobs[job_id].update(
                    status="failed", message=f"{type(exc).__name__}: {exc}",
                    error={"type": type(exc).__name__, "message": str(exc),
                           "traceback_tail": tb[-2000:]},
                    finished_at=time.time())
        finally:
            with _lock:
                _running = max(0, _running - 1)

    threading.Thread(target=_target, name=f"job-{kind}-{job_id}",
                     daemon=True).start()
    return job_id


def get(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with _lock:
        _prune_locked()
        jobs = sorted(_jobs.values(), key=lambda j: j["submitted_at"],
                      reverse=True)
        return [{k: v for k, v in j.items() if k != "result"}
                for j in jobs[:limit]]


def stats() -> dict[str, Any]:
    with _lock:
        return {
            "running": _running,
            "max_concurrent": settings.inference_max_concurrent,
            "tracked_jobs": len(_jobs),
            "job_ttl_seconds": settings.inference_job_ttl_seconds,
            "durable": False,
            "note": ("Jobs are in-process: they do not survive a restart and "
                     "are not shared across replicas."),
        }


def reset() -> None:
    global _running
    with _lock:
        _jobs.clear()
        _running = 0
