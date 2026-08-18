from __future__ import annotations

from fastapi import APIRouter, Query

from ..errors import Conflict, NotFound
from ..services import inference as svc
from ..worker import runner

router = APIRouter(prefix="/inference", tags=["inference"])


@router.get("/status", summary="Can live inference run in this deployment?")
def status() -> dict:
    a = svc.availability()
    return {**a, "runtime": svc.model_runtime_info(), "jobs": runner.stats()}


@router.post("/verify", summary="Start a forecast verification run (~35 s)")
def verify() -> dict:
    svc.require_available()
    try:
        job_id = runner.submit("verify", svc.run_verification)
    except runner.JobRejected as exc:
        raise Conflict(str(exc)) from exc
    return {
        "job_id": job_id,
        "status": "queued",
        "poll": f"/api/v1/inference/jobs/{job_id}",
        "estimated_seconds": 35,
    }


@router.get("/jobs", summary="Recent inference jobs")
def jobs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return runner.list_jobs(limit)


@router.get("/jobs/{job_id}", summary="Poll one inference job")
def job(job_id: str) -> dict:
    j = runner.get(job_id)
    if j is None:
        raise NotFound(f"no job with id '{job_id}'",
                       hint="Jobs are in-process and do not survive a restart.")
    return j
