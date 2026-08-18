from __future__ import annotations

from fastapi import APIRouter

from ..services import meta as svc

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get("/model", summary="Frozen model card")
def model_card() -> dict:
    return svc.model_card()


@router.get("/capabilities",
            summary="What is implemented, what research rejected, what is unsupported")
def capabilities() -> dict:
    return svc.capabilities()


@router.get("/provenance", summary="Source and hash of every served artefact")
def provenance() -> dict:
    return svc.provenance()
