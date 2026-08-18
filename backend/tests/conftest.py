"""Shared test fixtures."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)

from fastapi.testclient import TestClient          # noqa: E402
from app.config import settings                    # noqa: E402
from app.main import app                           # noqa: E402

API = settings.api_prefix


@pytest.fixture(scope="session")
def client() -> TestClient:
    if not Path(settings.product_db).exists():
        pytest.skip("product.duckdb not built — run scripts/build_product_db.py")
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_series(client) -> dict:
    """A real, high-volume series that exists in every environment."""
    r = client.get(f"{API}/series", params={"limit": 1})
    assert r.status_code == 200
    rows = r.json()
    assert rows, "no series returned"
    return rows[0]
