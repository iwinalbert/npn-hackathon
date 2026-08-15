"""
DEPLOYMENT READINESS — the tests that prove this runs somewhere other than the
machine it was written on.

The single most valuable test here is
`test_api_serves_with_no_research_tree_present`: it launches the app in a
subprocess whose environment points at a data directory containing only the
three product artefacts, with the project root redirected to an empty temp
directory. If the API can serve forecasts, hierarchy and accuracy under those
conditions, it can serve them from a container with the research data absent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.config import Settings, settings
from .conftest import API

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent


# ---------------------------------------------------------------------------
# Configuration is environment-driven and portable
# ---------------------------------------------------------------------------

def test_all_paths_are_configurable_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NPN_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("NPN_ENVIRONMENT", "production")
    monkeypatch.setenv("NPN_LOG_LEVEL", "WARNING")
    s = Settings()
    assert s.data_dir == tmp_path / "d"
    assert s.product_db == tmp_path / "d" / "product.duckdb"
    assert s.history_parquet == tmp_path / "d" / "history.parquet"
    assert s.backtest_parquet == tmp_path / "d" / "backtest.parquet"
    assert s.is_production is True


def test_cors_origins_accept_a_comma_separated_env_var(monkeypatch):
    monkeypatch.setenv("NPN_CORS_ORIGINS", "http://a.test, http://b.test")
    s = Settings()
    assert s.cors_origins == ["http://a.test", "http://b.test"]


def test_no_windows_paths_are_hard_coded_in_the_app():
    """A single `C:\\` in shipped code would break every Linux deployment."""
    offenders = []
    for p in (BACKEND / "app").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for marker in ("C:\\", "C:/", "\\Users\\", "/Users/"):
            if marker in text:
                offenders.append(f"{p.relative_to(BACKEND)}: {marker}")
    assert not offenders, f"hard-coded paths found: {offenders}"


def test_product_database_contains_no_baked_absolute_paths():
    """
    DuckDB bakes absolute paths into view SQL. A view over an external file
    would make the database file non-portable, which is why this build defines
    none.
    """
    import duckdb
    con = duckdb.connect(str(settings.product_db), read_only=True)
    try:
        views = con.execute(
            "SELECT view_name, sql FROM duckdb_views() WHERE NOT internal"
        ).fetchall()
    finally:
        con.close()
    bad = [v[0] for v in views
           if v[1] and ("C:/" in v[1] or "C:\\" in v[1] or "/Users/" in v[1])]
    assert not bad, f"views with baked absolute paths: {bad}"


def test_api_layer_imports_no_ml_libraries():
    """
    The API container must not need lightgbm/pandas/numpy. They are imported
    lazily inside the inference service only.
    """
    import ast
    eager = []
    for p in (BACKEND / "app").rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # module-level imports only: anything nested is lazy by construction
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ([a.name for a in node.names]
                         if isinstance(node, ast.Import)
                         else [node.module or ""])
                for n in names:
                    root = n.split(".")[0]
                    if root in {"lightgbm", "pandas", "numpy", "pipeline"}:
                        # is it at module level?
                        if node.col_offset == 0:
                            eager.append(f"{p.relative_to(BACKEND)}: {n}")
    assert not eager, f"eager ML imports would bloat the API container: {eager}"


def test_requirements_declare_every_api_dependency():
    req = (BACKEND / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("fastapi", "uvicorn", "pydantic", "pydantic-settings", "duckdb"):
        assert pkg in req, f"{pkg} missing from requirements.txt"
    # ML libraries belong in the inference extra, not the API runtime
    for pkg in ("lightgbm", "pandas", "numpy"):
        assert pkg not in req.split("# live inference")[0], (
            f"{pkg} should not be an API runtime dependency")
    inf = (BACKEND / "requirements-inference.txt").read_text(encoding="utf-8")
    assert "lightgbm==" in inf and "-r requirements.txt" in inf


def test_dockerfile_and_compose_exist_and_run_as_non_root():
    df = (BACKEND / "Dockerfile").read_text(encoding="utf-8")
    assert "USER app" in df, "container must not run as root"
    assert "HEALTHCHECK" in df
    assert "--target api" in df or "AS api" in df
    assert "AS full" in df
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert ":ro" in compose, "research artefacts must be mounted read-only"


# ---------------------------------------------------------------------------
# Graceful failure
# ---------------------------------------------------------------------------

def test_missing_data_layer_returns_503_not_a_crash(monkeypatch, tmp_path):
    """
    A container started before its data volume is mounted must report the
    problem, not die. 503 with a remedy is the correct behaviour.
    """
    from fastapi.testclient import TestClient

    import app.db as dbmod
    from app.config import Settings

    original = dbmod.settings
    dbmod.close_connection()
    try:
        dbmod.settings = Settings(data_dir=tmp_path / "empty")
        import app.main as mainmod
        with TestClient(mainmod.app) as c:
            # liveness still succeeds: the process is healthy
            assert c.get(f"{API}/health").status_code == 200
            ready = c.get(f"{API}/ready").json()
            assert ready["ready"] is False
            r = c.get(f"{API}/hierarchy/levels")
            assert r.status_code == 503
            assert r.json()["error"] == "service_unavailable"
    finally:
        dbmod.settings = original
        dbmod.close_connection()


# ---------------------------------------------------------------------------
# The portability proof
# ---------------------------------------------------------------------------

PORTABILITY_SCRIPT = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from fastapi.testclient import TestClient
    from app.main import app
    out = {}
    with TestClient(app) as c:
        out["ready"] = c.get("/api/v1/ready").json()["ready"]
        out["levels"] = len(c.get("/api/v1/hierarchy/levels").json())
        f = c.get("/api/v1/series/CA_3/FOODS_3_090/forecast").json()
        out["forecast_points"] = len(f["forecast"])
        out["total_28d"] = f["total_28d"]
        agg = c.get("/api/v1/hierarchy/aggregate",
                    params={"level": "total", "node_id": "ALL"}).json()
        out["chain_total"] = round(agg["total_28d"], 2)
        out["accuracy_windows"] = len(c.get("/api/v1/accuracy/windows").json())
        h = c.get("/api/v1/series/CA_3/FOODS_3_090/history",
                  params={"days": 30}).json()
        out["history_points"] = len(h["history"])
        inf = c.get("/api/v1/inference/status").json()
        out["inference_available"] = inf["available"]
        out["inference_reasons"] = inf["reasons"]
    print("RESULT_JSON:" + json.dumps(out))
""")


@pytest.mark.slow
def test_api_serves_with_no_research_tree_present(tmp_path):
    """
    THE DEPLOYMENT PROOF.

    Copies only the three product artefacts to a fresh directory, points the
    project root at an empty directory, and runs the API in a clean subprocess.
    Everything except live inference must work — which is exactly the contract
    the `api` Docker target promises.
    """
    data_dir = tmp_path / "product"
    data_dir.mkdir()
    for name in ("product.duckdb", "history.parquet", "backtest.parquet"):
        src = settings.data_dir / name
        if not src.exists():
            pytest.skip(f"{name} not built")
        shutil.copy2(src, data_dir / name)

    fake_root = tmp_path / "no_research_here"
    fake_root.mkdir()

    script = tmp_path / "probe.py"
    script.write_text(PORTABILITY_SCRIPT, encoding="utf-8")

    env = {k: v for k, v in os.environ.items()
           if not k.startswith("NPN_")}
    env.update({
        "NPN_DATA_DIR": str(data_dir),
        "NPN_PROJECT_ROOT": str(fake_root),
        "NPN_MODEL_DIRECT": str(fake_root / "missing_direct.txt"),
        "NPN_MODEL_RECURSIVE": str(fake_root / "missing_recursive.txt"),
        "NPN_FORECAST_CSV": str(fake_root / "missing_forecast.csv"),
        "NPN_ENVIRONMENT": "production",
        "PYTHONPATH": str(BACKEND),
    })

    proc = subprocess.run(
        [sys.executable, str(script), str(BACKEND)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"API failed without the research tree:\n{proc.stdout}\n{proc.stderr}")
    line = next(ln for ln in proc.stdout.splitlines()
                if ln.startswith("RESULT_JSON:"))
    out = json.loads(line[len("RESULT_JSON:"):])

    # Everything that does not need the model must work
    assert out["ready"] is True
    assert out["levels"] == 12
    assert out["forecast_points"] == 28
    assert out["total_28d"] > 0
    assert out["chain_total"] > 1_000_000
    assert out["accuracy_windows"] == 8
    assert out["history_points"] == 30

    # ...and inference must degrade gracefully, explaining itself
    assert out["inference_available"] is False
    assert any("not found" in r for r in out["inference_reasons"])
