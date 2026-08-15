"""
Application settings.

Every path is configurable via environment variable so the same image runs
locally and in a container where the research tree is mounted elsewhere.
Defaults resolve relative to the project root, which is found by walking up from
this file — the same convention the research scripts use, so the two layers
agree on where the project is without either importing the other.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 06_BACKEND/app/config.py -> 06_BACKEND/app -> 06_BACKEND -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Override any field with an NPN_ prefixed env var."""

    model_config = SettingsConfigDict(
        env_prefix="NPN_", env_file=".env", extra="ignore"
    )

    app_name: str = "NPN Demand Forecasting API"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # --- paths -----------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    product_db: Path = PROJECT_ROOT / "06_BACKEND" / "data" / "product.duckdb"
    panel_parquet: Path = (PROJECT_ROOT / "data" / "processed"
                           / "sales_long_full.parquet")
    calendar_csv: Path = PROJECT_ROOT / "data" / "raw" / "calendar.csv"

    # --- frozen model artefacts (read-only; used by the worker only) ------
    model_direct: Path = (PROJECT_ROOT / "models" / "champion"
                          / "model_11_blend_direct_final_forecast.txt")
    model_recursive: Path = (PROJECT_ROOT / "models" / "champion"
                             / "model_12_blend_recursive_shape_final.txt")
    forecast_csv: Path = (PROJECT_ROOT / "predictions" / "final_forecast"
                          / "final_forecast_28day_v3_diversity_blend.csv")

    # --- domain constants (mirror pipeline/config.py; see ADR in README) --
    n_series: int = 30_490
    horizon: int = 28
    forecast_origin_idx: int = 1940      # ZERO-BASED index of d_1941
    history_days: int = 1941

    # --- behaviour --------------------------------------------------------
    cors_origins: list[str] = [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://localhost:3000",
    ]
    cache_ttl_seconds: int = 300
    max_list_limit: int = 500
    default_history_days: int = 90
    enable_inference: bool = True

    @property
    def blend_weight_direct(self) -> float:
        """The frozen blend weight. Constant by definition — see MODEL_FREEZE.md."""
        return 0.60


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
