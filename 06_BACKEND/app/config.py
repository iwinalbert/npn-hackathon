"""
Application settings.

DEPLOYMENT CONTRACT
-------------------
Every path is an environment-overridable setting. Defaults resolve relative to
the project root so local development needs no configuration, but in a container
each one is set explicitly and nothing depends on the checkout layout.

The API's *required* runtime artefacts are only the three product-owned files in
`data_dir`:

    product.duckdb    history.parquet    backtest.parquet

The research tree (`models/`, `data/raw/`, `predictions/`, ...) is NOT required
to serve the API. It is required only for the optional inference worker, which
degrades gracefully when it is absent — see `app/services/inference.py`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# 06_BACKEND/app/config.py -> 06_BACKEND/app -> 06_BACKEND -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "06_BACKEND" / "data"


class Settings(BaseSettings):
    """Runtime configuration. Override any field with an NPN_-prefixed env var."""

    model_config = SettingsConfigDict(
        env_prefix="NPN_", env_file=".env", extra="ignore",
        env_nested_delimiter="__",
    )

    app_name: str = "NPN Demand Forecasting API"
    version: str = "2.0.0"
    api_prefix: str = "/api/v1"
    environment: str = Field(default="development",
                             description="development | staging | production")

    # --- product data layer (REQUIRED at runtime) ------------------------
    data_dir: Path = DEFAULT_DATA_DIR

    # --- research artefacts (OPTIONAL; inference worker only) ------------
    project_root: Path = PROJECT_ROOT
    model_direct: Path = (PROJECT_ROOT / "models" / "champion"
                          / "model_11_blend_direct_final_forecast.txt")
    model_recursive: Path = (PROJECT_ROOT / "models" / "champion"
                             / "model_12_blend_recursive_shape_final.txt")
    forecast_csv: Path = (PROJECT_ROOT / "predictions" / "final_forecast"
                          / "final_forecast_28day_v3_diversity_blend.csv")

    # --- domain constants (mirror pipeline/config.py) --------------------
    # day_idx is ZERO-BASED: 1940 == d_1941 == 2016-05-22 == forecast origin.
    n_series: int = 30_490
    horizon: int = 28
    forecast_origin_idx: int = 1940
    history_days_total: int = 1941

    # --- behaviour --------------------------------------------------------
    # NoDecode: take the raw env string and parse it ourselves, so both
    # 'a,b' and '["a","b"]' work. Without it pydantic-settings tries
    # JSON first and raises before the validator is reached.
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://localhost:3000",
    ]
    cors_allow_all: bool = Field(
        default=False,
        description="Development escape hatch. Never enable in production.")
    cache_ttl_seconds: int = 300
    max_list_limit: int = 500
    default_history_days: int = 90
    max_history_days: int = 1941

    # --- inference worker -------------------------------------------------
    enable_inference: bool = Field(
        default=True,
        description="Allow live model inference. Disable for a lean API-only "
                    "container without lightgbm/pandas installed.")
    inference_max_concurrent: int = 1
    inference_job_ttl_seconds: int = 3600
    inference_timeout_seconds: int = 600

    # --- GenAI assistant --------------------------------------------------
    # The key is a SecretStr so it cannot be printed, logged or serialised by
    # accident: repr() and str() both render "**********". It is read from the
    # environment (NPN_GEMINI_API_KEY) or the plain GEMINI_API_KEY alias, and it
    # never leaves this process — the browser never sees it.
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.0-flash"
    genai_enabled: bool = True
    genai_timeout_seconds: float = 30.0
    genai_max_question_chars: int = 800
    genai_max_output_tokens: int = 900
    genai_temperature: float = 0.2

    # --- observability ----------------------------------------------------
    log_level: str = "INFO"
    slow_request_ms: int = 1000

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v):
        """Accept a JSON list or a comma-separated string from the environment."""
        if isinstance(v, str):
            raw = v.strip()
            if raw.startswith("["):
                import json
                return json.loads(raw)
            return [o.strip() for o in raw.split(",") if o.strip()]
        return v

    # --- derived paths ----------------------------------------------------
    @property
    def product_db(self) -> Path:
        return self.data_dir / "product.duckdb"

    @property
    def history_parquet(self) -> Path:
        return self.data_dir / "history.parquet"

    @property
    def backtest_parquet(self) -> Path:
        return self.data_dir / "backtest.parquet"

    @property
    def required_artefacts(self) -> dict[str, Path]:
        return {
            "product_db": self.product_db,
            "history_parquet": self.history_parquet,
            "backtest_parquet": self.backtest_parquet,
        }

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def blend_weight_direct(self) -> float:
        """The frozen blend weight. Constant by definition — MODEL_FREEZE.md."""
        return 0.60

    @property
    def gemini_key_value(self) -> str | None:
        """
        The raw key, for the one place that needs it: constructing the client.

        Nothing else in the codebase should call this, and no response schema
        contains it. A test asserts the key never appears in any API response.
        """
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else None

    @property
    def genai_configured(self) -> bool:
        return bool(self.genai_enabled and self.gemini_key_value)


@lru_cache
def get_settings() -> Settings:
    """
    Build settings, accepting the plain `GEMINI_API_KEY` environment variable in
    addition to the prefixed `NPN_GEMINI_API_KEY`.

    The unprefixed name is what Google's own tooling and most hosting providers
    use, so supporting it avoids a class of "the key is set but the app says it
    is missing" deployment confusion.
    """
    import os

    overrides: dict[str, object] = {}
    if not os.environ.get("NPN_GEMINI_API_KEY"):
        plain = os.environ.get("GEMINI_API_KEY", "").strip()
        if plain:
            overrides["gemini_api_key"] = plain
    return Settings(**overrides)


settings = get_settings()
