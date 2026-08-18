
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from typing import Annotated

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BACKEND_ROOT / "data"

RESEARCH_ROOT = REPO_ROOT / "research"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix="NPN_", env_file=".env", extra="ignore",
        env_nested_delimiter="__",
        validate_by_name=True,
    )

    app_name: str = "NPN Demand Forecasting API"
    version: str = "2.0.0"
    api_prefix: str = "/api/v1"
    environment: str = Field(default="development",
                             description="development | staging | production")

    data_dir: Path = DEFAULT_DATA_DIR

    project_root: Path = RESEARCH_ROOT
    model_direct: Path = (RESEARCH_ROOT / "models" / "champion"
                          / "model_11_blend_direct_final_forecast.txt")
    model_recursive: Path = (RESEARCH_ROOT / "models" / "champion"
                             / "model_12_blend_recursive_shape_final.txt")
    forecast_csv: Path = (RESEARCH_ROOT / "predictions" / "final_forecast"
                          / "final_forecast_28day_v3_diversity_blend.csv")

    n_series: int = 30_490
    horizon: int = 28
    forecast_origin_idx: int = 1940
    history_days_total: int = 1941

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

    enable_inference: bool = Field(
        default=True,
        description="Allow live model inference. Disable for a lean API-only "
                    "container without lightgbm/pandas installed.")
    inference_max_concurrent: int = 1
    inference_job_ttl_seconds: int = 3600
    inference_timeout_seconds: int = 600

    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NPN_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model: str = "gemini-3.7-flash"
    genai_enabled: bool = True
    genai_timeout_seconds: float = 30.0
    genai_max_question_chars: int = 800
    genai_max_output_tokens: int = 2048
    genai_thinking_budget: int = 0
    genai_temperature: float = 0.2

    log_level: str = "INFO"
    slow_request_ms: int = 1000

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v):
        if isinstance(v, str):
            raw = v.strip()
            if raw.startswith("["):
                import json
                return json.loads(raw)
            return [o.strip() for o in raw.split(",") if o.strip()]
        return v

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
        return 0.60

    @property
    def gemini_key_value(self) -> str | None:
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else None

    @property
    def genai_configured(self) -> bool:
        return bool(self.genai_enabled and self.gemini_key_value)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
