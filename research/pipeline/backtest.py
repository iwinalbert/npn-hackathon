
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data_loader import M5Data
from .features import FeatureBuilder


@dataclass
class BacktestWindow:

    origin_idx: int
    horizon: int = config.HORIZON
    data: M5Data = field(repr=False, default=None)

    @property
    def train_cutoff_day(self) -> str:
        return f"d_{self.origin_idx + 1}"

    @property
    def target_day_idxs(self) -> np.ndarray:
        return self.origin_idx + 1 + np.arange(self.horizon)

    @property
    def first_target_day(self) -> str:
        return f"d_{self.origin_idx + 2}"

    @property
    def last_target_day(self) -> str:
        return f"d_{self.origin_idx + 1 + self.horizon}"

    def describe(self) -> dict:
        d = self.data
        return {
            "forecast_origin_day": self.train_cutoff_day,
            "forecast_origin_date": str(d.date_of(self.origin_idx).date()),
            "training_days_available": f"d_1 .. {self.train_cutoff_day}",
            "training_dates_available": (
                f"{d.date_of(0).date()} .. {d.date_of(self.origin_idx).date()}"
            ),
            "validation_days": f"{self.first_target_day} .. {self.last_target_day}",
            "validation_dates": (
                f"{d.date_of(self.target_day_idxs[0]).date()} .. "
                f"{d.date_of(self.target_day_idxs[-1]).date()}"
            ),
            "horizon_days": self.horizon,
            "validation_has_known_sales": bool(
                self.target_day_idxs.max() <= config.LAST_KNOWN_DAY_IDX
            ),
        }


class Backtester:

    def __init__(self, data: M5Data, feature_builder: FeatureBuilder | None = None):
        self.d = data
        self.fb = feature_builder or FeatureBuilder(data)


    def make_window(self, origin_idx: int, horizon: int = config.HORIZON) -> BacktestWindow:
        return BacktestWindow(origin_idx=origin_idx, horizon=horizon, data=self.d)


    def training_origins(
        self,
        validation_origin: int,
        n_origins: int,
        stride: int = config.HORIZON,
        horizon: int = config.HORIZON,
        min_origin: int = 400,
    ) -> list[int]:
        newest = validation_origin - horizon
        origins = [newest - i * stride for i in range(n_origins)]
        origins = [o for o in origins if o >= min_origin]
        return sorted(origins)


    def build_training_frame(
        self,
        origins: list[int],
        horizon: int = config.HORIZON,
        series_idx: np.ndarray | None = None,
        validation_origin: int | None = None,
    ) -> pd.DataFrame:
        frames = []
        for o in origins:
            f = self.fb.build_origin_frame(o, horizon=horizon, series_idx=series_idx,
                                           include_target=True)
            frames.append(f)

        train = pd.concat(frames, ignore_index=True)

        if validation_origin is not None:
            first_validation_day = validation_origin + 1
            max_train_target = int(train["target_day_idx"].max())
            if max_train_target >= first_validation_day:
                raise AssertionError(
                    f"LEAKAGE: a training row targets day index {max_train_target}, "
                    f"which is inside the validation window starting at "
                    f"{first_validation_day}."
                )

        if train["sales"].isna().any():
            raise AssertionError("training frame contains rows with unknown target sales")

        return train


    def build_validation_frame(
        self,
        validation_origin: int,
        horizon: int = config.HORIZON,
        series_idx: np.ndarray | None = None,
    ) -> pd.DataFrame:
        frame = self.fb.build_origin_frame(
            validation_origin, horizon=horizon, series_idx=series_idx, include_target=True
        )
        if frame["sales"].isna().any():
            raise AssertionError(
                "validation frame has unknown targets — the window runs past d_1941"
            )
        return frame


    def build_future_frame(
        self,
        origin_idx: int = config.FINAL_FORECAST_ORIGIN_IDX,
        horizon: int = config.HORIZON,
        series_idx: np.ndarray | None = None,
    ) -> pd.DataFrame:
        frame = self.fb.build_origin_frame(
            origin_idx, horizon=horizon, series_idx=series_idx, include_target=False
        )
        return frame
