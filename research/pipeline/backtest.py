"""
Fixed-origin 28-day backtesting framework.

WHAT A BACKTEST IS, IN PLAIN ENGLISH
------------------------------------
We cannot score a model on the real forecast window (2016-05-23 to 2016-06-19),
because nobody has those sales — that is precisely what we must predict. So we
rewind: we pretend an earlier day was "today", forecast the 28 days after it, and
compare against the sales we already have on record for those days.

WHY RANDOM TRAIN/TEST SPLITTING WOULD BE WRONG
----------------------------------------------
A random split would let the model train on 2016-05-10 while being tested on
2016-04-30 — learning from the future to explain the past. It would also let a
series' own later behaviour bleed into its earlier rows. The resulting score would
look excellent and mean nothing. Time-series validation must always cut on TIME.

THE THREE DATA BLOCKS, KEPT STRICTLY SEPARATE
---------------------------------------------
    TRAINING DATA   every observation on or before the training cutoff
    VALIDATION DATA the 28 real, already-observed days after the cutoff.
                    Used only to score. Never trained on. Never used to build a
                    feature.
    FINAL FORECAST  2016-05-23 .. 2016-06-19. Genuinely unknown; no truth exists
                    anywhere. Produced once, at the very end.

A subtlety that is easy to get wrong: it is not enough for the training TARGETS to
sit before the cutoff. The training FEATURES must also be buildable from data
before the cutoff. That is why every training origin here satisfies
    origin + horizon <= validation_origin
so no training row's 28-day target block can ever reach into validation days.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data_loader import M5Data
from .features import FeatureBuilder


@dataclass
class BacktestWindow:
    """One fixed-origin 28-day evaluation window."""

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
    """
    Assembles leakage-safe training and validation frames around a chosen origin.
    """

    def __init__(self, data: M5Data, feature_builder: FeatureBuilder | None = None):
        self.d = data
        self.fb = feature_builder or FeatureBuilder(data)

    # ------------------------------------------------------------------

    def make_window(self, origin_idx: int, horizon: int = config.HORIZON) -> BacktestWindow:
        return BacktestWindow(origin_idx=origin_idx, horizon=horizon, data=self.d)

    # ------------------------------------------------------------------

    def training_origins(
        self,
        validation_origin: int,
        n_origins: int,
        stride: int = config.HORIZON,
        horizon: int = config.HORIZON,
        min_origin: int = 400,
    ) -> list[int]:
        """
        Choose the historical origins that training rows are built from.

        The newest permitted training origin is `validation_origin - horizon`.
        Anything later would produce a training row whose 28-day target block
        overlaps the validation window — i.e. the model would be fitted on the
        very days it is about to be scored on.

        `min_origin` keeps origins far enough into the history that the longest
        lookback (lag_28 / rolling_mean_28) has real data behind it rather than
        the artificial start of the panel.
        """
        newest = validation_origin - horizon
        origins = [newest - i * stride for i in range(n_origins)]
        origins = [o for o in origins if o >= min_origin]
        return sorted(origins)

    # ------------------------------------------------------------------

    def build_training_frame(
        self,
        origins: list[int],
        horizon: int = config.HORIZON,
        series_idx: np.ndarray | None = None,
        validation_origin: int | None = None,
    ) -> pd.DataFrame:
        """
        Stack per-origin feature frames into one training table.

        If `validation_origin` is supplied, this asserts that no training row's
        target day can fall on or after the first validation day. That assertion
        is the last line of defence against the most damaging possible bug in the
        project, so it is checked here rather than assumed.
        """
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

        # Training rows must have a real observed target.
        if train["sales"].isna().any():
            raise AssertionError("training frame contains rows with unknown target sales")

        return train

    # ------------------------------------------------------------------

    def build_validation_frame(
        self,
        validation_origin: int,
        horizon: int = config.HORIZON,
        series_idx: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        The scoring frame: features built standing at `validation_origin`, targets
        being the 28 real days that follow.
        """
        frame = self.fb.build_origin_frame(
            validation_origin, horizon=horizon, series_idx=series_idx, include_target=True
        )
        if frame["sales"].isna().any():
            raise AssertionError(
                "validation frame has unknown targets — the window runs past d_1941"
            )
        return frame

    # ------------------------------------------------------------------

    def build_future_frame(
        self,
        origin_idx: int = config.FINAL_FORECAST_ORIGIN_IDX,
        horizon: int = config.HORIZON,
        series_idx: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        The real, genuinely-unknown forecast window: d_1942 .. d_1969
        (2016-05-23 .. 2016-06-19).

        Built exactly like a validation frame, with one difference: there is no
        target to attach, because no file on earth contains those sales. Calendar,
        event, SNAP and sell_price for these days ARE available and are used —
        that is legitimate future-known context, not leakage.
        """
        frame = self.fb.build_origin_frame(
            origin_idx, horizon=horizon, series_idx=series_idx, include_target=False
        )
        return frame
