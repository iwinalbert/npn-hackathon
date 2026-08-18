
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .features import FeatureBuilder, FEATURE_GROUPS


class FeatureBuilderV2(FeatureBuilder):

    def _extra_demand(self, origin: int) -> dict[str, np.ndarray]:
        s = self.d.sales_wide
        out: dict[str, np.ndarray] = {}

        start = max(0, origin - 13)
        w14 = s[:, start:origin + 1].astype(np.float64)
        out["rolling_mean_14"] = w14.mean(axis=1).astype(np.float32)
        out["rolling_std_14"] = w14.std(axis=1).astype(np.float32)

        w7 = s[:, max(0, origin - 6):origin + 1]
        out["rolling_zero_count_7"] = (w7 == 0).sum(axis=1).astype(np.float32)

        m7 = s[:, max(0, origin - 6):origin + 1].astype(np.float64).mean(axis=1)
        m28 = s[:, max(0, origin - 27):origin + 1].astype(np.float64).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["demand_momentum_7_28"] = np.where(
                m28 > 0, m7 / m28, np.nan).astype(np.float32)
        return out

    def _extra_price(self, origin: int, target_days: np.ndarray) -> dict:
        p = self.d.price_wide
        w_o = int(self.d.day_to_week[origin])

        def pct_change(back_weeks: int) -> np.ndarray:
            prev = p[:, max(0, w_o - back_weeks)]
            cur = p[:, w_o]
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(prev > 0, cur / prev - 1.0, np.nan).astype(np.float32)

        per_series = {
            "price_pct_change_1w": pct_change(1),
            "price_pct_change_4w": pct_change(4),
        }

        base = p[:, w_o]
        weeks = self.d.day_to_week[target_days]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_day = np.where(base[None, :] > 0,
                               p[:, weeks].T / base[None, :] - 1.0,
                               np.nan).astype(np.float32)
        return per_series, per_day

    def build_origin_frame(self, origin_idx, horizon=config.HORIZON,
                           series_idx=None, include_target=True):
        frame = super().build_origin_frame(
            origin_idx, horizon=horizon, series_idx=series_idx,
            include_target=include_target)

        if series_idx is None:
            series_idx = np.arange(self.d.sales_wide.shape[0])
        n_s = len(series_idx)
        target_days = origin_idx + 1 + np.arange(horizon)

        cal = self.d.calendar
        dts = pd.to_datetime(cal["date"]).dt
        dom = dts.day.to_numpy(np.int16)[target_days]
        woy = dts.isocalendar().week.to_numpy().astype(np.int16)[target_days]
        frame["day_of_month"] = np.repeat(dom, n_s)
        frame["week_of_year"] = np.repeat(woy, n_s)

        for name, arr in self._extra_demand(origin_idx).items():
            frame[name] = np.tile(arr[series_idx], horizon)

        per_series, per_day = self._extra_price(origin_idx, target_days)
        for name, arr in per_series.items():
            frame[name] = np.tile(arr[series_idx], horizon)
        frame["price_vs_origin_pct"] = per_day[:, series_idx].ravel()

        cat = frame["cat_id"].to_numpy()
        store = frame["store_id"].to_numpy()
        snap = frame["snap"].to_numpy()
        wknd = frame["is_weekend"].to_numpy()
        ev = (frame["event_name_1"].to_numpy() > 0).astype(np.int8)

        foods_code = self.d.cat_maps["cat_id"].get("FOODS", 0)
        frame["snap_food"] = (snap * (cat == foods_code)).astype(np.int8)
        frame["snap_x_cat"] = (snap * 3 + cat).astype(np.int16)
        frame["snap_x_store"] = (snap * 10 + store).astype(np.int16)
        frame["weekend_x_cat"] = (wknd * 3 + cat).astype(np.int16)
        frame["event_x_cat"] = (ev * 3 + cat).astype(np.int16)

        return frame


V2_GROUPS = {
    "A2_demand": ["rolling_mean_14", "rolling_std_14", "rolling_zero_count_7",
                  "demand_momentum_7_28"],
    "B2_calendar": ["day_of_month", "week_of_year"],
    "C2_price": ["price_pct_change_1w", "price_pct_change_4w", "price_vs_origin_pct"],
    "D2_interactions": ["snap_food", "snap_x_cat", "snap_x_store",
                        "weekend_x_cat", "event_x_cat"],
}

V2_CATEGORICAL = ["day_of_month", "week_of_year", "snap_x_cat", "snap_x_store",
                  "weekend_x_cat", "event_x_cat"]

BASE32 = [c for cols in FEATURE_GROUPS.values() for c in cols]


def feature_set(*groups: str) -> list[str]:
    out = list(BASE32)
    for g in groups:
        out.extend(V2_GROUPS[g])
    return out


V2_SETS: dict[str, list[str]] = {
    "v2_base": list(BASE32),
    "v2_A_demand": feature_set("A2_demand"),
    "v2_B_calendar": feature_set("B2_calendar"),
    "v2_C_price": feature_set("C2_price"),
    "v2_D_interactions": feature_set("D2_interactions"),
    "v2_all": feature_set("A2_demand", "B2_calendar", "C2_price", "D2_interactions"),
}

V2_LABELS = {
    "v2_base": "Current best 32 features (reference)",
    "v2_A_demand": "+ A. Short-term demand dynamics (4 features)",
    "v2_B_calendar": "+ B. Calendar expansion (2 features)",
    "v2_C_price": "+ C. Price dynamics (3 features)",
    "v2_D_interactions": "+ D. Interactions (5 features)",
    "v2_all": "+ All v2 groups (14 features)",
}


def categorical_for(cols: list[str]) -> list[int]:
    from .features import CATEGORICAL_FEATURES
    cats = set(CATEGORICAL_FEATURES) | set(V2_CATEGORICAL)
    return [i for i, c in enumerate(cols) if c in cats]
