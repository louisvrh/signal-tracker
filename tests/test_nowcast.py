"""Nowcast tests (S23; tracker/backtest/nowcast.py) on synthetic vintages: no real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracker import store
from tracker.backtest import nowcast as nc
from tracker.registry import load_registry

REG = load_registry()
DAYS = pd.date_range("2016-01-01", "2025-12-31", freq="D")
MONTHS = pd.date_range("2016-01-01", "2025-11-01", freq="MS")
TESTS = pd.date_range("2019-01-15", "2024-12-15", freq="MS") + pd.Timedelta(days=14)
TRUTH = pd.Timestamp("2025-01-01", tz="UTC")


def rows(dates, indicator: str, country: str, values, available) -> pd.DataFrame:
    available = pd.DatetimeIndex(available).tz_localize("UTC")
    return pd.DataFrame({"date": pd.DatetimeIndex(dates), "country": country, "indicator": indicator,
                         "value": np.asarray(values, float), "unit": "x",
                         "frequency": REG.indicator(indicator).frequency, "source": indicator.split("_")[0],
                         "fetched_at": available, "available_at": available, "vintage_type": "estimated"})


def world(related: bool = True, seed: int = 3, countries=("DEU", "FRA")) -> pd.DataFrame:
    """Port exports whose monthly growth wanders; official exports that follow it (or don't),
    published 60 days after each month."""
    rng = np.random.default_rng(seed)
    parts = []
    for c in countries:
        growth = np.cumsum(rng.normal(0, 0.004, len(MONTHS)))  # monthly log level of activity
        month_level = pd.Series(np.exp(growth), index=MONTHS)
        daily = month_level.reindex(DAYS, method="ffill").to_numpy() * 1e6 * rng.lognormal(0, 0.02, len(DAYS))
        parts.append(rows(DAYS, "portwatch_exports", c, daily, DAYS + pd.Timedelta(days=11)))
        driver = growth if related else np.cumsum(rng.normal(0, 0.004, len(MONTHS)))
        official = 1e9 * np.exp(1.5 * driver + rng.normal(0, 0.003, len(MONTHS)))
        published = MONTHS + pd.offsets.MonthEnd(0) + pd.Timedelta(days=60)
        parts.append(rows(MONTHS, "comtrade_exports", c, official, published))
    return pd.concat(parts, ignore_index=True)


SPEC = nc.NowcastSpec("H1", "portwatch_exports", {"DEU": "comtrade_exports", "FRA": "comtrade_exports"})


def test_monthly_means_need_nearly_complete_months() -> None:
    days = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    s = pd.Series(1.0, index=days).drop(pd.date_range("2024-02-01", "2024-02-10"))  # February a third missing
    out = nc.monthly_mean(s)
    assert list(out.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")]
    weekdays = pd.Series(2.0, index=days[days.dayofweek < 5])
    assert len(nc.monthly_mean(weekdays, calendar="weekdays")) == 3


def test_a_signal_that_drives_the_target_beats_the_benchmark_and_a_placebo_does_not() -> None:
    good = nc.run(store.Vintages(world(related=True)), REG, SPEC, TESTS, TRUTH)
    placebo = nc.run(store.Vintages(world(related=False)), REG, SPEC, TESTS, TRUTH)
    g, p = nc.score(good, "arx", "ar"), nc.score(placebo, "arx", "ar")
    assert g["n"] > 60 and g["rmse_ratio"] < 0.75 and g["dm_p"] < 0.01 and g["beats_benchmark_both_halves"]
    assert p["rmse_ratio"] > 0.9
    assert (good["h"] >= 2).all()  # Comtrade-like 60-day lag: the last published month is 2+ months back
    assert (good["n_train"] >= nc.MIN_TRAIN).all()


def test_each_month_is_forecast_once_before_its_official_value_is_out() -> None:
    f = nc.run(store.Vintages(world()), REG, SPEC, TESTS, TRUTH)
    assert not f.duplicated(["country", "month"]).any()
    published = f["month"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=60)
    assert (f["test_date"].dt.tz_localize(None) < published).all()
    assert (f["test_date"].dt.tz_localize(None) > f["month"] + pd.offsets.MonthEnd(0)).all()  # month complete
    # Scored only on official values out before the vault starts (TRUTH); later months stay unscored.
    known = published < TRUTH.tz_localize(None)
    assert f.loc[known, "y_true"].notna().all() and f.loc[~known, "y_true"].isna().all() and (~known).any()


def test_nothing_published_after_the_vault_start_changes_any_result() -> None:
    clean = world()
    rng = np.random.default_rng(9)
    poisoned = clean.copy()
    late = poisoned["available_at"] > TRUTH
    poisoned.loc[late, "value"] = rng.uniform(1e3, 1e12, int(late.sum()))
    revisions = poisoned[~late].copy()
    revisions["available_at"] = revisions["fetched_at"] = TRUTH + pd.Timedelta(seconds=1)
    revisions["value"] = 7.0
    poisoned = pd.concat([poisoned, revisions], ignore_index=True)
    a = nc.run(store.Vintages(clean), REG, SPEC, TESTS, TRUTH)
    b = nc.run(store.Vintages(poisoned), REG, SPEC, TESTS, TRUTH)
    pd.testing.assert_frame_equal(a, b)


def test_ex_2020_drops_2020_months_from_training_and_scoring() -> None:
    f = nc.run(store.Vintages(world()), REG, nc.NowcastSpec(**{**SPEC.__dict__, "exclude_2020": True}), TESTS, TRUTH)
    assert not (f["month"].dt.year == 2020).any()
    x = pd.Series(1.0, index=pd.date_range("2018-01-01", "2021-12-01", freq="MS"))
    t = nc.training_rows(x, x, 2, exclude_2020=True)
    assert not (t.index.year == 2020).any() and len(t) == 48 - 2 - 12


def test_diebold_mariano_and_scores() -> None:
    rng = np.random.default_rng(0)
    clearly_better = pd.Series(-1 + rng.normal(0, 0.5, 60))
    stat, p = nc.diebold_mariano(clearly_better)
    assert stat < -5 and p < 1e-6
    stat, p = nc.diebold_mariano(pd.Series(rng.normal(0, 1, 60)))
    assert p > 0.01
    f = pd.DataFrame({"test_date": pd.date_range("2024-01-15", periods=4, freq="MS"), "country": "DEU",
                      "y_true": [1.0, 2.0, 3.0, 4.0], "y_last": [0.0, 0.0, 5.0, 5.0],
                      "arx": [1.5, 2.5, 3.5, 4.5], "ar": [-1.0, -1.0, 6.0, 6.0]})  # ar calls every move wrong
    s = nc.score(f, "arx", "ar")
    assert s["rmse"] == pytest.approx(0.5) and s["hit_rate"] == 1.0 and s["hit_rate_benchmark"] == 0.0
    assert s["ratio_first_half"] == pytest.approx(0.5 / np.sqrt(6.5)) and s["n"] == 4


def test_a_country_without_any_official_data_is_skipped_not_fatal() -> None:
    """Taiwan has port data but no IMF series: the pooled model must just leave it out."""
    vintages = world(countries=("DEU", "FRA", "TWN"))
    vintages = vintages[~((vintages["country"] == "TWN") & (vintages["indicator"] == "comtrade_exports"))]
    spec = nc.NowcastSpec("H1b", "portwatch_exports", {"DEU": "comtrade_exports", "FRA": "comtrade_exports",
                                                       "TWN": "comtrade_exports"})
    f = nc.run(store.Vintages(vintages), REG, spec, TESTS, TRUTH)
    assert set(f["country"]) == {"DEU", "FRA"} and f["pooled"].notna().all()
    assert nc.training_rows(pd.Series(dtype="float64"), pd.Series(dtype="float64"), 2, False).empty


def test_a_forecast_that_calls_no_direction_has_no_hit_rate() -> None:
    f = pd.DataFrame({"test_date": pd.date_range("2024-01-15", periods=4, freq="MS"), "country": "DEU",
                      "y_true": [1.0, 2.0, 3.0, 4.0], "y_last": [0.0, 0.0, 5.0, 5.0],
                      "arx": [1.5, 2.5, 3.5, 4.5], "rw": [0.0, 0.0, 5.0, 5.0]})
    s = nc.score(f, "arx", "rw")
    assert s["hit_rate"] == 1.0 and np.isnan(s["hit_rate_benchmark"])
