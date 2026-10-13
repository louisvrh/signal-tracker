"""Why-the-nowcasts-failed diagnostics (tracker/backtest/why.py) on synthetic data: no real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracker import store
from tracker.backtest import nowcast as nc
from tracker.backtest import why
from tests.test_nowcast import REG, SPEC, TESTS, TRUTH, world

MONTHS = pd.date_range("2018-01-01", "2025-12-01", freq="MS")


def frame(x, y) -> dict[str, pd.DataFrame]:
    return {"AAA": pd.DataFrame({"x": x, "y": y}, index=MONTHS)}


def test_lead_lag_peaks_where_the_signal_leads() -> None:
    rng = np.random.default_rng(0)
    driver = pd.Series(np.cumsum(rng.normal(0, 1, len(MONTHS) + 2)))
    y = driver.iloc[2:].to_numpy()
    x_aligned = y + rng.normal(0, 0.3, len(MONTHS))
    x_late = driver.iloc[:-2].to_numpy() + rng.normal(0, 0.3, len(MONTHS))  # x_t tracks y_(t-2)
    aligned = why.lead_lag(frame(x_aligned, y))
    late = why.lead_lag(frame(x_late, y))
    assert aligned.loc[aligned["median_corr"].idxmax(), "lag"] == 0
    assert late.loc[late["median_corr"].idxmax(), "lag"] == 2  # y_t matches x_(t+2)


def test_periods_and_rolling_correlations_see_a_break() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, len(MONTHS))
    y = np.where(MONTHS < "2022-01-01", x, -x) + rng.normal(0, 0.1, len(MONTHS))
    p = why.by_period(frame(x, y)).set_index("period")
    assert p.loc["2020-21", "corr"] > 0.9 and p.loc["2022-23", "corr"] < -0.9
    r = why.rolling_corr(frame(x, y), window=12).set_index("date")["median_corr"]
    assert r[pd.Timestamp("2021-12-01")] > 0.9 and r[pd.Timestamp("2023-12-01")] < -0.9


def test_in_sample_oracle_and_walk_forward_ratios() -> None:
    v = store.Vintages(world(related=True))
    f = nc.run(v, REG, SPEC, TESTS, TRUTH)
    data = why.pairs(v.latest(as_of=TRUTH), SPEC, REG)
    per = why.in_sample_vs_oos(f, data)
    assert set(per["country"]) == {"DEU", "FRA"}
    assert (per["in_sample_ratio"] < 0.8).all() and (per["oracle_coef_ratio"] < 0.8).all()
    pooled = why.pooled_ratios(f, data)
    assert pooled["forecasts"] == f["y_true"].notna().sum()
    assert pooled["walk_forward_ratio"] == pytest.approx(nc.score(f, "arx", "ar")["rmse_ratio"])
    unrelated = store.Vintages(world(related=False))
    g = nc.run(unrelated, REG, SPEC, TESTS, TRUTH)
    assert why.pooled_ratios(g, why.pairs(unrelated.latest(as_of=TRUTH), SPEC, REG))["oracle_coef_ratio"] > 0.9


def test_benchmark_refitted_on_recent_months_only() -> None:
    v = store.Vintages(world(related=True))
    f = nc.run(v, REG, SPEC, TESTS, TRUTH)
    data = why.pairs(v.latest(as_of=TRUTH), SPEC, REG)
    out = why.benchmark_from(f, data, start="2020-01-01")
    assert 0 < out["forecasts"] <= f["y_true"].notna().sum()
    assert out["rmse_ar"] > 0 and out["rmse_last_value"] > 0 and out["rmse_ar_from_2020"] > 0
    assert why.benchmark_from(f, data, start="2030-01-01") == {"forecasts": 0}


def test_weekend_share_and_calendar_effect() -> None:
    w = why.weekend_share_change(pd.DatetimeIndex(["2024-06-01", "2025-03-01"]))
    # June 2024 had 10 weekend days, June 2023 had 8; March 2025 had 10, March 2024 had 10.
    assert w.iloc[0] == pytest.approx(2 / 30) and w.iloc[1] == pytest.approx(0)
    shares = why.weekend_share_change(MONTHS) * 100
    data = frame(-0.7 * shares + np.random.default_rng(2).normal(0, 0.1, len(MONTHS)), np.zeros(len(MONTHS)) + 1)
    c = why.calendar_effect(data)
    signal = c[c["series"] == "signal"].iloc[0]
    assert signal["pp_per_weekend_point"] == pytest.approx(-0.7, abs=0.05) and signal["r2"] > 0.9
    assert (c["series"] == "target").sum() == 0  # a constant target has no variance to explain


def test_value_volume_shares() -> None:
    idx = pd.date_range("2019-01-01", "2025-12-01", freq="MS")
    rng = np.random.default_rng(3)
    volume = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    price = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.04, len(idx)))), index=idx)
    port = volume * np.exp(rng.normal(0, 0.005, len(idx)))
    out = why.value_volume(volume * price, volume, price, port)
    assert out["r2_value_on_price"] > out["r2_value_on_volume"]  # prices move four times as much
    assert out["corr_port_volume"] > 0.9 and out["corr_port_volume"] > out["corr_port_value"]
