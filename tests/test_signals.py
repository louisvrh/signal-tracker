"""Composite scores, divergences and flags (S19; tracker/signals/composite.py, divergence.py)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tracker import store
from tracker.backtest import pit
from tracker.registry import load_registry
from tracker.signals import composite as comp
from tracker.signals import divergence as dv

REG = load_registry()
DAILY = pd.date_range("2018-01-01", "2025-12-31", freq="D")
WEEKDAYS = DAILY[DAILY.dayofweek < 5]
MONTHS = pd.date_range("2015-01-01", "2025-10-01", freq="MS")


# ------------------------------------------------------------------ flags


@pytest.mark.parametrize("values, expected", [
    ([-2, -2, -2, -1, -0.6, -0.4, -2], [0, 0, 1, 1, 1, 0, 0]),  # on at the 3rd week, off above -0.5
    ([-2, -2, -1, -2, -2, -2], [0, 0, 0, 0, 0, 1]),             # a week above -1.5 restarts the count
    ([-2, -2, np.nan, -2, -2, -2], [0, 0, 0, 0, 0, 1]),         # a missing week breaks the run too
    ([-2, -2, -2, np.nan, -0.6, -0.3], [0, 0, 1, 1, 1, 0]),     # ... but leaves an active flag on
    ([-1.6, -1.6, -1.6, -0.5, -0.49], [0, 0, 1, 1, 0]),          # off only strictly above -0.5
])
def test_flags_need_three_weeks_and_switch_off_with_hysteresis(values, expected) -> None:
    out = dv.flag_states(pd.Series(values, index=pd.date_range("2025-01-06", periods=len(values), freq="7D")))
    assert out.astype(int).tolist() == expected


def test_flag_table_marks_on_and_off_events() -> None:
    dates = pd.date_range("2025-01-06", periods=6, freq="7D")
    signals = pd.DataFrame({"date": dates, "country": "DEU", "signal": "d_mkt", "value": [-2, -2, -2, -1, 0, 0.0]})
    table = dv.flag_table(signals)
    assert table["event"].fillna("-").tolist() == ["-", "-", "on", "-", "off", "-"]


# ------------------------------------------------------------------ scores


def comps(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["country", "indicator", "category", "z", "weight"])
    return df.assign(date=pd.Timestamp("2025-06-02"), value_date=pd.Timestamp("2025-05-30"))


def test_score_is_the_mean_of_weighted_category_means_with_coverage() -> None:
    sc = comp.scores(comps([
        ("DEU", "portwatch_exports", "trade", -1.0, 1.0), ("DEU", "portwatch_imports", "trade", -3.0, 0.5),
        ("DEU", "entsoe_load", "energy", 1.0, 1.0), ("DEU", "ember_demand", "energy", 2.0, 1.0),
        ("ZAF", "portwatch_exports", "trade", -1.0, 1.0), ("ZAF", "portwatch_imports", "trade", -2.0, 1.0),
        ("SAU", "portwatch_exports", "trade", -4.0, 1.0),
    ])).set_index("country")
    trade = (-1.0 * 1 + -3.0 * 0.5) / 1.5  # the holiday-flagged component has half weight
    assert sc.loc["DEU", "trade"] == pytest.approx(trade) and sc.loc["DEU", "energy"] == pytest.approx(1.5)
    assert sc.loc["DEU", "score"] == pytest.approx((trade + 1.5) / 2) and sc.loc["DEU", "coverage"] == 4
    assert sc.loc["ZAF", "score"] == pytest.approx(-1.5) and np.isnan(sc.loc["ZAF", "energy"])
    assert np.isnan(sc.loc["SAU", "score"]) and sc.loc["SAU", "coverage"] == 1  # needs two components


def test_components_follow_the_registry() -> None:
    """ZAF is excluded from power-demand signals (PLAN D22), so it has trade components only."""
    energy = [c for c in comp.COMPONENTS if c.category == "energy"]
    assert all("ZAF" not in REG.signal_countries(c.indicator) for c in energy)
    assert {c.chain for c in energy if "load" in c.indicator} == {"wx>yoy>rz(3y)"}
    assert all(REG.indicator(c.indicator).role == "alt" for c in comp.COMPONENTS)


def test_weekly_grid_ends_on_the_snapshot_day() -> None:
    grid = comp.week_grid(pd.Timestamp("2025-06-04 07:30", tz="UTC"), start=pd.Timestamp("2025-05-01"))
    assert grid[-1] == pd.Timestamp("2025-06-04") and (grid.to_series().diff().dropna() == pd.Timedelta(days=7)).all()
    assert grid[0] >= pd.Timestamp("2025-05-01")


def test_latest_values_count_only_while_fresh() -> None:
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2025-01-10", "2025-03-01"]))
    grid = pd.DatetimeIndex(["2025-01-05", "2025-01-20", "2025-02-28", "2025-03-05", "2025-05-01"])
    vals, dates = comp.as_of_grid(s, grid, max_age_days=45)
    assert np.isnan(vals.iloc[0]) and vals.iloc[1] == 1.0 and np.isnan(vals.iloc[2]) and vals.iloc[3] == 2.0
    assert np.isnan(vals.iloc[4]) and dates.iloc[3] == pd.Timestamp("2025-03-01")
    monthly = comp.period_end(pd.Series([5.0], index=pd.DatetimeIndex(["2025-02-01"])), "M")
    assert monthly.index[0] == pd.Timestamp("2025-02-28")  # a month counts from its end


def test_three_month_return_and_mirror_months() -> None:
    prices = pd.Series(np.linspace(100, 200, len(WEEKDAYS)), index=WEEKDAYS)
    ret = dv.three_month_return(prices)
    day = pd.Timestamp("2024-06-03")
    base = prices[prices.index <= day - pd.Timedelta(days=91)].iloc[-1]
    assert ret[day] == pytest.approx((prices[day] / base - 1) * 100)
    assert ret.index.min() >= WEEKDAYS.min() + pd.Timedelta(days=91)
    months = pd.date_range("2024-01-01", periods=16, freq="MS")
    reporters = pd.Series([80.0] * 13 + [70.0, 45.0, 14.0], index=months)
    kept = dv.complete_mirror_months(pd.Series(1.0, index=months), reporters)
    assert kept.index.max() == pd.Timestamp("2025-02-01")  # 70 of the usual 80 still counts; 45 doesn't


# ----------------------------------------------- the engine hook (PLAN D39)


def rows(dates, indicator: str, country: str, values, lag_days: int) -> pd.DataFrame:
    available = (pd.DatetimeIndex(dates) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)
                 if REG.indicator(indicator).frequency == "M" else pd.DatetimeIndex(dates) + pd.Timedelta(days=lag_days))
    available = available.tz_localize("UTC")
    return pd.DataFrame({"date": pd.DatetimeIndex(dates), "country": country, "indicator": indicator,
                         "value": np.asarray(values, float), "unit": "x", "frequency": REG.indicator(indicator).frequency,
                         "source": indicator.split("_")[0], "fetched_at": available, "available_at": available,
                         "vintage_type": "estimated"})


def synthetic_vintages(seed: int = 4) -> pd.DataFrame:
    """Ports, Ember demand, official IP and exports and ETF prices for DEU and CHN, with a slump in
    German activity in 2024 that markets ignore."""
    rng = np.random.default_rng(seed)
    t = np.arange(len(DAILY))
    slump = np.where((DAILY >= "2024-03-01") & (DAILY < "2024-09-01"), 0.8, 1.0)
    parts = []
    for n, c in enumerate(["DEU", "CHN"]):
        for ind in ("portwatch_exports", "portwatch_imports"):
            level = (1e6 + 3e5 * np.sin(2 * np.pi * t / 365.25)) * (slump if c == "DEU" else 1.0)
            parts.append(rows(DAILY, ind, c, level * rng.lognormal(0, 0.05, len(t)), 11))
        m = np.arange(len(MONTHS))
        parts.append(rows(MONTHS, "ember_demand", c, 40 + 5 * np.sin(2 * np.pi * m / 12) + rng.normal(0, 0.5, len(m)), 45))
        etf = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(WEEKDAYS))))
        parts.append(rows(WEEKDAYS, "prices_country_etf", c, etf, 1))
    parts.append(rows(MONTHS, "eurostat_ip", "DEU", 100 + np.random.default_rng(9).normal(0, 1, len(MONTHS)), 47))
    parts.append(rows(MONTHS, "imf_exports", "CHN", 2e11 * np.exp(np.cumsum(rng.normal(0.003, 0.02, len(MONTHS)))), 122))
    parts.append(rows(WEEKDAYS, "prices_benchmark", "WLD", 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, len(WEEKDAYS)))), 1))
    return pd.concat(parts, ignore_index=True)


def test_panel_signals_plug_into_the_engine() -> None:
    names = set(pit.available_panel_signals())
    assert set(comp.COMPOSITE_SIGNALS) | set(dv.DIVERGENCE_SIGNALS) <= names


def test_engine_end_values_match_the_daily_command_and_nothing_later_leaks() -> None:
    """At a test date the engine's panel end values equal the live computation on the snapshot,
    and poisoning everything published afterwards changes none of them."""
    clean = synthetic_vintages()
    cut = pd.Timestamp("2024-08-05", tz="UTC")
    rng = np.random.default_rng(1)
    poisoned = clean.copy()
    future = poisoned["available_at"] > cut
    poisoned.loc[future, "value"] = rng.uniform(1e9, 1e12, int(future.sum()))
    revised = poisoned[~future].copy()
    revised["available_at"] = revised["fetched_at"] = cut + pd.Timedelta(seconds=1)
    revised["value"] = 1e12
    poisoned = pd.concat([poisoned, revised], ignore_index=True)

    panel = pit.available_panel_signals()
    dates = [cut - pd.Timedelta(days=7), cut]
    a = pit.compute(store.Vintages(clean), REG, dates, panel=panel)
    b = pit.compute(store.Vintages(poisoned), REG, dates, panel=panel)
    pd.testing.assert_frame_equal(a, b)

    snap = pit.snapshot(store.Vintages(clean), cut)
    live = dv.signals_frame(snap, cut, REG)
    end = a[a["as_of"] == cut].set_index(["signal", "country"])["value"]
    for (signal, country), value in end.items():
        mine = live[(live["signal"] == signal) & (live["country"] == country)].sort_values("date")["value"]
        assert value == pytest.approx(mine.iloc[-1])
    # Germany's slump shows: its activity score is well below China's, with every component counted.
    assert end[("activity", "DEU")] < end[("activity", "CHN")] - 1
    deu = live[(live["signal"] == "activity") & (live["country"] == "DEU")].sort_values("date").iloc[-1]
    assert set(json.loads(deu["components"])) == {"portwatch_exports", "portwatch_imports", "ember_demand"}
    assert {"d_mkt", "d_off"} <= set(end.index.get_level_values("signal"))


def test_signals_command_writes_its_tables() -> None:
    from tracker import daily_signals

    store.append_changes(synthetic_vintages(), "r1")
    result = daily_signals.run_signals(as_of=pd.Timestamp("2024-08-05", tz="UTC"))
    table = store.read_signal_table(daily_signals.SIGNALS_TABLE)
    flags = store.read_signal_table(daily_signals.FLAGS_TABLE)
    assert list(table.columns) == ["computed_at", "as_of", "date", "country", "signal", "value", "components"]
    assert {"activity", "activity_z", "d_mkt", "d_off", "z:portwatch_exports"} <= set(table["signal"])
    assert set(flags["signal"]) == {"d_mkt", "d_off"} and flags["flag"].dtype == bool
    assert len(result["top"]) == 4 and result["top"]["value"].is_monotonic_increasing
    assert all(daily_signals.describe(r).startswith(r["country"]) for _, r in result["top"].iterrows())
