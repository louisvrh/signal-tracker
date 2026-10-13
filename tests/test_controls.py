"""Engine controls (tracker/backtest/controls.py) and the nowcast engine's placebo options, on
synthetic data: no real data."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from tracker import store
from tracker.backtest import controls as ct
from tracker.backtest import nowcast as nc
from tracker.backtest import trading as tr
from tests.test_nowcast import REG, SPEC, TESTS, TRUTH, world


def test_shift_values_moves_values_later_and_wraps_round() -> None:
    x = pd.Series(np.arange(24.0), index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    out = nc.shift_values(x, 14)
    assert out.index.equals(x.index)  # same months, so the same forecast months and horizons
    assert out[pd.Timestamp("2021-12-01")] == x[pd.Timestamp("2020-10-01")]  # x'_m = x_(m-14)
    assert out[pd.Timestamp("2020-01-01")] == x[pd.Timestamp("2020-11-01")]  # wrapped: 14 back from 24 months
    assert sorted(out) == sorted(x) and nc.shift_values(x, 0) is x
    gappy = x.drop(pd.Timestamp("2020-10-01"))
    assert pd.Timestamp("2021-12-01") not in nc.shift_values(gappy, 14).index  # its source month is missing


def test_a_monthly_signal_is_used_as_it_is() -> None:
    v = store.Vintages(world())
    snap = v.latest(as_of=TRUTH)
    spec = nc.NowcastSpec("C", "comtrade_exports", {"DEU": "comtrade_exports"})
    x, y = nc.snapshot_features(snap, spec, REG)["DEU"]
    assert len(x) > 60 and np.allclose(x.to_numpy(), y.reindex(x.index).to_numpy())


def test_signal_from_borrows_another_countrys_series() -> None:
    snap = store.Vintages(world()).latest(as_of=TRUTH)
    own = nc.snapshot_features(snap, SPEC, REG)
    swapped = nc.snapshot_features(snap, replace(SPEC, signal_from={"DEU": "FRA", "FRA": "DEU"}), REG)
    pd.testing.assert_series_equal(swapped["DEU"][0], own["FRA"][0])
    pd.testing.assert_series_equal(swapped["DEU"][1], own["DEU"][1])  # the target stays the country's own


def test_placebos_lose_the_signal_that_the_real_world_has() -> None:
    v = store.Vintages(world(related=True))
    real = nc.score(nc.run(v, REG, SPEC, TESTS, TRUTH), "arx", "ar")
    shifted = nc.score(nc.run(v, REG, replace(SPEC, shift_months=17), TESTS, TRUTH), "arx", "ar")
    assert real["rmse_ratio"] < 0.75 and shifted["rmse_ratio"] > 0.9


def test_derangements_have_no_fixed_points() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = ct.derangement(["A", "B", "C", "D"], rng)
        assert sorted(d.values()) == ["A", "B", "C", "D"] and all(k != v for k, v in d.items())
    d = ct.derangement(["A", "B"], rng, pool=["A", "B", "C"])
    assert all(k != v and v in "ABC" for k, v in d.items())
    shifts = ct.placebo_shifts(rng, 5)
    assert len(set(shifts)) == 5 and not set(shifts) & {11, 12, 13, 23, 24, 25}


def test_delayed_and_synthetic_vintages() -> None:
    v = store.Vintages(world())
    late = ct.delayed(v, "comtrade_exports", 30)
    a = v.rows[v.rows["indicator"] == "comtrade_exports"]["available_at"].to_numpy()
    b = late.rows[late.rows["indicator"] == "comtrade_exports"]["available_at"].to_numpy()
    assert (b - a == np.timedelta64(30, "D")).all()
    truth = v.latest(as_of=TRUTH)
    synth = ct.synthetic_rows(truth, {"DEU": "comtrade_exports"}, 0.0, np.random.default_rng(1), start="2018-01-01")
    assert synth["date"].min() == pd.Timestamp("2018-01-01")
    month_end = (pd.DatetimeIndex(synth["date"]) + pd.offsets.MonthEnd(0)).tz_localize("UTC")
    assert (synth["available_at"] == month_end + pd.Timedelta(days=1)).all()
    reg = ct.with_synthetic(REG, "comtrade_exports")
    assert reg.indicator(ct.SYNTHETIC).frequency == "M" and ct.SYNTHETIC not in REG.indicators
    # Noise-free synthetic signal = the target itself, out a day after the month: a near-perfect nowcast.
    f = nc.run(ct.add_rows(v, synth), reg, nc.NowcastSpec("S", ct.SYNTHETIC, {"DEU": "comtrade_exports"}), TESTS, TRUTH)
    assert nc.score(f, "arx", "ar")["rmse_ratio"] < 0.05


def test_the_replica_matches_the_engine_and_the_timing_audit_is_clean() -> None:
    v = store.Vintages(world())
    engine = nc.run(v, REG, SPEC, TESTS, TRUTH)
    replica = ct.replicate_nowcast(v, REG, SPEC, TESTS, TRUTH)
    cmp = ct.compare_replication(engine, replica)
    assert cmp["matched"] == len(engine) > 50 and cmp["only_engine"] == cmp["only_replica"] == 0
    assert cmp["same_test_date"] and cmp["same_h"] and cmp["max_abs_diff_arx"] < 1e-8 and cmp["max_abs_diff_ar"] < 1e-8
    audit = ct.audit_timing(engine, v, SPEC, REG)
    assert audit == {"forecasts": len(engine), "signal_not_yet_published": 0, "target_already_published": 0,
                     "last_official_not_published": 0}
    # A leaky engine (forecasting a month already out) would be caught: pretend each forecast was a year later.
    leaky = engine.assign(test_date=engine["test_date"] + pd.Timedelta(days=365))
    assert ct.audit_timing(leaky, v, SPEC, REG)["target_already_published"] == len(engine)


def test_oracles_through_the_position_code() -> None:
    rng = np.random.default_rng(4)
    days = pd.bdate_range("2023-01-02", periods=300)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, 300), "B": rng.normal(0, 0.01, 300)}, index=days)
    good, bad = ct.oracle_daily(returns, tr.Costs(15, 2.0)), ct.oracle_daily(returns, tr.Costs(15, 2.0), anti=True)
    assert good["share_of_on_days_with_expected_sign"] == 1.0 and good["sharpe"] > 5 and bad["sharpe"] < -5
    mondays = days[days.dayofweek == 0]
    hold = ct.oracle_holds(returns, mondays, 20, tr.Costs(15, 2.0))
    assert hold["trades"] > 3 and hold["share_of_trades_with_expected_sign"] == 1.0


def test_shifted_event_dates_keep_their_spacing() -> None:
    days = pd.bdate_range("2022-01-03", periods=400)
    prices = pd.Series(np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, 400))), index=days)
    events = pd.DatetimeIndex([days[100], days[103], days[110]])
    means = ct.shifted_event_means(events, prices, days[0], days[-1] + pd.Timedelta(days=1), np.random.default_rng(3),
                                   draws=50, horizons=(10,))
    assert len(means[10]) == 50 and np.isfinite(means[10]).all()
    assert ct.randomisation_p(10.0, means[10]) == 0.0 and ct.randomisation_p(float(np.median(means[10])), means[10]) == 1.0


def test_e2_oracle_and_shuffle_rows() -> None:
    pit_rows = pd.DataFrame({"signal": "d_off", "country": ["DEU", "FRA", "DEU"], "value": [0.5, -1.0, 2.0],
                             "as_of": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-08"], utc=True)})
    forecasts = pd.DataFrame({"test_date": pd.to_datetime(["2024-01-10", "2024-01-10"], utc=True),
                              "country": ["DEU", "FRA"], "outcome": [1.0, 0.0]})
    used = ct.e2_used_rows(pit_rows, forecasts)
    assert used == {2: 1.0, 1: 0.0}  # DEU's latest D_off in the 14 days before the test
    oracle = ct.e2_oracle_rows(pit_rows, used, np.random.default_rng(0), noise=0.0)
    assert oracle.loc[2, "value"] == -1.0 and oracle.loc[1, "value"] == 1.0 and oracle.loc[0, "value"] == 0.5
    shuffled = ct.e2_shuffled_rows(pit_rows, np.random.default_rng(0))
    assert sorted(shuffled["value"]) == sorted(pit_rows["value"])

