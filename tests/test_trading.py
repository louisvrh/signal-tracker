"""Trading and event-study helpers (S24; tracker/backtest/trading.py) on synthetic prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracker.backtest import trading as tr

DAYS = pd.bdate_range("2024-01-01", "2024-12-31")


def test_a_signal_is_traded_from_the_next_close() -> None:
    """A Monday 00:00 snapshot signal is entered at Monday's close, so it earns from Tuesday's return."""
    signal = pd.Series([1.0, 0.0], index=pd.DatetimeIndex(["2024-01-08", "2024-01-15"], tz="UTC"))
    pos = tr.positions_from_signal(signal, DAYS)
    assert pos["2024-01-08"] == 0.0 and pos["2024-01-09"] == 1.0
    assert pos["2024-01-15"] == 1.0 and pos["2024-01-16"] == 0.0  # exit at Monday 15th's close
    holds = tr.fixed_holds(pd.DatetimeIndex(["2024-01-06"]), DAYS, 3)  # a Saturday: first close is Monday
    assert holds[holds > 0].index.tolist() == list(pd.to_datetime(["2024-01-09", "2024-01-10", "2024-01-11"]))


def test_costs_and_borrow_are_charged() -> None:
    returns = pd.Series(0.01, index=DAYS[:10])
    pos = pd.Series([0, 1, 1, 1, 0, 0, 1, 1, 0, 0], index=DAYS[:10], dtype=float)
    net = tr.backtest(pos, returns, tr.Costs(per_side_bp=10, borrow_pct_year=2.52, legs=2))
    # Day 1: +1% minus entry 2 legs x 10 bp and borrow 0.01% per day.
    assert net.iloc[1] == pytest.approx(0.01 - 0.002 - 0.0001)
    assert net.iloc[4] == pytest.approx(-0.002)  # exit cost, no return
    assert net.iloc[0] == 0.0
    m = tr.metrics(net, pos)
    assert m["trades"] == 2 and m["days_on"] == 5


def test_portfolio_weights_are_equal_across_open_trades() -> None:
    idx = DAYS[:4]
    positions = pd.DataFrame({"A": [1, 1, 1, 0], "B": [0, 1, 1, 1]}, index=idx, dtype=float)
    returns = pd.DataFrame({"A": [0.02] * 4, "B": [-0.02] * 4}, index=idx)
    net, on = tr.portfolio(positions, returns, tr.Costs(per_side_bp=0, borrow_pct_year=0))
    assert net.tolist() == pytest.approx([0.02, 0.0, 0.0, -0.02]) and on.tolist() == [1, 1, 1, 1]


def test_random_trades_and_deflated_sharpe() -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, len(DAYS)), index=DAYS)
    returns.iloc[100:120] = -0.01  # short this: a known good window
    pos = pd.Series(0.0, index=DAYS)
    pos.iloc[100:120] = 1.0
    shorted = -returns
    r = tr.random_trades(pos, shorted, tr.Costs(), n=300)
    assert r["random_p"] < 0.05
    good = tr.deflated_sharpe(3.0, [3.0, 0.1, -0.2, 0.3], 500, 0.0, 3.0)
    lucky = tr.deflated_sharpe(1.0, [1.0, 0.9, -1.0, 1.2, 0.5, -0.8] * 5, 250, 0.0, 3.0)
    assert good > 0.95 and lucky < good


def test_event_study_compares_events_with_other_days() -> None:
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.full(len(DAYS), 0.001))), index=DAYS)
    jump_after = [DAYS[50], DAYS[150]]
    for d in jump_after:
        prices[prices.index > d + pd.Timedelta(days=3)] *= 1.10
    study = tr.event_study(pd.DatetimeIndex(jump_after), prices, (10,))
    row = study.iloc[0]
    assert row["events"] == 2 and row["mean"] > row["unconditional_mean"] + 0.05
    assert row["share_up"] == 1.0
