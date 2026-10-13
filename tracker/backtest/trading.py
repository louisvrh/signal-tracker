"""Trading and event-study tests (S24; PLAN §5.6 items 4, 6 and 8, SIGNALS.md H3-H6).

* A strategy is a daily **position** series in a long-short pair (or one leg):
  +1 = the trade is on, 0 = flat, decided from signals known by each test date
  and entered at the **next close** after the signal is available.
* `backtest` turns positions and daily returns into net daily returns after
  costs: a cost per side on every change of position for each leg traded, and
  a yearly borrow fee on the short leg while the trade is on.
* `metrics`: annual return, Sharpe ratio (annualised, 252 days), maximum
  drawdown, hit rate (share of trades that made money) and turnover.
* Benchmarks: holding the same trade all the time ("always on"), and random
  entry dates with the same number and length of trades (`random_trades`).
* `deflated_sharpe` (Bailey and López de Prado, 2014) discounts the Sharpe
  ratio for the number of variants tried.
* `event_study` (H5): returns over fixed horizons after each event against
  the same horizons on every other day.

Pure: series in, frames and numbers out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649


@dataclass(frozen=True)
class Costs:
    """Trading costs (§5.6 item 6): per side, per leg, and a yearly borrow fee on the short leg."""

    per_side_bp: float = 15.0
    borrow_pct_year: float = 2.0
    legs: int = 2  # long-short pairs trade two instruments on every change
    short_leg: bool = True


def daily_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns of a price series (adjusted closes), indexed by the later day."""
    p = prices.dropna().sort_index()
    return p.pct_change().dropna()


def pair_returns(long: pd.Series | None, short: pd.Series | None) -> pd.Series:
    """Daily return of a long-short pair (long leg return − short leg return) on common days;
    one leg only if the other is None."""
    parts = []
    if long is not None:
        parts.append(daily_returns(long).rename("long"))
    if short is not None:
        parts.append(daily_returns(short).rename("short"))
    both = pd.concat(parts, axis=1, sort=True).dropna()
    out = both.get("long", 0.0) - both.get("short", 0.0)
    return pd.Series(out, index=both.index, dtype="float64")


def positions_from_signal(on: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Daily positions (0/1) from a signal known at its dates (UTC timestamps or days): the
    position changes at the close of the first trading day on or after each signal date, so it
    earns from the next day's return on (the trade is entered at the next close)."""
    sig = on.copy()
    sig.index = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo else pd.Timestamp(d)
                                  for d in sig.index]).normalize()
    sig = sig[~sig.index.duplicated(keep="last")].sort_index()
    days = pd.DatetimeIndex(trading_days).sort_values()
    pos = np.searchsorted(days.to_numpy(), sig.index.to_numpy(), side="left")
    state = pd.Series(np.nan, index=days)
    for p, value in zip(pos, sig.to_numpy(dtype=float)):
        if p < len(days):
            state.iloc[p] = value
    held_from_close = state.ffill().fillna(0.0)
    return held_from_close.shift(1).fillna(0.0)  # a close-of-day entry earns from the next day


def fixed_holds(entries: pd.DatetimeIndex, trading_days: pd.DatetimeIndex, hold_days: int) -> pd.Series:
    """Daily positions (0/1) for trades entered at the first close on or after each entry date
    and held for `hold_days` trading days; overlapping trades just extend the position."""
    days = pd.DatetimeIndex(trading_days).sort_values()
    pos = np.zeros(len(days))
    for e in entries:
        e = pd.Timestamp(e).tz_localize(None) if pd.Timestamp(e).tzinfo else pd.Timestamp(e)
        i = int(np.searchsorted(days.to_numpy(), e.normalize().to_datetime64(), side="left"))
        pos[i + 1: i + 1 + hold_days] = 1.0
    return pd.Series(pos, index=days)


def backtest(position: pd.Series, returns: pd.Series, costs: Costs) -> pd.Series:
    """Net daily strategy returns: position × return, less a cost on each change of position
    (per leg) and the borrow fee on the short leg while on."""
    pos = position.reindex(returns.index).fillna(0.0)
    gross = pos * returns
    changes = pos.diff().abs().fillna(pos.abs())
    trading = changes * costs.legs * costs.per_side_bp / 1e4
    borrow = pos.abs() * (costs.borrow_pct_year / 100 / TRADING_DAYS if costs.short_leg else 0.0)
    return gross - trading - borrow


def portfolio(positions: pd.DataFrame, returns: pd.DataFrame, costs: Costs) -> tuple[pd.Series, pd.Series]:
    """Equal weight across the trades that are on (columns = instruments or countries): daily net
    returns and the total weight (1 while anything is on, else 0). Costs follow each weight change."""
    pos = positions.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    count = pos.sum(axis=1)
    weights = pos.div(count.where(count > 0, 1.0), axis=0)
    gross = (weights * returns.fillna(0.0)).sum(axis=1)
    changes = weights.diff().abs().fillna(weights.abs()).sum(axis=1)
    trading = changes * costs.legs * costs.per_side_bp / 1e4
    borrow = weights.sum(axis=1) * (costs.borrow_pct_year / 100 / TRADING_DAYS if costs.short_leg else 0.0)
    return gross - trading - borrow, (count > 0).astype(float)


def random_portfolio(positions: pd.DataFrame, returns: pd.DataFrame, costs: Costs, n: int = 300,
                     seed: int = 0) -> dict[str, Any]:
    """`random_trades` for a portfolio: each column's trades keep their number and lengths but get
    random dates; the share of random portfolios with a total net return at least the real one's."""
    pos = positions.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    actual_net, _ = portfolio(pos, returns, costs)
    actual = float(np.prod(1 + actual_net.to_numpy()) - 1)
    runs = {}
    for col in pos.columns:
        on = pos[col] > 0
        runs[col] = [len(p) for _, p in pos[col][on].groupby((on != on.shift(fill_value=False)).cumsum()[on])]
    if not any(runs.values()):
        return {"random_p": float("nan"), "random_mean_return": float("nan")}
    rng = np.random.default_rng(seed)
    days = len(returns)
    totals = []
    for _ in range(n):
        fake = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        for col, lengths in runs.items():
            for length in lengths:
                start = int(rng.integers(0, max(days - length, 1)))
                fake.iloc[start:start + length, fake.columns.get_loc(col)] = 1.0
        net, _ = portfolio(fake, returns, costs)
        totals.append(float(np.prod(1 + net.to_numpy()) - 1))
    totals_arr = np.array(totals)
    return {"random_p": float((totals_arr >= actual).mean()), "random_mean_return": float(totals_arr.mean()),
            "random_trials": n}


def trades(position: pd.Series, net: pd.Series) -> list[float]:
    """The net return of each trade (a run of days with the position on)."""
    on = position.reindex(net.index).fillna(0.0) > 0
    run_id = (on != on.shift(fill_value=False)).cumsum()
    out = []
    for _, part in net[on].groupby(run_id[on]):
        out.append(float(np.prod(1 + part.to_numpy()) - 1))
    return out


def metrics(net: pd.Series, position: pd.Series) -> dict[str, Any]:
    """Annual return, Sharpe, maximum drawdown, hit rate, trades, days on and turnover."""
    r = net.dropna()
    if r.empty:
        return {"days": 0}
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    sd = float(r.std(ddof=1))
    t = trades(position, net)
    pos = position.reindex(r.index).fillna(0.0)
    return {
        "days": int(len(r)), "first": f"{r.index.min():%Y-%m-%d}", "last": f"{r.index.max():%Y-%m-%d}",
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan"),
        "sharpe": float(r.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "trades": len(t), "hit_rate": float(np.mean([x > 0 for x in t])) if t else float("nan"),
        "days_on": int((pos > 0).sum()), "share_on": float((pos > 0).mean()),
        "turnover_per_year": float(pos.diff().abs().fillna(pos.abs()).sum() / years) if years > 0 else float("nan"),
        "skew": float(stats.skew(r)) if len(r) > 2 else float("nan"),
        "kurtosis": float(stats.kurtosis(r, fisher=False)) if len(r) > 3 else float("nan"),
    }


def random_trades(position: pd.Series, returns: pd.Series, costs: Costs, n: int = 1000,
                  seed: int = 0) -> dict[str, Any]:
    """Random entry dates with the same number and lengths of trades in the same period: the share
    of random strategies whose total net return is at least the real one's (a one-sided p-value)."""
    pos = position.reindex(returns.index).fillna(0.0)
    on = pos > 0
    lengths = [len(part) for _, part in pos[on].groupby((on != on.shift(fill_value=False)).cumsum()[on])]
    if not lengths:
        return {"random_p": float("nan"), "random_mean_return": float("nan")}
    actual = float(np.prod(1 + backtest(pos, returns, costs).to_numpy()) - 1)
    rng = np.random.default_rng(seed)
    days = len(returns)
    totals = []
    for _ in range(n):
        fake = np.zeros(days)
        for length in lengths:
            start = int(rng.integers(0, max(days - length, 1)))
            fake[start:start + length] = 1.0
        net = backtest(pd.Series(fake, index=returns.index), returns, costs)
        totals.append(float(np.prod(1 + net.to_numpy()) - 1))
    totals_arr = np.array(totals)
    return {"random_p": float((totals_arr >= actual).mean()), "random_mean_return": float(totals_arr.mean()),
            "random_trials": n}


def deflated_sharpe(sharpe_annual: float, trial_sharpes_annual: list[float], days: int, skew: float,
                    kurtosis: float) -> float:
    """Probability that the true Sharpe exceeds the best of `len(trial_sharpes)` lucky tries
    (Bailey and López de Prado 2014), from per-period (daily) Sharpe ratios. Kurtosis is the
    non-excess kind (3 for normal returns)."""
    trials = [s for s in trial_sharpes_annual if s is not None and not np.isnan(s)]
    n = len(trials)
    if n < 2 or days < 3 or np.isnan(sharpe_annual):
        return float("nan")
    sr = sharpe_annual / math.sqrt(TRADING_DAYS)
    var = float(np.var(np.array(trials) / math.sqrt(TRADING_DAYS), ddof=1))
    sr0 = math.sqrt(var) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
                            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * math.e)))
    denom = 1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(days - 1) / math.sqrt(denom)))


def event_study(events: pd.DatetimeIndex, prices: pd.Series, horizons_days: tuple[int, ...] = (10, 20, 40)) -> pd.DataFrame:
    """Returns over each horizon (trading days) from the first close on or after each event,
    against the same-horizon returns from every trading day (the unconditional distribution).

    Columns: horizon, events, mean, median, unconditional mean, difference, t (Welch), p (two-sided),
    share up (events) and share up (unconditional).
    """
    p = prices.dropna().sort_index()
    days = p.index
    rows = []
    for h in horizons_days:
        forward = (p.shift(-h) / p - 1).dropna()
        idx = []
        for e in events:
            e = pd.Timestamp(e).tz_localize(None) if pd.Timestamp(e).tzinfo else pd.Timestamp(e)
            i = int(np.searchsorted(days.to_numpy(), e.normalize().to_datetime64(), side="left"))
            if i < len(days) and days[i] in forward.index:
                idx.append(days[i])
        ev = forward.loc[idx]
        rest = forward.drop(index=idx, errors="ignore")
        t, pval = (stats.ttest_ind(ev, rest, equal_var=False) if len(ev) >= 2 else (float("nan"), float("nan")))
        rows.append({"horizon_days": h, "events": len(ev), "mean": float(ev.mean()) if len(ev) else float("nan"),
                     "median": float(ev.median()) if len(ev) else float("nan"),
                     "unconditional_mean": float(rest.mean()), "difference": float(ev.mean() - rest.mean())
                     if len(ev) else float("nan"), "t": float(t), "p": float(pval),
                     "share_up": float((ev > 0).mean()) if len(ev) else float("nan"),
                     "share_up_unconditional": float((rest > 0).mean())})
    return pd.DataFrame(rows)
