"""A hypothesis's one look at the vault, for event studies (SIGNALS.md "H5-vault",
pre-registered 2026-10-07 before any vault data was read). Pure functions.

The test is the cluster-respecting randomisation test of the diagnostics
(`controls.shifted_event_means` and `controls.randomisation_p`, CONTROLS.md §5):
the vault events' dates are moved together by one random number of trading days,
wrapping round inside the window, which keeps their clustering. Here the window is
the vault, one per horizon: the trading days from the vault start whose return over
that horizon is complete. The Welch t-test of `trading.event_study` is reported
next to it for comparison only.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from tracker.backtest import controls as ct
from tracker.backtest import trading as tr

SEED = 20261007
DRAWS = 1000
EPISODE_GAP_DAYS = 56  # 8 weeks: the event rule's calm period and the longest horizon (40 trading days)


def _naive(t: Any) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize(None) if t.tzinfo else t


def complete_days(prices: pd.Series, start: Any, horizon: int) -> pd.DatetimeIndex:
    """Trading days on or after `start` whose return over `horizon` trading days is complete in `prices`."""
    p = prices.dropna().sort_index()
    days = p.index[p.index >= _naive(start)]
    return days[:-horizon] if len(days) > horizon else days[:0]


def episodes(dates: Iterable[Any], gap_days: int = EPISODE_GAP_DAYS) -> list[int]:
    """Episode number (from 1) of each date, in date order: a date less than `gap_days`
    after the previous one joins its episode, so events whose outcomes overlap count once."""
    stamps = sorted(_naive(d) for d in dates)
    out, episode, previous = [], 0, None
    for d in stamps:
        if previous is None or (d - previous).days >= gap_days:
            episode += 1
        out.append(episode)
        previous = d
    return out


def vault_test(events: Iterable[Any], prices: pd.Series, start: Any, horizon: int,
               draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    """One pre-registered test: the mean return over `horizon` trading days after the events
    (from the first close on or after each), against placebo sets that move all the events
    together inside the window of trading days from `start` with a complete return.

    Returns the events counted, their dates and returns, the mean, the placebo median,
    the two-sided randomisation p, the side of the median the mean is on (+1 above, -1 below),
    the number of independent episodes, and the Welch t-test against every other day in the window.
    A fresh generator per test (`seed`), so each p-value doesn't depend on which other tests ran.
    """
    p = prices.dropna().sort_index()
    window = complete_days(p, start, horizon)
    forward = (p.shift(-horizon) / p - 1).reindex(window)
    counted = []
    for e in events:
        i = int(np.searchsorted(window.to_numpy(), _naive(e).normalize().to_datetime64(), side="left"))
        if i < len(window) and _naive(e) >= _naive(start):
            counted.append((pd.Timestamp(e), window[i], float(forward.iloc[i])))
    out: dict[str, Any] = {"horizon_days": horizon, "events": len(counted), "window_days": len(window),
                           "window_first": window[0] if len(window) else None,
                           "window_last": window[-1] if len(window) else None,
                           "event_dates": [c[0] for c in counted], "event_returns": [c[2] for c in counted],
                           "episodes": max(episodes([c[0] for c in counted]), default=0)}
    if not counted:
        return {**out, "mean": float("nan"), "placebo_median": float("nan"), "p": float("nan"), "side": 0,
                "welch_t": float("nan"), "welch_p": float("nan"), "other_days_mean": float("nan")}
    mean = float(np.mean(out["event_returns"]))
    end = window[-1] + pd.Timedelta(days=1)
    placebo = ct.shifted_event_means(pd.DatetimeIndex([c[1] for c in counted]), p, window[0], end,
                                     np.random.default_rng(seed), draws=draws, horizons=(horizon,))[horizon]
    median = float(np.median(placebo))
    # Prices from the window's first day: every other day with a complete return is a window day.
    welch = tr.event_study(pd.DatetimeIndex([c[1] for c in counted]), p[p.index >= window[0]], (horizon,)).iloc[0]
    return {**out, "mean": mean, "placebo_median": median, "p": ct.randomisation_p(mean, placebo),
            "side": int(np.sign(mean - median)), "welch_t": float(welch["t"]), "welch_p": float(welch["p"]),
            "other_days_mean": float(welch["unconditional_mean"])}


def format_p(p: float, draws: int = DRAWS) -> str:
    """A randomisation p-value for text: below the resolution of `draws` placebo sets it reads '< 0.001'."""
    if p != p:
        return "–"
    return f"< {1 / draws:g}" if p < 1 / draws else f"{p:.3f}"


def overlap_share(horizon: int, window_days: int) -> float:
    """Share of placebo shifts (uniform over 1..window-1, wrapping) that move the events by fewer than
    `horizon` trading days either way, so their outcome days overlap the real events'."""
    return min(1.0, 2 * (horizon - 1) / max(window_days - 1, 1))


def verdict(result: dict[str, Any], training_side: int = 1, alpha: float = 0.05) -> tuple[bool, str]:
    """(passed, plain-English reason) for the primary test: same side of the placebo median as
    in training and p < `alpha`; no event counted means not testable, which is not a pass."""
    if not result.get("events"):
        return False, "Not testable on the vault: no event with a complete return at the primary horizon."
    same = result["side"] == training_side
    significant = result["p"] < alpha
    where = "above" if result["side"] > 0 else "below" if result["side"] < 0 else "at"
    trained = "above" if training_side > 0 else "below"
    if same and significant:
        return True, (f"Passed: the mean return after the vault events is {where} the placebo median, as in training, "
                      f"with p {format_p(result['p'])}, below {alpha}.")
    reasons = []
    if not same:
        reasons.append(f"the mean is {where} the placebo median (training: {trained})")
    if not significant:
        reasons.append(f"p {format_p(result['p'])} is not below {alpha}")
    return False, "Failed: " + " and ".join(reasons) + "."
