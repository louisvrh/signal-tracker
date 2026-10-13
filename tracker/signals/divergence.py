"""Divergences and flags (S19; PLAN §5.5, SIGNALS.md §4).

On the composite's weekly grid (composite.py), per country:

* **Market divergence** D_mkt = z(activity score) − z(market), where the market
  side is the country ETF's 3-month return minus URTH's (MSCI World), robust-z
  scored over 3 years of trading days. Strongly negative: real activity is
  weakening but the market hasn't reacted (a short candidate).
* **Official divergence** D_off = z(activity score) − z(latest official YoY):
  the robust z of the country's most recent official print, industrial
  production where there is a monthly series, otherwise official exports
  (OFFICIAL_IP, OFFICIAL_EXPORTS). Strongly negative points to a downside
  surprise in the next official release.
* **Flags** (`flag_states`): on once D < −1.5 for 3 consecutive weekly
  observations, off once D > −0.5 (hysteresis). Starting thresholds; tune them
  on training data only.

Like composite.py, everything comes from one snapshot, so the daily `signals`
command and the backtest engine (`PANEL_SIGNALS`, PLAN D39) share the code.
Pure: frames in, frames out.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tracker.signals import chain as chains
from tracker.signals import composite as comp
from tracker.signals import transforms as tf

if TYPE_CHECKING:
    from tracker.registry import Registry

ETF = "prices_country_etf"
BENCHMARK = "prices_benchmark"
BENCHMARK_COUNTRY = "WLD"
RETURN_DAYS = 91  # "3 months" of calendar days
MARKET_MAX_AGE_DAYS = 10
OFFICIAL_IP = ("eurostat_ip", "oecd_ip", "fred_indpro", "oecd_ip_manufacturing")
OFFICIAL_EXPORTS = ("comtrade_exports", "imf_exports", "comtrade_mirror_exports")
OFFICIAL_CHAIN = "yoy>rz(3y)"
MIRROR = "comtrade_mirror_exports"
MIRROR_REPORTERS = "comtrade_mirror_reporters"
MIRROR_MIN_SHARE = 0.8  # a mirror month counts once this share of the usual reporters has filed
FLAG_ON, FLAG_OFF, FLAG_WEEKS = -1.5, -0.5, 3
DIVERGENCE_SIGNALS = ("d_mkt", "d_off", "market_z", "official_z")


def needed_indicators() -> list[str]:
    """Every indicator the divergences read, on top of the composite's."""
    return sorted({ETF, BENCHMARK, MIRROR_REPORTERS, *OFFICIAL_IP, *OFFICIAL_EXPORTS} | set(comp.needed_indicators()))


def complete_mirror_months(values: pd.Series, reporters: pd.Series | None) -> pd.Series:
    """Taiwan's mirror trade without its incomplete latest months: a month counts once at least
    MIRROR_MIN_SHARE of the median reporter count of the 12 months before it has filed (NOTES, S10)."""
    if reporters is None or reporters.empty:
        return values
    usual = reporters.shift(1).rolling(12, min_periods=6).median()
    complete = reporters >= MIRROR_MIN_SHARE * usual
    return values[complete.reindex(values.index, fill_value=False) | usual.reindex(values.index).isna()]


# ------------------------------------------------------------------ market


def three_month_return(prices: pd.Series, days: int = RETURN_DAYS) -> pd.Series:
    """Return (%) over the last `days` calendar days at each trading day: against the last
    close on or before that earlier date."""
    p = tf.clean(prices)
    if p.empty:
        return p
    earlier = p.index - pd.Timedelta(days=days)
    pos = np.searchsorted(p.index.to_numpy(), earlier.to_numpy(), side="right") - 1
    base = np.where(pos >= 0, p.to_numpy()[np.maximum(pos, 0)], np.nan)
    ok = p.index >= p.index.min() + pd.Timedelta(days=days)
    return pd.Series(np.where(ok, (p.to_numpy() / base - 1) * 100, np.nan), index=p.index).dropna()


def market_z(etf: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Robust z (3 years of trading days) of the ETF's 3-month return minus the benchmark's."""
    excess = (three_month_return(etf) - three_month_return(benchmark)).dropna()
    return tf.robust_z(excess, "D") if not excess.empty else excess


# ---------------------------------------------------------------- official


def official_z(snapshot_groups: dict, registry: "Registry", country: str,
               grid: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """z of the latest official YoY at each grid date, and which indicator it came from.

    Industrial production if the country has a fresh monthly series;
    otherwise the official export series with the most recent fresh print.
    """
    candidates = {}
    for ind_id in (*OFFICIAL_IP, *OFFICIAL_EXPORTS):
        values = comp._series(snapshot_groups, ind_id, country)
        if values is None or ind_id not in registry.indicators:
            continue
        if ind_id == MIRROR:
            values = complete_mirror_months(values, comp._series(snapshot_groups, MIRROR_REPORTERS, country))
        ctx = chains.context_for(registry, ind_id, country)
        try:
            z = chains.apply(OFFICIAL_CHAIN, values, ctx).values
        except chains.ChainError:
            continue
        if z.empty:
            continue
        ind = registry.indicator(ind_id)
        z = comp.period_end(z, ind.frequency)
        candidates[ind_id] = comp.as_of_grid(z, grid, comp.MAX_AGE_DAYS[ind.frequency])
    value = pd.Series(np.nan, index=grid)
    source = pd.Series(None, index=grid, dtype=object)
    for ind_id in OFFICIAL_IP:  # IP first, in order
        if ind_id in candidates:
            v, _ = candidates[ind_id]
            take = value.isna() & v.notna()
            value[take], source[take] = v[take], ind_id
    exports = [i for i in OFFICIAL_EXPORTS if i in candidates]
    if exports:  # otherwise the freshest export print
        dates = pd.concat({i: candidates[i][1] for i in exports}, axis=1)
        vals = pd.concat({i: candidates[i][0] for i in exports}, axis=1)
        best = dates.apply(lambda r: r.dropna().idxmax() if r.notna().any() else None, axis=1)
        for i in exports:
            take = value.isna() & (best == i) & vals[i].notna()
            value[take], source[take] = vals[i][take], i
    return value, source


# ------------------------------------------------------------------- flags


def flag_states(values: pd.Series, on: float = FLAG_ON, off: float = FLAG_OFF, weeks: int = FLAG_WEEKS) -> pd.Series:
    """Persistence and hysteresis: a flag switches on at the `weeks`-th consecutive weekly value
    below `on`, and off at the first value above `off`. A missing value breaks a run and leaves
    an active flag as it is. Returns a boolean series on `values`' index."""
    state, run, out = False, 0, []
    for v in values.to_numpy(dtype="float64"):
        if np.isnan(v):
            run = 0
        else:
            run = run + 1 if v < on else 0
            if not state and run >= weeks:
                state = True
            elif state and v > off:
                state = False
        out.append(state)
    return pd.Series(out, index=values.index, dtype=bool)


def flag_table(signals: pd.DataFrame, names: tuple[str, ...] = ("d_mkt", "d_off")) -> pd.DataFrame:
    """Flag state per (country, divergence) on its weekly history, with on/off events.

    Columns: date, country, signal, value, flag, event ("on", "off" or None).
    """
    rows = []
    for (country, name), part in signals[signals["signal"].isin(names)].groupby(["country", "signal"]):
        part = part.sort_values("date")
        values = pd.Series(part["value"].to_numpy(), index=pd.DatetimeIndex(part["date"]))
        state = flag_states(values)
        before = state.shift(1, fill_value=False)
        event = np.where(state & ~before, "on", np.where(~state & before, "off", None))
        rows.append(pd.DataFrame({"date": values.index, "country": country, "signal": name,
                                  "value": values.to_numpy(), "flag": state.to_numpy(), "event": event}))
    columns = ["date", "country", "signal", "value", "flag", "event"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


# ------------------------------------------------------------ divergences


def divergence_frame(snapshot: pd.DataFrame, as_of: Any, registry: "Registry",
                     activity: pd.DataFrame | None = None) -> pd.DataFrame:
    """D_mkt and D_off (and their market and official sides) on the weekly grid from one snapshot:
    rows (date, country, signal, value, components)."""
    activity = comp.cached_activity(snapshot, as_of, registry) if activity is None else activity
    act = activity[activity["signal"] == "activity_z"]
    grid = comp.week_grid(as_of)
    groups = {key: part for key, part in snapshot.groupby(["indicator", "country"], sort=False)}
    bench = comp._series(groups, BENCHMARK, BENCHMARK_COUNTRY)
    parts = []
    for country, a in act.groupby("country", sort=True):
        a_z = pd.Series(a["value"].to_numpy(), index=pd.DatetimeIndex(a["date"]))
        g = grid[grid >= a_z.index.min()]
        a_z = a_z.reindex(g)
        etf = comp._series(groups, ETF, country)
        if etf is not None and bench is not None:
            m_z, m_date = comp.as_of_grid(market_z(etf, bench), g, MARKET_MAX_AGE_DAYS)
            info = [json.dumps({"activity_z": round(float(x), 3), "market_z": round(float(y), 3),
                                "etf": registry.countries[country].country_etf, "price_date": f"{d:%Y-%m-%d}"})
                    if pd.notna(y) and pd.notna(x) else None for x, y, d in zip(a_z, m_z, m_date)]
            parts.append(pd.DataFrame({"date": g, "country": country, "signal": "market_z", "value": m_z.to_numpy(),
                                       "components": info}))
            parts.append(pd.DataFrame({"date": g, "country": country, "signal": "d_mkt",
                                       "value": (a_z - m_z).to_numpy(), "components": info}))
        o_z, o_src = official_z(groups, registry, country, g)
        info = [json.dumps({"activity_z": round(float(x), 3), "official_z": round(float(y), 3), "official": s})
                if pd.notna(y) and pd.notna(x) else None for x, y, s in zip(a_z, o_z, o_src)]
        parts.append(pd.DataFrame({"date": g, "country": country, "signal": "official_z", "value": o_z.to_numpy(),
                                   "components": info}))
        parts.append(pd.DataFrame({"date": g, "country": country, "signal": "d_off",
                                   "value": (a_z - o_z).to_numpy(), "components": info}))
    if not parts:
        return pd.DataFrame(columns=["date", "country", "signal", "value", "components"])
    out = pd.concat(parts, ignore_index=True)
    return out[out["value"].notna()].reset_index(drop=True)


def signals_frame(snapshot: pd.DataFrame, as_of: Any, registry: "Registry") -> pd.DataFrame:
    """Every S19 signal's weekly history from one snapshot: the composite (composite.py) and the
    divergences. What the daily `signals` command writes, and what the panel signals read."""
    key = (pd.Timestamp(as_of), id(registry))
    if _LAST.get("snapshot") is snapshot and _LAST.get("key") == key:
        return _LAST["frame"]
    activity = comp.cached_activity(snapshot, as_of, registry)
    frame = pd.concat([activity, divergence_frame(snapshot, as_of, registry, activity)], ignore_index=True)
    _LAST.update(snapshot=snapshot, key=key, frame=frame)
    return frame


_LAST: dict[str, Any] = {}


def _panel(signal: str):
    def run(snapshot: pd.DataFrame, as_of: pd.Timestamp, registry: "Registry") -> pd.DataFrame:
        frame = signals_frame(snapshot, as_of, registry)
        return frame.loc[frame["signal"] == signal, ["date", "country", "value", "components"]]
    run.__name__ = f"panel_{signal}"
    run.__doc__ = f"Panel signal `{signal}` (divergence.py) for the backtest engine."
    return run


PANEL_SIGNALS = {name: _panel(name) for name in DIVERGENCE_SIGNALS}
