"""Composite country activity scores (S19; PLAN §5.5, SIGNALS.md §3).

For each country, on a weekly grid that ends on the snapshot date:

1. **Components.** Each alternative indicator's robust z of its YoY change,
   sign-flipped where `higher_is_stronger` is false:
   * trade: PortWatch export and import volumes, `yoy>rz(3y)`;
   * energy: weather-normalised total and weekday-overnight load (ENTSO-E,
     NESO, EIA), `wx>yoy>rz(3y)` (S18), and Ember monthly demand, `yoy>rz(3y)`.
   A component counts at a week if its latest value is recent enough
   (MAX_AGE_DAYS), and only for countries `Registry.signal_countries()` keeps
   (PLAN D22: ZAF has no energy category).
2. **Categories.** The weighted mean of the components within trade and within
   energy. A daily component whose YoY window overlaps a moving holiday gets
   half weight (SIGNALS.md §1).
3. **Score.** The mean of the category scores, reported with its coverage (the
   number of contributing components); it needs at least two.
4. **Score z.** The robust z of the weekly score over its own trailing 3 years
   (at least 2 years of scores), the "z(activity score)" that divergences use.

Everything is computed from one snapshot (the data as known on a date), so the
same functions serve the daily `signals` command (latest snapshot) and the
backtest engine (`PANEL_SIGNALS`, PLAN D39), which keeps each country's end
value. Pure: frames in, frames out.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tracker.signals import chain as chains
from tracker.signals import transforms as tf
from tracker.signals import weather as wx
from tracker.signals.moving_holidays import holiday_windows

if TYPE_CHECKING:
    from tracker.registry import Registry


@dataclass(frozen=True)
class Component:
    """One alternative indicator in the composite: its category and the chain that scores it."""

    indicator: str
    category: str
    chain: str


LOAD_CHAIN = "wx>yoy>rz(3y)"
COMPONENTS: tuple[Component, ...] = (
    Component("portwatch_exports", "trade", "yoy>rz(3y)"),
    Component("portwatch_imports", "trade", "yoy>rz(3y)"),
    Component("entsoe_load", "energy", LOAD_CHAIN),
    Component("entsoe_overnight_load", "energy", LOAD_CHAIN),
    Component("neso_load", "energy", LOAD_CHAIN),
    Component("neso_overnight_load", "energy", LOAD_CHAIN),
    Component("eia_us48_load", "energy", LOAD_CHAIN),
    Component("eia_us48_overnight_load", "energy", LOAD_CHAIN),
    Component("ember_demand", "energy", "yoy>rz(3y)"),
)
CATEGORIES = ("trade", "energy")
# A latest value older than this (days since its period ended) doesn't count at a week.
MAX_AGE_DAYS = {"D": 45, "W": 60, "M": 130, "Q": 220}
MIN_COVERAGE = 2
HOLIDAY_WEIGHT = 0.5
SCORE_Z_YEARS = 3
SCORE_Z_MIN_MONTHS = 24
GRID_START = pd.Timestamp("2016-01-04")  # the weekly grid never starts earlier (no component does)

COMPOSITE_SIGNALS = ("activity", "activity_trade", "activity_energy", "activity_z")


def needed_indicators() -> list[str]:
    """Every indicator the composite reads (components and the degree days for `wx`)."""
    return sorted({c.indicator for c in COMPONENTS} | {wx.HDD, wx.CDD})


def week_grid(as_of: pd.Timestamp, start: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Weekly dates ending on the snapshot's day (tz-naive), every 7 days back to `start`."""
    end = pd.Timestamp(as_of)
    end = (end.tz_convert("UTC").tz_localize(None) if end.tzinfo else end).normalize()
    first = max(pd.Timestamp(start) if start is not None else GRID_START, GRID_START)
    if first > end:
        return pd.DatetimeIndex([])
    weeks = (end - first).days // 7
    return pd.DatetimeIndex([end - pd.Timedelta(days=7 * k) for k in range(weeks, -1, -1)])


def period_end(values: pd.Series, frequency: str) -> pd.Series:
    """The series re-dated to the last day of each period (monthly values count from their month's end)."""
    shift = {"D": None, "W": pd.Timedelta(days=6), "M": pd.offsets.MonthEnd(0), "Q": pd.offsets.QuarterEnd(0)}[frequency]
    if shift is None:
        return values
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(values.index) + shift)


def as_of_grid(values: pd.Series, grid: pd.DatetimeIndex, max_age_days: int) -> tuple[pd.Series, pd.Series]:
    """Each grid date's latest value on or before it, and that value's date; NaN where it's
    older than `max_age_days` or there is none."""
    s = values.dropna().sort_index()
    if s.empty or not len(grid):
        return pd.Series(np.nan, index=grid), pd.Series(pd.NaT, index=grid)
    pos = np.searchsorted(s.index.to_numpy(), grid.to_numpy(), side="right") - 1
    ok = pos >= 0
    vals = np.full(len(grid), np.nan)
    dates = np.full(len(grid), np.datetime64("NaT"), dtype="datetime64[us]")
    vals[ok] = s.to_numpy()[pos[ok]]
    dates[ok] = s.index.to_numpy().astype("datetime64[us]")[pos[ok]]
    age = (grid.to_numpy().astype("datetime64[us]") - dates).astype("timedelta64[D]").astype(float)
    fresh = ok & (age <= max_age_days)
    vals[~fresh] = np.nan
    return pd.Series(vals, index=grid), pd.Series(np.where(fresh, dates, np.datetime64("NaT")), index=grid)


# ---------------------------------------------------------------- components


@lru_cache(maxsize=512)
def _windows(calendar: str, holidays: tuple[str, ...], first_year: int, last_year: int) -> pd.DataFrame:
    return holiday_windows(calendar, list(holidays), first_year, last_year)


_SCORE_CACHE: dict[str, pd.Series] = {}
_SCORE_CACHE_SIZE = 20000


def _digest(*series: pd.Series) -> str:
    h = hashlib.sha1()
    for s in series:
        h.update(np.asarray(pd.DatetimeIndex(s.index).asi8).tobytes())
        h.update(s.to_numpy(dtype="float64").tobytes())
        h.update(b"|")
    return h.hexdigest()


def _series(groups: dict[tuple[str, str], pd.DataFrame], indicator: str, country: str) -> pd.Series | None:
    part = groups.get((indicator, country))
    if part is None or part.empty:
        return None
    return pd.Series(part["value"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(part["date"])).sort_index()


def component_score(registry: "Registry", component: Component, country: str, values: pd.Series,
                    extra: dict[str, pd.Series]) -> pd.Series:
    """The component's z-score series (direction-adjusted). Empty if the chain can't run yet.

    Memoised by the exact input, so a series that hasn't changed since an
    earlier snapshot (monthly data tested weekly) isn't recomputed.
    """
    key = "|".join((component.chain, component.indicator, country, _digest(values, *extra.values())))
    found = _SCORE_CACHE.get(key)
    if found is not None:
        return found
    ind = registry.indicator(component.indicator)
    ref = registry.countries.get(country)
    windows = None
    if ref is not None and ref.moving_holidays and len(values):
        windows = _windows(ref.holiday_calendar, tuple(ref.moving_holidays), values.index.min().year,
                           values.index.max().year)
    ctx = chains.context_for(registry, component.indicator, country, windows, extra)
    try:
        z = chains.apply(component.chain, values, ctx).values
    except chains.ChainError:
        z = pd.Series(dtype="float64")
    if not ind.higher_is_stronger:
        z = -z
    if len(_SCORE_CACHE) >= _SCORE_CACHE_SIZE:
        _SCORE_CACHE.clear()
    _SCORE_CACHE[key] = z
    return z


def _holiday_weight(registry: "Registry", component: Component, country: str, dates: pd.Series) -> np.ndarray:
    """HOLIDAY_WEIGHT where a daily component's YoY window (dated `dates`) overlaps a moving holiday."""
    ind = registry.indicator(component.indicator)
    ref = registry.countries.get(country)
    weight = np.ones(len(dates))
    valid = dates.notna().to_numpy()
    if ind.frequency != "D" or ref is None or not ref.moving_holidays or not valid.any():
        return weight
    known = pd.DatetimeIndex(dates[valid])
    windows = _windows(ref.holiday_calendar, tuple(ref.moving_holidays), known.min().year - 1, known.max().year)
    flagged = tf.holiday_flags(known, windows).to_numpy()
    weight[np.flatnonzero(valid)[flagged]] = HOLIDAY_WEIGHT
    return weight


def component_frame(snapshot: pd.DataFrame, as_of: Any, registry: "Registry") -> pd.DataFrame:
    """Every component's z on the weekly grid, per country.

    Columns: date (grid date), country, indicator, category, z, value_date (the
    date of the value used), weight (1, or HOLIDAY_WEIGHT in a moving-holiday window).
    """
    grid = week_grid(as_of)
    groups = {key: part for key, part in snapshot.groupby(["indicator", "country"], sort=False)}
    rows = []
    for comp in COMPONENTS:
        if comp.indicator not in registry.indicators:
            continue
        ind = registry.indicator(comp.indicator)
        for country in registry.signal_countries(ind):
            values = _series(groups, comp.indicator, country)
            if values is None:
                continue
            extra = {}
            if "wx" in comp.chain:
                for dd in (wx.HDD, wx.CDD):
                    found = _series(groups, dd, country)
                    if found is not None:
                        extra[dd] = found
            z = component_score(registry, comp, country, values, extra)
            if z.empty:
                continue
            z = period_end(z, ind.frequency)
            start = max(z.index.min(), grid.min()) if len(grid) else None
            g = grid[grid >= start] if start is not None else grid
            val, when = as_of_grid(z, g, MAX_AGE_DAYS[ind.frequency])
            keep = val.notna().to_numpy()
            if not keep.any():
                continue
            rows.append(pd.DataFrame({
                "date": g[keep], "country": country, "indicator": comp.indicator, "category": comp.category,
                "z": val.to_numpy()[keep], "value_date": when.to_numpy()[keep],
                "weight": _holiday_weight(registry, comp, country, when)[keep],
            }))
    columns = ["date", "country", "indicator", "category", "z", "value_date", "weight"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


# ------------------------------------------------------------------- scores


def _weighted(part: pd.DataFrame) -> float:
    return float(np.average(part["z"], weights=part["weight"]))


def scores(components: pd.DataFrame) -> pd.DataFrame:
    """Category scores, the country score and its coverage per (date, country).

    Columns: date, country, trade, energy, score, coverage (components that
    contributed), n_trade, n_energy. The score is NaN with fewer than
    MIN_COVERAGE components.
    """
    columns = ["date", "country", *CATEGORIES, "score", "coverage", "n_trade", "n_energy"]
    if components.empty:
        return pd.DataFrame(columns=columns)
    c = components.assign(wz=components["z"] * components["weight"])
    sums = c.groupby(["date", "country", "category"]).agg(wz=("wz", "sum"), w=("weight", "sum"), n=("z", "size"))
    cat = (sums["wz"] / sums["w"]).unstack("category")
    counts = sums["n"].unstack("category").fillna(0).astype(int)
    out = pd.DataFrame(index=cat.index)
    for name in CATEGORIES:
        out[name] = cat[name] if name in cat else np.nan
        out[f"n_{name}"] = counts[name] if name in counts else 0
    out["coverage"] = out["n_trade"] + out["n_energy"]
    out["score"] = out[list(CATEGORIES)].mean(axis=1, skipna=True).where(out["coverage"] >= MIN_COVERAGE)
    return out.reset_index()[columns]


def score_z(score: pd.DataFrame) -> pd.Series:
    """Robust z of each country's weekly score over its trailing 3 years (at least 2 years of scores),
    aligned with `score`'s rows."""
    out = pd.Series(np.nan, index=score.index)
    for country, part in score.groupby("country", sort=False):
        s = pd.Series(part["score"].to_numpy(), index=pd.DatetimeIndex(part["date"])).dropna()
        if s.empty:
            continue
        z = tf.robust_z(s, "W", window_years=SCORE_Z_YEARS, min_history_months=SCORE_Z_MIN_MONTHS)
        out.loc[part.index] = z.reindex(pd.DatetimeIndex(part["date"])).to_numpy()
    return out


def _components_json(components: pd.DataFrame) -> pd.Series:
    """One JSON string per (date, country): each component's z, weight and value date."""
    if components.empty:
        return pd.Series(dtype=object)
    records = components.assign(value_date=components["value_date"].dt.strftime("%Y-%m-%d"),
                                z=components["z"].round(3))
    return records.groupby(["date", "country"]).apply(
        lambda p: json.dumps({r.indicator: {"z": r.z, "w": r.weight, "date": r.value_date, "cat": r.category}
                              for r in p.itertuples()}), include_groups=False)


def activity_frame(snapshot: pd.DataFrame, as_of: Any, registry: "Registry") -> pd.DataFrame:
    """The composite's weekly history from one snapshot: rows (date, country, signal, value, components)
    for COMPOSITE_SIGNALS, plus each component's z as signal `z:<indicator>` (anomaly alerts)."""
    comps = component_frame(snapshot, as_of, registry)
    sc = scores(comps)
    if sc.empty:
        return pd.DataFrame(columns=["date", "country", "signal", "value", "components"])
    sc["activity_z"] = score_z(sc)
    detail = _components_json(comps)
    sc = sc.join(detail.rename("components"), on=["date", "country"])
    meta = sc.apply(lambda r: json.dumps({"coverage": int(r["coverage"]), "n_trade": int(r["n_trade"]),
                                          "n_energy": int(r["n_energy"])}), axis=1)
    parts = []
    for signal, column, comp in (("activity", "score", sc["components"]), ("activity_trade", "trade", meta),
                                 ("activity_energy", "energy", meta), ("activity_z", "activity_z", meta)):
        parts.append(pd.DataFrame({"date": sc["date"], "country": sc["country"], "signal": signal,
                                   "value": sc[column].astype("float64"), "components": comp}))
    z_rows = comps.assign(signal="z:" + comps["indicator"], value=comps["z"],
                          components=comps.apply(lambda r: json.dumps({"date": f"{r['value_date']:%Y-%m-%d}",
                                                                       "w": r["weight"]}), axis=1))
    parts.append(z_rows[["date", "country", "signal", "value", "components"]])
    out = pd.concat(parts, ignore_index=True)
    return out[out["value"].notna()].reset_index(drop=True)


# ---------------------------------------------------- backtest hook (PLAN D39)

_LAST: dict[str, Any] = {}


def cached_activity(snapshot: pd.DataFrame, as_of: Any, registry: "Registry") -> pd.DataFrame:
    """`activity_frame`, computed once per snapshot object (the panel signals share it)."""
    key = (pd.Timestamp(as_of), id(registry))
    if _LAST.get("snapshot") is snapshot and _LAST.get("key") == key:
        return _LAST["frame"]
    frame = activity_frame(snapshot, as_of, registry)
    _LAST.update(snapshot=snapshot, key=key, frame=frame)
    return frame


def _panel(signal: str):
    def run(snapshot: pd.DataFrame, as_of: pd.Timestamp, registry: "Registry") -> pd.DataFrame:
        frame = cached_activity(snapshot, as_of, registry)
        return frame.loc[frame["signal"] == signal, ["date", "country", "value", "components"]]
    run.__name__ = f"panel_{signal}"
    run.__doc__ = f"Panel signal `{signal}` (composite.py) for the backtest engine."
    return run


PANEL_SIGNALS = {name: _panel(name) for name in COMPOSITE_SIGNALS}
