"""Nowcast tests (S23; PLAN §5.6 items 2-5 and 8, SIGNALS.md H1, H1b, H2, H2b).

On each monthly test date the backtest engine's snapshot (the data as known
then, D39) gives two things per country:

* the **signal**: the alternative series' monthly YoY for its latest complete
  month m (daily values averaged per month, at least 90% of the days present;
  weather-normalised first for power demand, `wx`, D45; the same monthly YoY
  rules as official data: Lunar New Year January-February combined, no
  comparison across a known break);
* the **target's history**: the official YoY values published by then; the
  latest is month L < m.

If m's official value isn't out yet, every model forecasts it, fitted only on
months whose official value was published by the test date, with each month's
signal as first computed (its real-time value):

* `arx` (primary): y_k = a + b·x_k + c·y_(k−h), h = m − L, per country;
* `ar` (benchmark, §5.6): the same without the signal, y_k = a + c·y_(k−h);
* `x`: the signal alone, y_k = a + b·x_k;
* `pooled`: `arx` on all countries together with an intercept per country;
* `rw`: the last published value, y_L.

Forecasts are scored against the official value as known when the vault
starts, so nothing published after that date is ever read (the vault stays
closed). Pure: vintages and specs in, frames out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from tracker import store
from tracker.backtest import pit
from tracker.signals import chain as chains
from tracker.signals import transforms as tf
from tracker.signals import weather as wx

if TYPE_CHECKING:
    from tracker.registry import Registry

MIN_MONTH_SHARE = 0.9  # a month's average needs this share of its expected days
MIN_TRAIN = 24  # months with both a signal and a published official value
DM_LAGS = 3  # Newey-West lags for the Diebold-Mariano test (forecast horizons are 1-4 months)
MODELS = ("arx", "ar", "x", "pooled", "rw")


@dataclass(frozen=True)
class NowcastSpec:
    """One nowcast hypothesis.

    Attributes:
        hypothesis: id as in SIGNALS.md (H1, H1b, H2, H2b).
        signal: the alternative indicator (daily).
        targets: country -> official indicator (monthly).
        weather: weather-normalise the signal first (`wx`, power demand only).
        exclude_2020: drop target months in 2020 from training and scoring (§5.6 item 8).
        countries: the countries tested (default: every key of `targets`).
        signal_from: placebo controls only: target country -> the country whose signal is used
            (default: its own).
        shift_months: placebo controls only: each month's signal is the value from this many
            months earlier (x'_m = x_(m−k)), wrapping round from the series' end (`shift_values`),
            so the signal's months, and the horizon, stay the same; 0 = as is.
    """

    hypothesis: str
    signal: str
    targets: dict[str, str]
    weather: bool = False
    exclude_2020: bool = False
    countries: tuple[str, ...] = field(default=())
    signal_from: dict[str, str] = field(default_factory=dict)
    shift_months: int = 0

    def tested(self) -> tuple[str, ...]:
        return self.countries or tuple(self.targets)


# ------------------------------------------------------------------ features


def monthly_mean(values: pd.Series, calendar: str = "all", min_share: float = MIN_MONTH_SHARE) -> pd.Series:
    """Average per day of each calendar month (indexed by month start), for months with at least
    `min_share` of their expected days (weekdays only for weekday series)."""
    s = tf.clean(values)
    if s.empty:
        return s
    month = s.index.to_period("M")
    groups = s.groupby(month)
    mean, count = groups.mean(), groups.size()
    periods = mean.index
    if calendar == "weekdays":
        expected = np.array([np.busday_count(p.start_time.date(), (p.end_time + pd.Timedelta(days=1)).date())
                             for p in periods])
    else:
        expected = np.array([p.days_in_month for p in periods])
    ok = count.to_numpy() >= min_share * expected
    out = pd.Series(mean.to_numpy()[ok], index=periods[ok].to_timestamp())
    return out


def signal_yoy(values: pd.Series, ctx: chains.Context, weather: bool) -> pd.Series:
    """The signal's monthly YoY (%): daily values (weather-normalised if asked), averaged per month,
    against the same month a year earlier (Lunar New Year and breaks as for official data). A
    monthly signal (e.g. a positive control's official series) is used as it is."""
    if ctx.frequency == "M":
        monthly = tf.clean(values)
    else:
        daily = chains.apply("wx", values, ctx).values if weather else tf.clean(values)
        monthly = monthly_mean(daily, ctx.calendar)
    return tf.yoy(monthly, "M", lunar_new_year=ctx.lunar_new_year, breaks=ctx.breaks)


def shift_values(x: pd.Series, months: int) -> pd.Series:
    """Placebo: each month gets the value from `months` earlier (x'_m = x_(m−k)), wrapping round
    from the series' end for its first months, so the series keeps its months (and the forecast
    month and horizon don't change). Uses only values the snapshot already holds."""
    if months == 0 or x.empty:
        return x
    index = pd.DatetimeIndex(x.index)
    first = index.min()
    span = _months_between(index.max(), first) + 1
    by_pos = dict(zip((_months_between(d, first) for d in index), x.to_numpy()))
    values = [by_pos.get((_months_between(d, first) - months) % span, np.nan) for d in index]
    return pd.Series(values, index=index, name=x.name).dropna()


def official_yoy(values: pd.Series, ctx: chains.Context) -> pd.Series:
    """An official monthly series' YoY (%), with the same rules."""
    return tf.yoy(tf.clean(values), "M", lunar_new_year=ctx.lunar_new_year, breaks=ctx.breaks)


def _series(snap: pd.DataFrame, indicator: str, country: str) -> pd.Series:
    part = snap[(snap["indicator"] == indicator) & (snap["country"] == country)]
    return pit.as_series(part) if not part.empty else pd.Series(dtype="float64")


def snapshot_features(snap: pd.DataFrame, spec: NowcastSpec, registry: "Registry") -> dict[str, tuple[pd.Series, pd.Series]]:
    """Per country: (signal YoY, official YoY) as computable from one snapshot."""
    out = {}
    for country in spec.tested():
        source = spec.signal_from.get(country, country)  # placebo controls may borrow another country's signal
        values = _series(snap, spec.signal, source)
        target = spec.targets[country]
        labels = _series(snap, target, country)
        if values.empty:
            continue
        extra = {}
        if spec.weather:
            extra = {dd: _series(snap, dd, source) for dd in (wx.HDD, wx.CDD)}
            if any(s.empty for s in extra.values()):
                continue
        try:
            x = signal_yoy(values, chains.context_for(registry, spec.signal, source, extra=extra), spec.weather)
        except chains.ChainError:
            continue
        y = official_yoy(labels, chains.context_for(registry, target, country)) if not labels.empty \
            else pd.Series(dtype="float64")
        out[country] = (shift_values(x.dropna(), spec.shift_months), y.dropna())
    return out


# -------------------------------------------------------------------- models


def _months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def _fit_predict(X: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> float:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(x_new @ beta)


def training_rows(x_first: pd.Series, y: pd.Series, h: int, exclude_2020: bool) -> pd.DataFrame:
    """Months k with a real-time signal x_k, a published y_k and y_(k−h): columns x, y, y_lag."""
    if x_first.empty or y.empty:  # e.g. a country with port data but no official series (TWN in IMF)
        return pd.DataFrame(columns=["x", "y", "y_lag"], dtype="float64")
    lag = pd.Series(y.to_numpy(), index=pd.DatetimeIndex(y.index) + pd.DateOffset(months=h))
    rows = pd.concat([x_first.rename("x"), y.rename("y"), lag.rename("y_lag")], axis=1, sort=True).dropna()
    if exclude_2020:
        rows = rows[rows.index.year != 2020]
    return rows


def forecast_country(x_first: pd.Series, x_now: pd.Series, y: pd.Series, exclude_2020: bool) -> dict[str, Any] | None:
    """Every single-country model's forecast for the latest signal month without a published
    official value; None if there is none or too little training data."""
    if x_now.empty or y.empty:
        return None
    month, last = x_now.index.max(), y.index.max()
    if month <= last:
        return None  # the official value is already out: nothing to nowcast
    if exclude_2020 and month.year == 2020:
        return None
    h = _months_between(month, last)
    rows = training_rows(x_first, y, h, exclude_2020)
    if len(rows) < MIN_TRAIN:
        return None
    one = np.ones(len(rows))
    x_new, y_last = float(x_now.loc[month]), float(y.loc[last])
    return {
        "month": month, "h": h, "x": x_new, "y_last": y_last, "last_month": last, "n_train": len(rows),
        "arx": _fit_predict(np.column_stack([one, rows["x"], rows["y_lag"]]), rows["y"].to_numpy(),
                            np.array([1.0, x_new, y_last])),
        "ar": _fit_predict(np.column_stack([one, rows["y_lag"]]), rows["y"].to_numpy(), np.array([1.0, y_last])),
        "x_only": _fit_predict(np.column_stack([one, rows["x"]]), rows["y"].to_numpy(), np.array([1.0, x_new])),
        "rw": y_last,
    }


def pooled_forecasts(rows: dict[str, dict[str, Any]], data: dict[str, tuple[pd.Series, pd.Series]],
                     exclude_2020: bool) -> dict[str, float]:
    """`arx` pooled over the countries forecast on one date, with an intercept per country; each
    country's own h decides its lag."""
    out = {}
    for country, row in rows.items():
        frames = []
        for other, (x_first, y) in data.items():
            t = training_rows(x_first, y, row["h"], exclude_2020)
            if len(t):
                frames.append(t.assign(country=other))
        train = pd.concat(frames, ignore_index=True)
        names = sorted(train["country"].unique())
        if country not in names:
            continue
        dummies = (train["country"].to_numpy()[:, None] == np.array(names)[None, :]).astype(float)
        X = np.column_stack([dummies, train["x"], train["y_lag"]])
        new = np.concatenate([(np.array(names) == country).astype(float), [row["x"], row["y_last"]]])
        out[country] = _fit_predict(X, train["y"].to_numpy(), new)
    return out


def run(vintages: store.Vintages, registry: "Registry", spec: NowcastSpec, test_dates: Iterable[Any],
        truth_as_of: Any) -> pd.DataFrame:
    """Walk-forward nowcasts on every test date, scored against the official values known at
    `truth_as_of` (the vault's start).

    Returns one row per (country, target month), forecast on the first test date at which the
    month's signal was complete and its official value not yet out: test_date, country, month, h,
    last_month, x, y_last, n_train, arx, ar, x_only, pooled, rw, y_true (NaN if unknown at
    `truth_as_of`), x_vintage.
    """
    first_seen: dict[str, pd.Series] = {}
    done: set[tuple[str, pd.Timestamp]] = set()
    rows = []
    for test in sorted(pd.Timestamp(d) for d in test_dates):
        snap = pit.snapshot(vintages, test)
        data = snapshot_features(snap, spec, registry)
        realtime: dict[str, tuple[pd.Series, pd.Series]] = {}
        for country, (x, y) in data.items():
            seen = first_seen.get(country, pd.Series(dtype="float64"))
            new = x[~x.index.isin(seen.index)]
            seen = pd.concat([seen, new]).sort_index() if len(new) else seen
            first_seen[country] = seen
            realtime[country] = (seen, y)
        today = {}
        for country, (x, y) in data.items():
            row = forecast_country(first_seen[country], x, y, spec.exclude_2020)
            if row is None or (country, row["month"]) in done:
                continue
            today[country] = row
        pooled = pooled_forecasts(today, realtime, spec.exclude_2020) if today else {}
        for country, row in today.items():
            done.add((country, row["month"]))
            rows.append({"test_date": test, "country": country, **row, "pooled": pooled.get(country, np.nan)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    truth = pit.snapshot(vintages, truth_as_of)
    true_y = {}
    for country in out["country"].unique():
        labels = _series(truth, spec.targets[country], country)
        if labels.empty:
            continue
        y = official_yoy(labels, chains.context_for(registry, spec.targets[country], country))
        for month, value in y.dropna().items():
            true_y[(country, month)] = value
    out["y_true"] = [true_y.get((c, m), np.nan) for c, m in zip(out["country"], out["month"])]
    if spec.exclude_2020:
        out = out[out["month"].dt.year != 2020]
    return out.reset_index(drop=True)


# ------------------------------------------------------------------- scoring


def diebold_mariano(loss_diff: pd.Series, lags: int = DM_LAGS) -> tuple[float, float]:
    """DM statistic and two-sided p-value for a series of loss differentials (model − benchmark),
    with a Newey-West long-run variance. Negative = the model's loss is lower."""
    d = loss_diff.dropna().to_numpy(dtype=float)
    n = len(d)
    if n < 5:
        return float("nan"), float("nan")
    centred = d - d.mean()
    lrv = float(centred @ centred) / n
    for k in range(1, min(lags, n - 1) + 1):
        weight = 1 - k / (lags + 1)
        lrv += 2 * weight * float(centred[k:] @ centred[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / math.sqrt(lrv / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def score(forecasts: pd.DataFrame, model: str, benchmark: str = "ar") -> dict[str, Any]:
    """Out-of-sample metrics of `model` against `benchmark` on the scored forecasts.

    RMSE of each and their ratio (below 1 = the model is better), the directional hit rate
    (forecast and outcome on the same side of the last published value), the DM test on squared
    errors averaged over countries per test date, and the ratio in each half of the test dates.
    """
    f = forecasts.dropna(subset=["y_true", model, benchmark])
    if f.empty:
        return {"n": 0}
    e_m, e_b = f[model] - f["y_true"], f[benchmark] - f["y_true"]
    rmse_m, rmse_b = float(np.sqrt((e_m ** 2).mean())), float(np.sqrt((e_b ** 2).mean()))
    moved = np.sign(f["y_true"] - f["y_last"])

    def hit_rate_of(col: str) -> pd.Series:
        # Only forecasts that call a direction count: `rw` (the last value itself) never does.
        called = np.sign(f[col] - f["y_last"])
        return (called == moved)[(moved != 0) & (called != 0)]

    hits, hits_b = hit_rate_of(model), hit_rate_of(benchmark)
    diff = (e_m ** 2 - e_b ** 2).groupby(f["test_date"]).mean().sort_index()
    dm, p = diebold_mariano(diff)
    dates = sorted(f["test_date"].unique())
    middle = dates[len(dates) // 2] if dates else None
    halves = []
    for part in (f[f["test_date"] < middle], f[f["test_date"] >= middle]):
        if len(part):
            halves.append(float(np.sqrt(((part[model] - part["y_true"]) ** 2).mean())
                                / np.sqrt(((part[benchmark] - part["y_true"]) ** 2).mean())))
        else:
            halves.append(float("nan"))
    return {
        "n": int(len(f)), "countries": int(f["country"].nunique()), "test_dates": len(dates),
        "first_test": f"{min(dates):%Y-%m-%d}", "last_test": f"{max(dates):%Y-%m-%d}",
        "rmse": rmse_m, "rmse_benchmark": rmse_b, "rmse_ratio": rmse_m / rmse_b if rmse_b > 0 else float("nan"),
        "hit_rate": float(hits.mean()) if len(hits) else float("nan"),
        "hit_rate_benchmark": float(hits_b.mean()) if len(hits_b) else float("nan"),
        "dm_stat": dm, "dm_p": p, "ratio_first_half": halves[0], "ratio_second_half": halves[1],
        "beats_benchmark_both_halves": bool(halves[0] < 1 and halves[1] < 1),
    }


def score_by_country(forecasts: pd.DataFrame, model: str, benchmark: str = "ar") -> pd.DataFrame:
    """`score` per country: n, RMSE ratio and hit rate."""
    rows = []
    for country, part in forecasts.groupby("country"):
        s = score(part, model, benchmark)
        if s.get("n", 0):
            rows.append({"country": country, "n": s["n"], "rmse_ratio": s["rmse_ratio"], "hit_rate": s["hit_rate"],
                         "rmse": s["rmse"], "rmse_benchmark": s["rmse_benchmark"]})
    return pd.DataFrame(rows)


def in_sample(snapshot: pd.DataFrame, spec: NowcastSpec, registry: "Registry") -> pd.DataFrame:
    """Full-sample fit of the official YoY on the same month's signal per country (appendix only):
    n, correlation, slope and R²."""
    rows = []
    for country, (x, y) in snapshot_features(snapshot, spec, registry).items():
        pair = pd.concat([x.rename("x"), y.rename("y")], axis=1, sort=True).dropna()
        if spec.exclude_2020:
            pair = pair[pair.index.year != 2020]
        if len(pair) < 6:
            continue
        slope, intercept, r, p, _ = stats.linregress(pair["x"], pair["y"])
        rows.append({"country": country, "n": len(pair), "first": f"{pair.index.min():%Y-%m}",
                     "last": f"{pair.index.max():%Y-%m}", "corr": r, "slope": slope, "r2": r * r, "p": p})
    return pd.DataFrame(rows)
