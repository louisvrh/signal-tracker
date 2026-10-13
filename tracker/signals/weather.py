"""Weather-normalised power demand (S18; PLAN §5.4, SIGNALS.md §2).

Daily load is modelled on the country's population-weighted heating and
cooling degree days (HDD18, CDD22), day of week and public holidays, plus a
smooth annual cycle:

    load = a + b_hdd * HDD + b_cdd * CDD + day-of-week + holiday + year-end
           + two annual harmonics (sine and cosine of the day of year) + error

The annual harmonics take the seasonal pattern that isn't weather (daylight,
summer holidays, rooftop solar), so the degree-day coefficients measure how
demand responds to weather that is warmer or colder than usual for the time
of year: exactly what normalisation removes. Without them the coefficients
also carried those seasonal effects, and more than 3 times as much weather
was left in the normalised YoY (tested 2026-10-07 on load alone; NOTES.md).

The model is refitted at the start of every calendar month on the 3 years
before that month (a trailing, rolling window; at least one full year) and
applied to that month's days. Every output on a date therefore uses only data
up to that date: appending data never changes an earlier value, so the `wx`
chain step is one-sided (the append test in tests/test_weather.py decides it).

Outputs per day:

* **expected**: the model's prediction; **anomaly** = actual - expected.
* **weather effect**: b_hdd * (HDD - normal HDD) + b_cdd * (CDD - normal CDD),
  where "normal" is the average degree days for that time of year (the same
  day of year +-15 days) in the fit window.
* **normalised** = actual - weather effect: the demand there would have been
  in normal weather. Its YoY (28-day windows, 364 days apart) is the signal
  SIGNALS.md §2 robust-z scores; in chain terms `wx>yoy>rz(3y)`.

The fit is robust (Huber weights, PLAN D6), so a few extreme days (the 2020
lockdowns, a blackout) don't tilt the weather coefficients.

Everything here is pure: series in, frames out. `adjust` is what the chain
step `wx` calls, with degree days in `context.extra` (PLAN D34).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable

import holidays as holidays_lib
import numpy as np
import pandas as pd

from tracker.signals import transforms as tf

if TYPE_CHECKING:
    from tracker.signals.chain import Context

HDD = "openmeteo_hdd18"
CDD = "openmeteo_cdd22"
WINDOW_YEARS = 3
MIN_TRAIN_DAYS = 365  # a fit needs its data to span at least a full year (every season)
MIN_TRAIN_SHARE = 0.8  # ... and to hold this share of the days that span should have
NORMAL_HALF_WIDTH = 15  # normal degree days: the same day of year +-15 days
HUBER_K = 1.345  # Huber's tuning constant (95% efficiency for normal errors)
IRLS_ITERATIONS = 6
YEAR_END = ((12, 24), (1, 1))  # 24 December to 1 January: the holiday week's lower demand
HARMONICS = 2  # annual sine/cosine pairs for the seasonal pattern that isn't weather

FIT_COLUMNS = ["month", "train_start", "train_end", "n", "r2", "scale", "mean_load", "b_hdd", "b_cdd",
               "b_holiday", "b_year_end"]


@dataclass
class WeatherModel:
    """The model's output for one load series.

    Attributes:
        daily: one row per date with a prediction: actual, hdd, cdd, expected,
            anomaly, anomaly_pct (of expected), weather_effect, normalised.
        fits: one row per monthly refit (FIT_COLUMNS): the training span, its
            size, in-sample R2, residual scale (1.4826 x MAD), the mean load
            and the coefficients.
    """

    daily: pd.DataFrame
    fits: pd.DataFrame


# ------------------------------------------------------------------ calendar


@lru_cache(maxsize=256)
def _holiday_dates(calendar: str, first_year: int, last_year: int) -> frozenset[pd.Timestamp]:
    if not calendar:
        return frozenset()
    found = holidays_lib.country_holidays(calendar, years=range(first_year, last_year + 1))
    return frozenset(pd.Timestamp(d) for d in found)


def public_holidays(dates: pd.DatetimeIndex, calendar: str) -> np.ndarray:
    """True on the country's national public holidays that fall on a weekday
    (a weekend holiday changes little; the day-of-week terms cover weekends)."""
    if not len(dates):
        return np.zeros(0, dtype=bool)
    found = _holiday_dates(calendar, int(dates.year.min()), int(dates.year.max()))
    return np.array([d in found for d in dates]) & (dates.dayofweek < 5)


def year_end(dates: pd.DatetimeIndex) -> np.ndarray:
    """True from 24 December to 1 January, when demand is low beyond the public holidays."""
    (m1, d1), (m2, d2) = YEAR_END
    return ((dates.month == m1) & (dates.day >= d1)) | ((dates.month == m2) & (dates.day <= d2))


def design(dates: pd.DatetimeIndex, hdd: np.ndarray, cdd: np.ndarray, holiday_calendar: str,
           calendar: str = "all", harmonics: int = HARMONICS) -> tuple[np.ndarray, list[str]]:
    """The regression's columns for these dates: constant, HDD, CDD, one dummy per
    day of week but the first (Tuesday-Friday for weekday-only series), holiday,
    year-end, and `harmonics` annual sine/cosine pairs."""
    weekdays = range(5) if calendar == "weekdays" else range(7)
    names = ["const", "hdd", "cdd"] + [f"dow{d}" for d in list(weekdays)[1:]] + ["holiday", "year_end"]
    dow = dates.dayofweek.to_numpy()
    columns = [np.ones(len(dates)), hdd, cdd]
    columns += [(dow == d).astype(float) for d in list(weekdays)[1:]]
    columns += [public_holidays(dates, holiday_calendar).astype(float), year_end(dates).astype(float)]
    angle = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    for k in range(1, harmonics + 1):
        columns += [np.sin(k * angle), np.cos(k * angle)]
        names += [f"sin{k}", f"cos{k}"]
    return np.column_stack(columns), names


# --------------------------------------------------------------------- fitting


_FIT_CACHE: dict[str, tuple[np.ndarray, float, float]] = {}
_FIT_CACHE_SIZE = 20000


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype="float64").tobytes())
        h.update(b"|")
    return h.hexdigest()


def huber_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Robust linear fit by iteratively reweighted least squares with Huber weights.

    Returns (coefficients, in-sample R2, residual scale as 1.4826 x MAD).
    Columns that are all zero (a dummy with no days in the window) get a zero
    coefficient. Results are memoised by the exact data, so refitting an
    unchanged window (a later backtest date) costs nothing.
    """
    key = _digest(X, y)
    found = _FIT_CACHE.get(key)
    if found is not None:
        return found
    used = np.abs(X).sum(axis=0) > 0
    Xu = X[:, used]
    beta_u = np.linalg.lstsq(Xu, y, rcond=None)[0]
    scale = 0.0
    for _ in range(IRLS_ITERATIONS):
        resid = y - Xu @ beta_u
        scale = tf.MAD_SCALE * float(np.median(np.abs(resid - np.median(resid))))
        if scale <= 0:
            break
        u = np.abs(resid) / (HUBER_K * scale)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        sw = np.sqrt(w)
        beta_u = np.linalg.lstsq(Xu * sw[:, None], y * sw, rcond=None)[0]
    beta = np.zeros(X.shape[1])
    beta[used] = beta_u
    resid = y - X @ beta
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum(resid ** 2)) / total if total > 0 else float("nan")
    scale = tf.MAD_SCALE * float(np.median(np.abs(resid - np.median(resid))))
    if len(_FIT_CACHE) >= _FIT_CACHE_SIZE:
        _FIT_CACHE.clear()
    _FIT_CACHE[key] = (beta, r2, scale)
    return beta, r2, scale


def _normal(doy_window: np.ndarray, values_window: np.ndarray, doy_target: np.ndarray) -> np.ndarray:
    """Average of `values_window` over days within NORMAL_HALF_WIDTH days of year of each target day."""
    out = np.full(len(doy_target), np.nan)
    for i, d in enumerate(doy_target):
        dist = np.abs(doy_window - d)
        dist = np.minimum(dist, 366 - dist)
        near = dist <= NORMAL_HALF_WIDTH
        if near.any():
            out[i] = values_window[near].mean()
    return out


def _segment(load: pd.Series, hdd: pd.Series, cdd: pd.Series, holiday_calendar: str, calendar: str,
             window_years: int) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """The model on one stretch of data between known breaks."""
    frame = pd.DataFrame({"actual": load, "hdd": hdd.reindex(load.index), "cdd": cdd.reindex(load.index)})
    frame = frame.dropna()
    if frame.empty:
        return [], []
    dates = pd.DatetimeIndex(frame.index)
    X, names = design(dates, frame["hdd"].to_numpy(), frame["cdd"].to_numpy(), holiday_calendar, calendar)
    y = frame["actual"].to_numpy()
    i_hdd, i_cdd = names.index("hdd"), names.index("cdd")
    weather = pd.DataFrame({"hdd": hdd, "cdd": cdd}).dropna()
    weather = weather[(weather.index >= load.index.min())]  # nothing from before this stretch
    w_dates = pd.DatetimeIndex(weather.index)
    w_doy = w_dates.dayofyear.to_numpy()
    first = dates.min()
    share = 5 / 7 if calendar == "weekdays" else 1.0

    daily_parts, fits = [], []
    months = pd.period_range(dates.min(), dates.max(), freq="M")
    for month in months:
        m_start = month.start_time
        t_start = max(m_start - pd.DateOffset(years=window_years), first)
        span_days = (m_start - t_start).days
        if span_days < MIN_TRAIN_DAYS:
            continue
        train = (dates >= t_start) & (dates < m_start)
        if train.sum() < MIN_TRAIN_SHARE * share * span_days:
            continue
        target = (dates >= m_start) & (dates <= month.end_time)
        if not target.any():
            continue
        beta, r2, scale = huber_fit(X[train], y[train])
        in_window = (w_dates >= t_start) & (w_dates < m_start)
        doy = dates[target].dayofyear.to_numpy()
        hdd_norm = _normal(w_doy[in_window], weather["hdd"].to_numpy()[in_window], doy)
        cdd_norm = _normal(w_doy[in_window], weather["cdd"].to_numpy()[in_window], doy)
        expected = X[target] @ beta
        effect = (beta[i_hdd] * (frame["hdd"].to_numpy()[target] - hdd_norm)
                  + beta[i_cdd] * (frame["cdd"].to_numpy()[target] - cdd_norm))
        actual = y[target]
        daily_parts.append(pd.DataFrame({
            "actual": actual, "hdd": frame["hdd"].to_numpy()[target], "cdd": frame["cdd"].to_numpy()[target],
            "expected": expected, "anomaly": actual - expected,
            "anomaly_pct": (actual / np.where(expected > 0, expected, np.nan) - 1) * 100,
            "weather_effect": effect, "normalised": actual - effect,
        }, index=dates[target]))
        fits.append({
            "month": m_start, "train_start": dates[train].min(), "train_end": dates[train].max(),
            "n": int(train.sum()), "r2": r2, "scale": scale, "mean_load": float(y[train].mean()),
            "b_hdd": float(beta[i_hdd]), "b_cdd": float(beta[i_cdd]),
            "b_holiday": float(beta[names.index("holiday")]), "b_year_end": float(beta[names.index("year_end")]),
        })
    return daily_parts, fits


def normalise(
    load: pd.Series,
    hdd: pd.Series,
    cdd: pd.Series,
    holiday_calendar: str = "",
    calendar: str = "all",
    breaks: Iterable[pd.Timestamp] = (),
    window_years: int = WINDOW_YEARS,
) -> WeatherModel:
    """Fit the demand model month by month on a trailing window and return its daily output.

    Args:
        load: a daily load series (total MWh per day, or the weekday overnight mean in MW).
        hdd, cdd: the country's daily heating (base 18 °C) and cooling (base 22 °C)
            degree days, population-weighted (openmeteo_hdd18, openmeteo_cdd22).
        holiday_calendar: the `holidays` package code of the country (e.g. "DE"); empty for none.
        calendar: "weekdays" for weekday-only series (no weekend dummies).
        breaks: known structural breaks (PLAN D41); no fit window crosses one.
        window_years: the trailing fit window (3 years).

    Days without degree days, and days before a full year of data (load and
    degree days together) is available, get no output.
    """
    load, hdd, cdd = tf.clean(load), tf.clean(hdd), tf.clean(cdd)
    daily_parts: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    breaks = list(breaks)
    ids = tf.segment_ids(load.index, breaks) if breaks else np.zeros(len(load), dtype=int)
    for seg in np.unique(ids) if len(load) else []:
        part = load[ids == seg]
        parts, seg_fits = _segment(part, hdd, cdd, holiday_calendar, calendar, window_years)
        daily_parts += parts
        fits += seg_fits
    columns = ["actual", "hdd", "cdd", "expected", "anomaly", "anomaly_pct", "weather_effect", "normalised"]
    daily = pd.concat(daily_parts).sort_index() if daily_parts else pd.DataFrame(columns=columns, dtype="float64")
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return WeatherModel(daily=daily, fits=pd.DataFrame(fits, columns=FIT_COLUMNS))


def normalised_yoy(model: WeatherModel, calendar: str = "all", breaks: Iterable[pd.Timestamp] = ()) -> pd.Series:
    """YoY (%) of weather-normalised demand: 28-day windows 364 days apart (SIGNALS.md §1)."""
    return tf.yoy_daily(model.daily["normalised"], calendar=calendar, breaks=breaks)


# ---------------------------------------------------------- the chain step


def check(frequency: str, extra: dict[str, Any] | Any) -> list[str]:
    """Why the model can't run on a series (empty if it can): it needs daily load and degree days."""
    problems = []
    if frequency != "D":
        problems.append("weather adjustment needs daily load (the model works day by day); this series isn't daily")
    missing = [name for name in (HDD, CDD) if name not in (extra or {})]
    if missing:
        problems.append(f"weather adjustment needs the country's degree days ({', '.join(missing)}), "
                        "which weren't supplied")
    return problems


def adjust(values: pd.Series, context: "Context") -> pd.Series:
    """The `wx` chain step: weather-normalised demand on the series' dates.

    Degree days come from `context.extra` (openmeteo_hdd18, openmeteo_cdd22),
    as known at the same date in backtests. Raises ValueError (shown as the
    chain's reason) if they're missing, the series isn't daily, or there is
    less than a year of load and degree days together.
    """
    problems = check(context.frequency, context.extra)
    if problems:
        raise ValueError("; ".join(problems))
    model = normalise(values, context.extra[HDD], context.extra[CDD], holiday_calendar=context.holiday_calendar,
                      calendar=context.calendar, breaks=context.breaks)
    if model.daily.empty:
        start = pd.Series(context.extra[HDD]).dropna().index.min()
        since = f"degree days start {pd.Timestamp(start):%Y-%m-%d}" if pd.notna(start) else "no degree days"
        raise ValueError(f"the weather model needs at least a year of load and degree days together ({since})")
    return model.daily["normalised"].reindex(pd.DatetimeIndex(values.index))
