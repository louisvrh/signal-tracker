"""Signal transforms (PLAN §5.2-5.3, SIGNALS.md §1).

Pure functions: a series indexed by date (the period start) in, a series or
frame out. No file or network access. Every function drops missing values
from its input first and returns a result on the input's own dates (unless
it says otherwise), so callers can line results up with the original data.

* **Rolling sums** of daily series over 7 or 28 days.
* **YoY**: daily series compare a 28-day window with the same window 364 days
  earlier (weekdays stay aligned); monthly and quarterly series compare with
  the same period a year earlier. For Lunar New Year countries, January and
  February are compared combined.
* **Moving holidays**: flags for daily windows that overlap a holiday window
  (or whose comparison window does), a per-weekday interpolation across the
  holiday days of daily series, and the combined January-February level for
  monthly series.
* **STL** seasonal adjustment (MSTL for daily series, which have a weekly and
  an annual cycle), for series with at least 3 years of history.
* **Robust z-scores**: (x - rolling median) / (1.4826 x rolling MAD) over 3
  years, with at least 24 months of history.
* **Known structural breaks** (a definition or reporting change, PLAN D41):
  YoY gives no value where the two compared windows aren't on the same side
  of every break, and robust z-scores are computed within each stretch
  between breaks, so neither ever compares values across one.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL

Method = Literal["pct", "diff"]
Calendar = Literal["all", "weekdays"]

DAILY_WINDOW_DAYS = 28
DAILY_LAG_DAYS = 364  # 52 weeks: the comparison window starts on the same weekday
MIN_WINDOW_SHARE = 0.75  # a 28-day window needs this share of its days to be compared
PERIODS_PER_YEAR = {"W": 52, "M": 12, "Q": 4}
MIN_STL_YEARS = 3
MAD_SCALE = 1.4826  # makes the MAD a consistent estimate of the standard deviation for normal data
ROBUST_Z_YEARS = 3
ROBUST_Z_MIN_MONTHS = 24


# ------------------------------------------------------------------ helpers


def clean(values: pd.Series) -> pd.Series:
    """The series as float64, sorted by date, with missing values dropped."""
    s = pd.Series(values, dtype="float64").dropna()
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def expected_days(days: int, calendar: Calendar = "all") -> int:
    """How many values a complete `days`-day window holds (weekdays only: 5 of every 7)."""
    return days if calendar == "all" else days * 5 // 7


def change(current: pd.Series, prior: pd.Series, method: Method) -> pd.Series:
    """Percent change (x100; NaN from a zero or negative base) or plain difference."""
    if method == "diff":
        return current - prior
    return (current / prior.where(prior > 0) - 1) * 100


def segment_ids(dates: pd.DatetimeIndex, breaks: Iterable[pd.Timestamp]) -> np.ndarray:
    """Which stretch between known breaks each date is in: 0 before the first break,
    1 from the first break (inclusive) to the next, and so on."""
    cuts = np.sort(pd.DatetimeIndex(list(breaks)).to_numpy().astype("datetime64[us]"))
    return np.searchsorted(cuts, pd.DatetimeIndex(dates).to_numpy().astype("datetime64[us]"), side="right")


def comparable(first: pd.DatetimeIndex, last: pd.DatetimeIndex, breaks: Iterable[pd.Timestamp] = ()) -> np.ndarray:
    """True where the data from `first` to `last` (inclusive dates) lies between the same two breaks,
    i.e. a comparison drawing on that span doesn't cross a known structural break."""
    breaks = list(breaks)
    if not breaks:
        return np.ones(len(first), dtype=bool)
    return segment_ids(first, breaks) == segment_ids(last, breaks)


def _daily_calendar(s: pd.Series, before_days: int = 0) -> pd.DatetimeIndex:
    return pd.date_range(s.index.min() - pd.Timedelta(days=before_days), s.index.max(), freq="D")


# ------------------------------------------------------------ rolling sums


def rolling_sum(values: pd.Series, days: int = DAILY_WINDOW_DAYS, calendar: Calendar = "all") -> pd.Series:
    """Trailing `days`-day sum of a daily series (e.g. 7 or 28 days).

    The window ending on each date must be complete (every day, or every
    weekday for `calendar="weekdays"`); otherwise the sum is NaN, because a
    sum over missing days would look like a fall.
    """
    s = clean(values)
    return s.rolling(f"{days}D", min_periods=expected_days(days, calendar)).sum()


# --------------------------------------------------------------------- YoY


def yoy_daily(
    values: pd.Series,
    window_days: int = DAILY_WINDOW_DAYS,
    lag_days: int = DAILY_LAG_DAYS,
    method: Method = "pct",
    calendar: Calendar = "all",
    min_share: float = MIN_WINDOW_SHARE,
    breaks: Iterable[pd.Timestamp] = (),
) -> pd.Series:
    """YoY of a daily series: each `window_days` window against the same window `lag_days` earlier.

    With complete windows this is the ratio of the two 28-day sums. A window
    with a few days missing (exchange holidays, a failed download) is scaled
    to a full window, i.e. compared by its average, as long as it holds at
    least `min_share` of its expected days; otherwise the result is NaN.
    364 days is exactly 52 weeks, so both windows have the same weekdays.
    Dates whose two windows don't lie between the same known `breaks` get NaN.
    """
    s = clean(values)
    if s.empty:
        return s
    grid = s.reindex(_daily_calendar(s)).rolling(f"{window_days}D", min_periods=1)
    need = math.ceil(min_share * expected_days(window_days, calendar))
    mean = grid.mean().where(grid.count() >= need)
    prior = pd.Series(mean.reindex(mean.index - pd.Timedelta(days=lag_days)).to_numpy(), index=mean.index)
    out = change(mean, prior, method).reindex(s.index)
    first = s.index - pd.Timedelta(days=lag_days + window_days - 1)
    return out.where(comparable(first, s.index, breaks))


def combine_jan_feb(values: pd.Series) -> pd.Series:
    """Monthly levels with January and February both replaced by their average.

    Used for Lunar New Year countries, whose holiday moves activity between the
    two months. A January whose February isn't there yet (the latest data) is
    left as it is.
    """
    s = clean(values)
    out = s.copy()
    for year in sorted(set(s.index.year)):
        jan, feb = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 2, 1)
        if jan in s.index and feb in s.index:
            out[jan] = out[feb] = (s[jan] + s[feb]) / 2
    return out


def yoy_periodic(
    values: pd.Series,
    frequency: Literal["W", "M", "Q"],
    method: Method = "pct",
    combine_lunar_new_year: bool = False,
    breaks: Iterable[pd.Timestamp] = (),
) -> pd.Series:
    """YoY of a weekly, monthly or quarterly series: each period against the same period a year earlier.

    Weekly: 364 days earlier. Monthly and quarterly: 12 months earlier.
    With `combine_lunar_new_year` (monthly only), January and February are
    compared combined: February carries the YoY of the January-February
    average, and January is NaN (it can't be known before February is).
    Periods whose comparison spans a known break (from the start of the
    earlier period to the end of the current one) get NaN.
    """
    s = clean(values)
    if frequency == "W":
        prior_dates = s.index - pd.Timedelta(days=DAILY_LAG_DAYS)
        last = s.index + pd.Timedelta(days=6)
    else:
        prior_dates = s.index - pd.DateOffset(months=12)
        last = s.index + pd.DateOffset(months=1 if frequency == "M" else 3) - pd.Timedelta(days=1)
    if combine_lunar_new_year:
        if frequency != "M":
            raise ValueError("January-February combining only applies to monthly series")
        combined = combine_jan_feb(s)
        # A February whose January is missing isn't a combined value: leave it out.
        has_jan = pd.Series([(pd.Timestamp(d.year, 1, 1) in s.index) for d in s.index], index=s.index)
        combined = combined.where(~((s.index.month == 2) & ~has_jan.to_numpy()))
        prior = pd.Series(combined.reindex(prior_dates).to_numpy(), index=s.index)
        out = change(combined, prior, method)
        # February's value also uses the January before it, a year earlier.
        first = pd.DatetimeIndex(np.where(s.index.month == 2, prior_dates - pd.DateOffset(months=1), prior_dates))
        return out.where(s.index.month != 1).where(comparable(first, last, breaks))
    prior = pd.Series(s.reindex(prior_dates).to_numpy(), index=s.index)
    return change(s, prior, method).where(comparable(prior_dates, last, breaks))


def yoy(
    values: pd.Series,
    frequency: Literal["D", "W", "M", "Q"],
    method: Method = "pct",
    lunar_new_year: bool = False,
    calendar: Calendar = "all",
    breaks: Iterable[pd.Timestamp] = (),
) -> pd.Series:
    """YoY as SIGNALS.md §1 defines it for the series' frequency.

    Daily: `yoy_daily` (28-day window, 364 days earlier). Weekly, monthly,
    quarterly: `yoy_periodic`; monthly series of Lunar New Year countries
    (`lunar_new_year=True`) compare January and February combined. No value
    compares data across a known structural break (`breaks`).
    """
    if frequency == "D":
        return yoy_daily(values, method=method, calendar=calendar, breaks=breaks)
    return yoy_periodic(values, frequency, method=method,
                        combine_lunar_new_year=lunar_new_year and frequency == "M", breaks=breaks)


# ---------------------------------------------------------- moving holidays


def holiday_days(windows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily boolean calendar from `start` to `end`: True on days inside a holiday window.

    `windows` has columns start and end (inclusive dates), as returned by
    `moving_holidays.holiday_windows`.
    """
    days = pd.date_range(start, end, freq="D")
    inside = np.zeros(len(days), dtype=bool)
    for first, last in zip(windows["start"], windows["end"]):
        inside |= (days >= first) & (days <= last)
    return pd.Series(inside, index=days)


def holiday_flags(
    dates: pd.DatetimeIndex,
    windows: pd.DataFrame,
    window_days: int = DAILY_WINDOW_DAYS,
    lag_days: int = DAILY_LAG_DAYS,
) -> pd.Series:
    """Flag daily YoY windows distorted by a moving holiday.

    A date is flagged when its trailing `window_days` window, or the
    comparison window `lag_days` earlier, overlaps a holiday window: the
    holiday then falls in one year's window but not necessarily the other's.
    Flagged windows are shaded on charts and get half weight in composites
    (SIGNALS.md §1).

    Returns:
        Boolean series indexed by `dates`.
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return pd.Series(dtype=bool, index=dates)
    start = dates.min() - pd.Timedelta(days=lag_days + window_days)
    days = holiday_days(windows, start, dates.max()).astype(float)
    in_window = days.rolling(window_days, min_periods=1).max().astype(bool)
    current = in_window.reindex(dates).to_numpy()
    prior = in_window.reindex(dates - pd.Timedelta(days=lag_days), fill_value=False).to_numpy()
    return pd.Series(current | prior, index=dates)


def interpolate_holidays(values: pd.Series, windows: pd.DataFrame) -> pd.Series:
    """Daily levels with the days inside moving-holiday windows replaced.

    Each holiday day gets a straight line between the nearest ordinary days
    *with the same weekday* before and after the window, so the weekly
    pattern survives. A window still open at the end of the data (no ordinary
    day after it yet) is held at the last ordinary value of that weekday; one
    at the start takes the first. The result uses data after each window, so
    its values inside the latest window revise as data arrives.
    """
    s = clean(values)
    if s.empty or windows.empty:
        return s
    holiday = holiday_days(windows, s.index.min(), s.index.max()).reindex(s.index).to_numpy()
    masked = s.where(~holiday)
    out = masked.groupby(s.index.dayofweek).transform(
        lambda g: g.interpolate(method="time", limit_area="inside").ffill().bfill()
    )
    return out.fillna(s)  # a weekday with no ordinary day at all keeps its values


# --------------------------------------------------------------------- STL


def seasonal_periods(frequency: str, calendar: Calendar = "all") -> tuple[int, ...]:
    """The seasonal cycles STL removes, in periods: daily weekly + annual; else annual."""
    if frequency == "D":
        return (7, 365) if calendar == "all" else (5, 261)
    return (PERIODS_PER_YEAR[frequency],)


def regular(values: pd.Series, frequency: str, calendar: Calendar = "all") -> pd.Series:
    """The series on a regular grid (every day, every weekday, week, month or quarter),
    with internal gaps filled by linear interpolation in time."""
    s = clean(values)
    if s.empty:
        return s
    if frequency == "D":
        freq = "D" if calendar == "all" else "B"
    else:
        freq = {"W": "7D", "M": "MS", "Q": "QS"}[frequency]
    grid = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    return s.reindex(grid.union(s.index)).interpolate(method="time").reindex(grid)


def stl_components(
    values: pd.Series,
    frequency: str,
    calendar: Calendar = "all",
    robust: bool = True,
    min_years: int = MIN_STL_YEARS,
) -> pd.DataFrame:
    """STL decomposition (MSTL for daily series: weekly and annual cycles).

    Needs at least `min_years` years of data. Monthly series use a robust
    fit (outliers such as 2020 don't bend the seasonal pattern); daily MSTL
    uses the statsmodels defaults, because a robust fit of a 365-day cycle is
    several times slower.

    Returns:
        Frame indexed by the input's dates with columns observed, trend,
        seasonal (all seasonal cycles added up) and resid.
    """
    s = clean(values)
    periods = seasonal_periods(frequency, calendar)
    grid = regular(s, frequency, calendar)
    need = min_years * periods[-1]
    if len(grid) < need:
        raise ValueError(f"STL needs at least {min_years} years of data ({need} periods); the series has {len(grid)}")
    if len(periods) == 1:
        fit = STL(grid.to_numpy(), period=periods[0], robust=robust).fit()
        seasonal = np.asarray(fit.seasonal)
    else:
        fit = MSTL(grid.to_numpy(), periods=periods).fit()
        seasonal = np.asarray(fit.seasonal).sum(axis=1)
    out = pd.DataFrame({"trend": np.asarray(fit.trend), "seasonal": seasonal, "resid": np.asarray(fit.resid)},
                       index=grid.index).reindex(s.index)
    out.insert(0, "observed", s)
    return out


def seasonally_adjust(values: pd.Series, frequency: str, calendar: Calendar = "all") -> pd.Series:
    """The series minus its STL seasonal component (see `stl_components`)."""
    parts = stl_components(values, frequency, calendar)
    return parts["observed"] - parts["seasonal"]


# ---------------------------------------------------------------- robust z


def rolling_mad(s: pd.Series, window: int | str, min_periods: int) -> pd.Series:
    """Rolling median absolute deviation from each window's own median."""
    return s.rolling(window, min_periods=min_periods).apply(
        lambda w: np.median(np.abs(w - np.median(w))), raw=True
    )


def robust_z(
    values: pd.Series,
    frequency: str,
    window_years: float = ROBUST_Z_YEARS,
    min_history_months: int = ROBUST_Z_MIN_MONTHS,
    breaks: Iterable[pd.Timestamp] = (),
) -> pd.Series:
    """Robust z-score: (x - rolling median) / (1.4826 x rolling MAD), PLAN §5.3.

    The trailing window covers `window_years` (3) years and includes the
    current value; a score needs `min_history_months` (24) months of history
    since the series' first value. Daily windows are in calendar days
    (365 per year), so series with weekday-only data work too. A window whose
    MAD is zero gives NaN. Charts clip scores at ±5; this function doesn't.
    With known `breaks`, each stretch between breaks is scored on its own
    (the history requirement starts again after a break).
    """
    s = clean(values)
    if s.empty:
        return s
    breaks = list(breaks)
    if breaks:
        ids = segment_ids(s.index, breaks)
        parts = [robust_z(s[ids == i], frequency, window_years, min_history_months) for i in np.unique(ids)]
        return pd.concat(parts).reindex(s.index)
    if frequency == "D":
        window: int | str = f"{round(365 * window_years)}D"
        min_periods = 1
        enough = s.index >= s.index.min() + pd.DateOffset(months=min_history_months)
    else:
        per_year = PERIODS_PER_YEAR[frequency]
        window = round(per_year * window_years)
        min_periods = max(1, round(per_year * min_history_months / 12))
        enough = np.ones(len(s), dtype=bool)
    median = s.rolling(window, min_periods=min_periods).median()
    mad = rolling_mad(s, window, min_periods)
    z = (s - median) / (MAD_SCALE * mad.where(mad > 0))
    return z.where(enough)
