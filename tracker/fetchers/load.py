"""Local-day electricity load figures from interval data (EIA and ENTSO-E).

Load is reported as average power (MW) over intervals of 15, 30 or 60
minutes, timestamped in UTC. `daily_load` turns such a series into:

* daily total load: MWh over each complete local calendar day;
* weekday overnight load: mean MW over 00:00-05:00 local time, Monday to
  Friday, when that window is fully covered;
* incomplete days: days with missing intervals, which are *not* stored, so a
  gap is never averaged over silently.

Each value is weighted by the time it covers. The source's resolution isn't
passed in (it changes by country and year), so a value's length is taken as
the gap to its nearer neighbour, at most one hour: 15 minutes inside a
15-minute series, 60 inside an hourly one, and a gap next to a value doesn't
stretch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OVERNIGHT_HOURS = range(0, 5)  # local hour starts 00..04, i.e. 00:00-05:00
MAX_STEP = pd.Timedelta(hours=1)


def interval_hours(starts: pd.DatetimeIndex, max_step: pd.Timedelta = MAX_STEP) -> np.ndarray:
    """Hours covered by each value: the gap to its nearer neighbour, at most `max_step`.

    `starts` must be sorted and free of duplicates.
    """
    cap = max_step / pd.Timedelta(hours=1)
    if len(starts) == 0:
        return np.array([], dtype=float)
    gaps = np.asarray((starts[1:] - starts[:-1]) / pd.Timedelta(hours=1), dtype=float)  # hours between neighbours
    prev = np.concatenate([[np.inf], gaps])
    nxt = np.concatenate([gaps, [np.inf]])
    return np.minimum(np.minimum(prev, nxt), cap)


def daily_load(
    starts_utc: pd.DatetimeIndex, values: np.ndarray | pd.Series, tz: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate interval load (MW, interval starts in UTC) to local days.

    Returns (daily_total, weekday_overnight, incomplete):
    * daily_total: date, value = MWh, complete local days only;
    * weekday_overnight: date, value = mean MW over local 00:00-05:00 on
      Monday-Friday, when those five hours are fully covered;
    * incomplete: date, hours (covered), expected (23, 24 or 25) for days
      with missing intervals, including days without any value between the
      first and last day with data.
    Missing values (NaN) count as gaps. Duplicate timestamps keep the last value.
    """
    empty = pd.DataFrame(columns=["date", "value"])
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(starts_utc).tz_convert("UTC")).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if s.empty:
        return empty, empty.copy(), pd.DataFrame(columns=["date", "hours", "expected"])

    hours = interval_hours(s.index)
    local = s.index.tz_convert(tz)
    df = pd.DataFrame({
        "day": local.tz_localize(None).normalize(),
        "hour": local.hour,
        "hours": hours,
        "mwh": s.to_numpy() * hours,
    })

    # Every day from the first to the last with data, so a day without any value is a gap too.
    days = pd.date_range(df["day"].min(), df["day"].max(), freq="D")
    midnights = days.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    next_midnights = (days + pd.Timedelta(days=1)).tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    expected = pd.Series((next_midnights - midnights) / pd.Timedelta(hours=1), index=days)

    per_day = df.groupby("day").agg(hours=("hours", "sum"), mwh=("mwh", "sum")).reindex(days, fill_value=0.0)
    full = np.isclose(per_day["hours"].to_numpy(), expected.reindex(per_day.index).to_numpy())
    daily = pd.DataFrame({"date": per_day.index[full], "value": per_day["mwh"].to_numpy()[full]})
    gaps = per_day[~full]
    incomplete = pd.DataFrame({
        "date": gaps.index,
        "hours": gaps["hours"].round(2).to_numpy(),
        "expected": expected.reindex(gaps.index).to_numpy(),
    })

    night = df[df["hour"].isin(OVERNIGHT_HOURS) & (df["day"].dt.dayofweek < 5)]
    per_night = night.groupby("day").agg(hours=("hours", "sum"), mwh=("mwh", "sum"))
    covered = per_night[np.isclose(per_night["hours"].to_numpy(), len(OVERNIGHT_HOURS))]
    overnight = pd.DataFrame({"date": covered.index, "value": (covered["mwh"] / len(OVERNIGHT_HOURS)).to_numpy()})
    return daily, overnight, incomplete
