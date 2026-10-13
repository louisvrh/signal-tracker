"""Moving-holiday windows per country (PLAN §5.2).

Pure functions: country settings in, a table of date windows out. The dashboard
shades these windows on charts; S17's transforms can use them to flag daily
windows that overlap a moving holiday.

How each window is built:

* **lunar_new_year**, **diwali**: the country's public-holiday days from the
  `holidays` package whose English name matches the holiday, merged into one
  span when they are at most 3 days apart (bridging weekends). Diwali (one
  public holiday in India) is widened to the five-day festival (±2 days).
* **ramadan_eid**: Ramadan plus the Eid al-Fitr holidays: from 30 days before
  the first Eid al-Fitr day to the last one.
* **easter**: computed from the date of Easter itself (the package has no Holy
  Week days for MEX and only Good Friday for BRA): Maundy Thursday to Easter
  Monday.
* **golden_week**: fixed dates: CHN 1-7 October, JPN 29 April-5 May.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Iterable

import holidays
import pandas as pd

WINDOW_COLUMNS = ["holiday", "start", "end"]

# English holiday names (holidays package, language en_US) per moving holiday.
NAME_PATTERNS = {
    "lunar_new_year": re.compile(r"lunar new year|chinese new year|korean new year", re.IGNORECASE),
    "ramadan_eid": re.compile(r"eid al-fitr", re.IGNORECASE),
    "diwali": re.compile(r"diwali|deepavali", re.IGNORECASE),
}
# Days added before and after the matched public holidays.
PADDING_DAYS = {"lunar_new_year": (0, 0), "ramadan_eid": (30, 0), "diwali": (2, 2)}
MERGE_GAP_DAYS = 3
EASTER_OFFSETS = (-3, 1)  # Maundy Thursday .. Easter Monday
GOLDEN_WEEK = {"CN": ((10, 1), (10, 7)), "JP": ((4, 29), (5, 5))}

LABELS = {
    "lunar_new_year": "Lunar New Year",
    "ramadan_eid": "Ramadan and Eid al-Fitr",
    "easter": "Easter",
    "diwali": "Diwali",
    "golden_week": "Golden Week",
}


def easter_sunday(year: int) -> date:
    """Date of (Western) Easter Sunday: the anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _merge_days(days: Iterable[date], max_gap: int = MERGE_GAP_DAYS) -> list[tuple[date, date]]:
    """Merge sorted days into (first, last) spans; gaps up to `max_gap` days are bridged."""
    spans: list[tuple[date, date]] = []
    for day in sorted(set(days)):
        if spans and (day - spans[-1][1]).days <= max_gap:
            spans[-1] = (spans[-1][0], day)
        else:
            spans.append((day, day))
    return spans


def _named_holiday_spans(holiday: str, holiday_calendar: str, years: list[int]) -> list[tuple[date, date]]:
    calendar = holidays.country_holidays(holiday_calendar, years=years, language="en_US")
    pattern = NAME_PATTERNS[holiday]
    days = [d for d, name in calendar.items() if pattern.search(name)]
    before, after = PADDING_DAYS[holiday]
    return [(first - timedelta(days=before), last + timedelta(days=after)) for first, last in _merge_days(days)]


def holiday_windows(
    holiday_calendar: str,
    moving_holidays: Iterable[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Windows of every moving holiday a country observes, for the given years.

    Args:
        holiday_calendar: the country's `holidays` package code (e.g. "CN").
        moving_holidays: the country's `moving_holidays` from countries.yaml.
        start_year, end_year: inclusive range of years.

    Returns:
        DataFrame with columns holiday, start, end (datetime64, both inclusive),
        sorted by start. Empty if the country has no moving holidays.
    """
    years = list(range(start_year, end_year + 1))
    rows: list[dict[str, object]] = []
    for holiday in moving_holidays:
        if holiday in NAME_PATTERNS:
            spans = _named_holiday_spans(holiday, holiday_calendar, years)
        elif holiday == "easter":
            first, last = EASTER_OFFSETS
            spans = [(easter_sunday(y) + timedelta(days=first), easter_sunday(y) + timedelta(days=last)) for y in years]
        elif holiday == "golden_week":
            if holiday_calendar not in GOLDEN_WEEK:
                raise ValueError(f"no Golden Week dates defined for calendar '{holiday_calendar}'")
            (m1, d1), (m2, d2) = GOLDEN_WEEK[holiday_calendar]
            spans = [(date(y, m1, d1), date(y, m2, d2)) for y in years]
        else:
            raise ValueError(f"unknown moving holiday '{holiday}'")
        rows += [{"holiday": holiday, "start": first, "end": last} for first, last in spans]
    if not rows:
        return pd.DataFrame({"holiday": pd.Series(dtype="str"), "start": pd.Series(dtype="datetime64[us]"),
                             "end": pd.Series(dtype="datetime64[us]")})
    out = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    out["start"] = pd.to_datetime(out["start"]).astype("datetime64[us]")
    out["end"] = pd.to_datetime(out["end"]).astype("datetime64[us]")
    return out.sort_values(["start", "holiday"]).reset_index(drop=True)
