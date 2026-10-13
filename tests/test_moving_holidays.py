"""Moving-holiday windows (PLAN §5.2)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tracker.registry import load_registry
from tracker.signals.moving_holidays import _merge_days, easter_sunday, holiday_windows


def spans(df: pd.DataFrame, holiday: str) -> list[tuple[str, str]]:
    rows = df[df["holiday"] == holiday]
    return [(s.date().isoformat(), e.date().isoformat()) for s, e in zip(rows["start"], rows["end"])]


def test_easter_sunday_known_dates() -> None:
    known = {2015: date(2015, 4, 5), 2019: date(2019, 4, 21), 2024: date(2024, 3, 31),
             2025: date(2025, 4, 20), 2026: date(2026, 4, 5), 2038: date(2038, 4, 25)}
    assert {y: easter_sunday(y) for y in known} == known


def test_easter_window_is_maundy_thursday_to_easter_monday() -> None:
    w = holiday_windows("DE", ["easter"], 2024, 2025)
    assert spans(w, "easter") == [("2024-03-28", "2024-04-01"), ("2025-04-17", "2025-04-21")]
    # Mexico has no Holy Week public holidays in the package; the window comes from Easter itself.
    assert spans(holiday_windows("MX", ["easter"], 2025, 2025), "easter") == [("2025-04-17", "2025-04-21")]


def test_lunar_new_year_and_golden_week_for_china() -> None:
    w = holiday_windows("CN", ["lunar_new_year", "golden_week"], 2025, 2025)
    lny = spans(w, "lunar_new_year")
    assert len(lny) == 1 and lny[0][0] <= "2025-01-29" <= lny[0][1]
    assert spans(w, "golden_week") == [("2025-10-01", "2025-10-07")]
    assert spans(holiday_windows("JP", ["golden_week"], 2025, 2025), "golden_week") == [("2025-04-29", "2025-05-05")]


def test_ramadan_window_ends_with_eid_al_fitr() -> None:
    w = holiday_windows("TR", ["ramadan_eid"], 2025, 2025)
    # Eid al-Fitr 2025 in Turkey: 30 March - 1 April; Ramadan began 1 March.
    assert spans(w, "ramadan_eid") == [("2025-02-28", "2025-04-01")]


def test_diwali_is_widened_to_five_days() -> None:
    assert spans(holiday_windows("IN", ["diwali"], 2025, 2025), "diwali") == [("2025-10-18", "2025-10-22")]


def test_every_watchlist_country_gets_one_window_per_holiday_and_year() -> None:
    reg = load_registry()
    for c in reg.countries.values():
        w = holiday_windows(c.holiday_calendar, c.moving_holidays, 2015, 2027)
        assert sorted(set(w["holiday"])) == sorted(c.moving_holidays), c.iso3
        assert (w.groupby("holiday").size() == 13).all(), c.iso3
        assert (w["end"] >= w["start"]).all()


def test_no_holidays_gives_an_empty_frame_and_unknown_names_fail() -> None:
    empty = holiday_windows("US", [], 2025, 2025)
    assert empty.empty and list(empty.columns) == ["holiday", "start", "end"]
    with pytest.raises(ValueError, match="unknown moving holiday"):
        holiday_windows("US", ["thanksgiving"], 2025, 2025)
    with pytest.raises(ValueError, match="Golden Week"):
        holiday_windows("US", ["golden_week"], 2025, 2025)


def test_merge_days_bridges_short_gaps_only() -> None:
    days = [date(2025, 1, 28), date(2025, 1, 31), date(2025, 2, 3), date(2025, 2, 10)]
    assert _merge_days(days) == [(date(2025, 1, 28), date(2025, 2, 3)), (date(2025, 2, 10), date(2025, 2, 10))]
