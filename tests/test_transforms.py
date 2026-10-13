"""Signal transforms (tracker/signals/transforms.py) on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracker.signals import transforms as tf
from tracker.signals.moving_holidays import holiday_windows

RNG = np.random.default_rng(17)


def weekly_pattern(dates: pd.DatetimeIndex) -> np.ndarray:
    """Weekend dips: 1.0 on weekdays, 0.6 on Saturday and Sunday."""
    return np.where(dates.dayofweek >= 5, 0.6, 1.0)


def growth(dates: pd.DatetimeIndex, rate: float = 0.05) -> np.ndarray:
    """Smooth exponential growth of `rate` per 364 days, from the first date."""
    days = (dates - dates[0]).days.to_numpy()
    return (1 + rate) ** (days / 364)


# ------------------------------------------------------------ rolling sums


def test_rolling_sums_need_complete_windows() -> None:
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    s = pd.Series(1.0, index=dates)
    assert tf.rolling_sum(s, 7).iloc[6] == 7 and tf.rolling_sum(s, 7).iloc[:6].isna().all()
    assert tf.rolling_sum(s, 28).iloc[27] == 28
    gappy = s.drop(dates[40])
    out = tf.rolling_sum(gappy, 28)
    assert out[dates[39]] == 28 and out[dates[41:68]].isna().all() and out.index[-1] == dates[-1]
    weekdays = s[dates.dayofweek < 5]
    assert tf.rolling_sum(weekdays, 28, calendar="weekdays").dropna().eq(20).all()


# --------------------------------------------------------------------- YoY


def test_daily_yoy_is_weekday_aligned_and_exact() -> None:
    dates = pd.date_range("2023-01-02", periods=728, freq="D")  # two 52-week years
    level = np.where(np.arange(728) >= 364, 1.05, 1.0)
    s = pd.Series(100 * weekly_pattern(dates) * level, index=dates)
    out = tf.yoy_daily(s)
    assert out[dates < dates[364]].isna().all()
    assert np.allclose(out[dates >= dates[364 + 27]], 5.0)
    # A few missing days: the window is compared by its average, close to the truth
    # (a missing weekday shifts the weekly mix slightly).
    holes = s.drop(dates[[500, 501, 610]])
    out = tf.yoy_daily(holes)
    assert (out[out.index >= dates[391]] - 5.0).abs().max() < 1.5
    # Too many missing days in a window: no value.
    sparse = s.drop(dates[450:465])
    assert tf.yoy_daily(sparse)[dates[470]] != tf.yoy_daily(sparse)[dates[470]]  # NaN


def test_daily_yoy_on_weekday_series_and_differences() -> None:
    dates = pd.bdate_range("2023-01-02", "2024-12-27")  # two 52-week years of weekdays
    second_year = dates >= dates[0] + pd.Timedelta(days=364)
    s = pd.Series(np.where(second_year, 55.0, 50.0), index=dates)
    settled = dates >= dates[0] + pd.Timedelta(days=364 + 27)
    out = tf.yoy_daily(s, calendar="weekdays")
    assert out[~second_year].isna().all() and np.allclose(out[settled], 10.0)
    diff = tf.yoy_daily(s - 52.0, method="diff", calendar="weekdays")
    assert np.allclose(diff[settled], 5.0)


def test_monthly_yoy_and_zero_base() -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="MS")
    s = pd.Series([100.0] * 12 + [110.0] * 12 + [0.0], index=dates)
    out = tf.yoy(s, "M")
    assert out.iloc[:12].isna().all() and np.allclose(out.iloc[12:24], 10.0)
    base = pd.Series([0.0] + [5.0] * 11 + [7.0], index=dates[:13])
    assert np.isnan(tf.yoy(base, "M").iloc[12])
    assert np.allclose(tf.yoy(s, "M", method="diff").iloc[12:24], 10.0)
    quarterly = pd.Series([4.0, 4, 4, 4, 5, 5, 5, 5], index=pd.date_range("2024-01-01", periods=8, freq="QS"))
    assert np.allclose(tf.yoy(quarterly, "Q").iloc[4:], 25.0)


# ------------------------------------------------ moving holidays: the fix


def lunar_new_year_months(start_year: int, end_year: int) -> dict[int, int]:
    """Month (1 or 2) in which China's Lunar New Year holiday starts, per year."""
    w = holiday_windows("CN", ["lunar_new_year"], start_year, end_year)
    return {s.year: s.month for s in w["start"]}


def test_lunar_new_year_shift_misleads_naive_monthly_yoy_but_not_the_combined_one() -> None:
    """Fake Chinese monthly exports: steady 5% growth, but the Lunar New Year month
    loses 30% of its activity. The holiday falls in January in 2023 and 2025 and
    in February in 2024 and 2026, so a naive month-on-same-month YoY swings wildly."""
    dates = pd.date_range("2022-01-01", "2026-12-01", freq="MS")
    lny = lunar_new_year_months(2022, 2026)
    assert [lny[y] for y in (2023, 2024, 2025, 2026)] == [1, 2, 1, 2]  # the shift the test relies on
    base = 100 * 1.05 ** (dates.year - 2022)
    dip = np.array([0.7 if d.month == lny[d.year] else 1.0 for d in dates])
    s = pd.Series(base * dip, index=dates)

    naive = tf.yoy(s, "M")
    fixed = tf.yoy(s, "M", lunar_new_year=True)
    jan_feb = dates[(dates.month <= 2) & (dates.year >= 2023)]
    # The naive YoY misreads every January and February (e.g. +50% or -26.5% instead of +5%).
    assert (naive[jan_feb] - 5).abs().min() > 20
    # Combined, February carries the true growth; January is left empty.
    feb = jan_feb[jan_feb.month == 2]
    assert np.allclose(fixed[feb], 5.0)
    assert fixed[jan_feb[jan_feb.month == 1]].isna().all()
    # Other months are untouched.
    other = dates[(dates.month > 2) & (dates.year >= 2023)]
    assert np.allclose(fixed[other], 5.0) and np.allclose(naive[other], 5.0)


def test_combine_jan_feb_levels() -> None:
    dates = pd.date_range("2025-01-01", "2026-01-01", freq="MS")
    s = pd.Series(np.arange(13.0) + 10, index=dates)
    out = tf.combine_jan_feb(s)
    assert out[pd.Timestamp("2025-01-01")] == out[pd.Timestamp("2025-02-01")] == 10.5
    assert out[pd.Timestamp("2026-01-01")] == 22.0  # February 2026 not there yet
    assert out.iloc[2:12].equals(s.iloc[2:12])


def test_lunar_new_year_shift_in_daily_data_is_flagged_and_fixed() -> None:
    """Fake Chinese daily port volumes: weekly pattern, 5% growth a year, and half the
    usual activity on Lunar New Year holiday days. The naive 28-day YoY is badly wrong
    around the holiday; the flags cover every distorted day, and interpolating across
    the holiday before taking the YoY recovers the true growth."""
    dates = pd.date_range("2022-01-03", "2026-06-30", freq="D")
    windows = holiday_windows("CN", ["lunar_new_year"], 2022, 2026)
    holiday = tf.holiday_days(windows, dates[0], dates[-1]).to_numpy()
    true_level = 1000 * weekly_pattern(dates) * growth(dates)
    s = pd.Series(true_level * np.where(holiday, 0.5, 1.0), index=dates)

    naive = tf.yoy_daily(s)
    flags = tf.holiday_flags(dates, windows)
    checked = dates >= dates[0] + pd.Timedelta(days=364 + 27)
    error = (naive - 5.0).abs()[checked]
    assert error.max() > 5  # a holiday in one window only: several points off
    assert error[~flags[checked]].max() < 0.01  # every distorted day is flagged
    assert flags[checked].mean() < 0.3  # ...and the flags don't cover everything

    fixed = tf.yoy_daily(tf.interpolate_holidays(s, windows))
    assert (fixed - 5.0).abs()[checked].max() < 0.1


def test_holiday_flags_cover_the_window_and_its_comparison_a_year_later() -> None:
    windows = pd.DataFrame({"holiday": ["x"], "start": [pd.Timestamp("2025-03-10")], "end": [pd.Timestamp("2025-03-12")]})
    dates = pd.date_range("2025-01-01", "2026-12-31", freq="D")
    flags = tf.holiday_flags(dates, windows)
    flagged = dates[flags.to_numpy()]
    # Windows ending 10 March .. 6 April 2025 contain the holiday; 364 days later, their comparisons do.
    first = pd.date_range("2025-03-10", "2025-04-08", freq="D")
    second = first + pd.Timedelta(days=364)
    assert flagged.equals(first.append(second))


def test_interpolate_holidays_keeps_the_weekly_pattern_and_holds_an_open_window() -> None:
    dates = pd.date_range("2025-01-06", periods=70, freq="D")
    s = pd.Series(100 * weekly_pattern(dates), index=dates)
    hit = pd.date_range("2025-02-03", "2025-02-09", freq="D")
    dipped = s.copy()
    dipped[hit] *= 0.3
    windows = pd.DataFrame({"holiday": ["x", "y"], "start": [hit[0], dates[-3]], "end": [hit[-1], dates[-1] + pd.Timedelta(days=5)]})
    dipped[dates[-3:]] *= 0.3
    out = tf.interpolate_holidays(dipped, windows)
    assert np.allclose(out, s)  # constant weekly pattern: interpolation restores it exactly


# --------------------------------------------------------------------- STL


def test_stl_removes_a_monthly_seasonal_pattern() -> None:
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    trend = np.linspace(100, 160, 120)
    season = 10 * np.sin(2 * np.pi * dates.month / 12)
    s = pd.Series(trend + season + RNG.normal(0, 0.5, 120), index=dates)
    parts = tf.stl_components(s, "M")
    assert list(parts.columns) == ["observed", "trend", "seasonal", "resid"]
    assert np.corrcoef(parts["seasonal"], season)[0, 1] > 0.98
    adjusted = tf.seasonally_adjust(s, "M")
    assert np.abs(adjusted - trend).max() < 3 < np.abs(s - trend).max()
    with pytest.raises(ValueError, match="at least 3 years"):
        tf.stl_components(s.iloc[:30], "M")


def test_mstl_removes_weekly_and_annual_cycles_from_daily_data() -> None:
    dates = pd.date_range("2019-01-01", periods=4 * 365, freq="D")
    t = np.arange(len(dates))
    season = 8 * np.sin(2 * np.pi * t / 7) + 20 * np.sin(2 * np.pi * t / 365)
    s = pd.Series(500 + 0.02 * t + season + RNG.normal(0, 1, len(t)), index=dates)
    adjusted = tf.seasonally_adjust(s, "D")
    assert np.std(adjusted - (500 + 0.02 * t)) < 3 < np.std(season)
    weekdays = s[dates.dayofweek < 5]
    assert tf.seasonal_periods("D", "weekdays") == (5, 261)
    assert len(tf.seasonally_adjust(weekdays, "D", calendar="weekdays")) == len(weekdays)


# ---------------------------------------------------------------- robust z


def test_robust_z_matches_the_formula_and_ignores_outliers() -> None:
    dates = pd.date_range("2015-01-01", periods=60, freq="MS")
    values = RNG.normal(100, 5, 60)
    values[30] = 400.0  # a 2020-style extreme month
    s = pd.Series(values, index=dates)
    z = tf.robust_z(s, "M")
    assert z.iloc[:23].isna().all() and z.iloc[23:].notna().all()  # 24 months of history needed
    window = values[59 - 35:60]
    med = np.median(window)
    mad = np.median(np.abs(window - med))
    assert z.iloc[59] == pytest.approx((values[59] - med) / (1.4826 * mad))
    # The outlier barely moves later scores (a mean/std z-score would shrink them all).
    std_z = (s - s.rolling(36, min_periods=24).mean()) / s.rolling(36, min_periods=24).std()
    assert z.iloc[31:59].abs().max() < 3.5 and z.iloc[30] > 30 and std_z.iloc[30] < 6


def test_robust_z_daily_uses_calendar_windows_and_zero_mad_is_missing() -> None:
    dates = pd.date_range("2020-01-01", periods=1200, freq="D")
    s = pd.Series(RNG.normal(0, 1, 1200), index=dates)
    z = tf.robust_z(s, "D")
    first = z.first_valid_index()
    assert first == pd.Timestamp("2022-01-01")  # 24 months after the first value
    window = s[(dates > dates[-1] - pd.Timedelta(days=1095)) & (dates <= dates[-1])].to_numpy()
    med = np.median(window)
    assert z.iloc[-1] == pytest.approx((s.iloc[-1] - med) / (1.4826 * np.median(np.abs(window - med))))
    flat = pd.Series(1.0, index=pd.date_range("2015-01-01", periods=40, freq="MS"))
    assert tf.robust_z(flat, "M").isna().all()


# ------------------------------------------------------------ known structural breaks

BREAK = pd.Timestamp("2016-01-01")


def stepped(dates: pd.DatetimeIndex, step: float = 1.17) -> pd.Series:
    """5% a year growth with a reporting change that lifts the level 17% from BREAK on."""
    return pd.Series(growth(dates) * np.where(dates >= BREAK, step, 1.0), index=dates)


def test_daily_yoy_never_compares_across_a_break() -> None:
    dates = pd.date_range("2014-01-01", "2018-12-31", freq="D")
    s = stepped(dates)
    naive, fixed = tf.yoy_daily(s), tf.yoy_daily(s, breaks=[BREAK])
    affected = (dates >= BREAK) & (dates - pd.Timedelta(days=364 + 27) < BREAK)
    assert (naive[affected] - 5).abs().max() > 10  # the naive YoY reads the step as growth
    assert fixed[affected].isna().all()
    assert np.allclose(fixed[~affected].dropna(), naive[~affected].dropna())
    assert np.allclose(fixed[dates >= "2017-02-01"].dropna(), 5.0, atol=0.05)


def test_monthly_and_lunar_new_year_yoy_never_compare_across_a_break() -> None:
    dates = pd.date_range("2013-01-01", "2018-12-01", freq="MS")
    s = stepped(dates)
    out = tf.yoy_periodic(s, "M", breaks=[BREAK])
    assert out["2016-01-01":"2016-12-01"].isna().all()
    assert out["2015-01-01":"2015-12-01"].notna().all() and out["2017-01-01":].notna().all()
    mid = pd.Timestamp("2016-02-15")  # a break inside February: the February-to-February span also crosses it
    mid_out = tf.yoy_periodic(s, "M", breaks=[mid])
    assert mid_out[["2016-02-01", "2017-02-01"]].isna().all() and mid_out[["2016-01-01", "2017-03-01"]].notna().all()
    lny = tf.yoy_periodic(s, "M", combine_lunar_new_year=True, breaks=[pd.Timestamp("2016-02-01")])
    assert np.isnan(lny["2017-02-01"])  # Jan-Feb 2017 against Jan-Feb 2016, whose January precedes the break
    assert not np.isnan(lny["2018-02-01"])


def test_robust_z_scores_each_stretch_between_breaks_on_its_own() -> None:
    dates = pd.date_range("2010-01-01", "2021-12-01", freq="MS")
    rng = np.random.default_rng(3)
    s = pd.Series(100 + rng.normal(0, 1, len(dates)), index=dates) * np.where(dates >= BREAK, 1.17, 1.0)
    naive, fixed = tf.robust_z(s, "M"), tf.robust_z(s, "M", breaks=[BREAK])
    assert naive["2016-01-01":"2016-12-01"].min() > 5  # every post-break month looks extreme
    assert fixed["2016-01-01":"2017-11-01"].isna().all() and fixed.notna()["2017-12-01"]  # 24 months again
    assert fixed["2018-01-01":].abs().max() < 4
    pd.testing.assert_series_equal(fixed[:"2015-12-01"], naive[:"2015-12-01"])
