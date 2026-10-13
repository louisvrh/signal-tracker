"""Weather-normalised power demand (S18; tracker/signals/weather.py, the `wx` chain step)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tracker import store
from tracker.backtest import pit
from tracker.registry import Indicator, Registry, load_registry
from tracker.signals import chain as ch
from tracker.signals import transforms as tf
from tracker.signals import weather as wx

DATES = pd.date_range("2019-01-01", "2024-12-31", freq="D")
B_HDD, B_CDD = 20.0, 30.0
STEP = pd.Timestamp("2024-01-01")  # demand rises by STEP_SIZE from here: the "true" change
STEP_SIZE = 50.0


def climate(dates: pd.DatetimeIndex, seed: int = 1, cold: tuple[str, str] | None = None) -> tuple[pd.Series, pd.Series]:
    """Daily HDD18 and CDD22 from a seasonal temperature with day-to-day noise (and an optional cold spell)."""
    rng = np.random.default_rng(seed)
    doy = dates.dayofyear.to_numpy()
    temp = 14 - 10 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 3.5, len(dates))
    if cold:
        temp = temp - 5.0 * ((dates >= cold[0]) & (dates <= cold[1]))
    return (pd.Series(np.maximum(0, 18 - temp), index=dates), pd.Series(np.maximum(0, temp - 22), index=dates))


def demand(dates: pd.DatetimeIndex, hdd: pd.Series, cdd: pd.Series, seed: int = 2, holidays: str = "DE") -> pd.Series:
    """Load with known weather coefficients, weekly pattern, holidays, a non-weather annual cycle and a step."""
    rng = np.random.default_rng(seed)
    dow = dates.dayofweek.to_numpy()
    load = (1000 + B_HDD * hdd.to_numpy() + B_CDD * cdd.to_numpy()
            - 150 * (dow == 5) - 250 * (dow == 6)
            - 200 * wx.public_holidays(dates, holidays)
            - 40 * np.cos(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)  # daylight, summer holidays
            + STEP_SIZE * (dates >= STEP) + rng.normal(0, 10, len(dates)))
    return pd.Series(load, index=dates)


def context(hdd: pd.Series, cdd: pd.Series, calendar: str = "all", **kw) -> ch.Context:
    return ch.Context(frequency="D", unit="MWh", indicator="entsoe_load", country="DEU", calendar=calendar,
                      signal_group="power_demand", holiday_calendar="DE", extra={wx.HDD: hdd, wx.CDD: cdd}, **kw)


# ------------------------------------------------------------- the model


def test_the_model_recovers_the_weather_coefficients() -> None:
    hdd, cdd = climate(DATES)
    model = wx.normalise(demand(DATES, hdd, cdd), hdd, cdd, holiday_calendar="DE")
    full = model.fits[model.fits["month"] >= "2022-01-01"]  # full 3-year windows
    assert full["b_hdd"].between(B_HDD - 1, B_HDD + 1).all()
    assert full["b_cdd"].between(B_CDD - 3, B_CDD + 3).all()
    assert (full["r2"] > 0.95).all()
    assert model.fits["month"].min() == pd.Timestamp("2020-01-01")  # a full year of data first
    assert model.daily.index.min() == pd.Timestamp("2020-01-01")


def test_normalised_yoy_sees_through_a_cold_winter() -> None:
    """A winter 5 °C colder than usual raises raw YoY a lot; the normalised YoY stays near the
    true change in demand: the YoY of the same demand in normal weather (the step of 50)."""
    hdd, cdd = climate(DATES, cold=("2024-01-01", "2024-02-29"))
    load = demand(DATES, hdd, cdd)
    model = wx.normalise(load, hdd, cdd, holiday_calendar="DE")
    norm = wx.normalised_yoy(model)
    raw = tf.yoy_daily(load)
    draws = [climate(DATES, seed=s) for s in range(100, 160)]  # normal weather: the average over many years
    normal_hdd = sum(h for h, _ in draws) / len(draws)
    normal_cdd = sum(c for _, c in draws) / len(draws)
    truth = tf.yoy_daily(demand(DATES, normal_hdd, normal_cdd))
    day = pd.Timestamp("2024-02-28")
    assert abs(norm[day] - truth[day]) < 1.0
    assert raw[day] - truth[day] > 5.0
    # Away from the cold spell nothing much changes.
    assert abs(norm[pd.Timestamp("2024-07-15")] - truth[pd.Timestamp("2024-07-15")]) < 1.0


def test_the_fit_is_robust_to_a_block_of_extreme_days() -> None:
    """A lockdown-like month at 85% of normal demand barely moves the weather coefficients."""
    hdd, cdd = climate(DATES)
    load = demand(DATES, hdd, cdd)
    hit = load.copy()
    spell = (DATES >= "2020-03-20") & (DATES <= "2020-05-05")
    hit[spell] *= 0.85
    a = wx.normalise(load, hdd, cdd, holiday_calendar="DE").fits.set_index("month")
    b = wx.normalise(hit, hdd, cdd, holiday_calendar="DE").fits.set_index("month")
    later = a.index >= "2021-01-01"
    assert np.allclose(a.loc[later, "b_hdd"], b.loc[later, "b_hdd"], rtol=0.05)


def test_no_fit_window_crosses_a_known_break() -> None:
    hdd, cdd = climate(DATES)
    load = demand(DATES, hdd, cdd)
    brk = pd.Timestamp("2021-06-01")
    load[load.index >= brk] *= 1.17  # a reporting change
    model = wx.normalise(load, hdd, cdd, holiday_calendar="DE", breaks=[brk])
    dates = model.daily.index
    assert not ((dates >= brk) & (dates < "2022-06-01")).any()  # a full year after the break first
    assert dates[dates >= brk].min() == pd.Timestamp("2022-06-01")
    assert (model.fits.loc[model.fits["month"] >= brk, "train_start"] >= brk).all()


def test_weekday_series_have_no_weekend_terms() -> None:
    hdd, cdd = climate(DATES)
    load = demand(DATES, hdd, cdd)
    weekdays = load[load.index.dayofweek < 5]
    X, names = wx.design(weekdays.index, hdd.reindex(weekdays.index).to_numpy(), cdd.reindex(weekdays.index).to_numpy(),
                         "DE", calendar="weekdays")
    assert [n for n in names if n.startswith("dow")] == ["dow1", "dow2", "dow3", "dow4"]
    model = wx.normalise(weekdays, hdd, cdd, holiday_calendar="DE", calendar="weekdays")
    assert (model.daily.index.dayofweek < 5).all()
    assert model.fits.loc[model.fits["month"] >= "2022-01-01", "b_hdd"].between(B_HDD - 1.5, B_HDD + 1.5).all()


def test_holidays_are_weekday_public_holidays() -> None:
    days = pd.DatetimeIndex(["2024-12-25", "2024-12-28", "2024-10-03", "2024-10-04", "2026-10-03"])
    assert list(wx.public_holidays(days, "DE")) == [True, False, True, False, False]  # 2026-10-03 is a Saturday
    assert list(wx.year_end(pd.DatetimeIndex(["2024-12-23", "2024-12-24", "2025-01-01", "2025-01-02"]))) == \
        [False, True, True, False]


# ------------------------------------------------------- the chain step


def test_wx_is_one_sided_by_the_append_test() -> None:
    """Appending data (load and degree days) never changes an earlier normalised value,
    so the step is labelled one-sided, as the append test decides for every step (D32)."""
    hdd, cdd = climate(DATES)
    load = demand(DATES, hdd, cdd)
    unchanged = True
    for cut in (pd.Timestamp("2022-03-17"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-06-01")):
        def upto(s: pd.Series) -> pd.Series:
            return s[s.index <= cut]
        short = ch.apply("wx", upto(load), context(upto(hdd), upto(cdd))).values
        full = ch.apply("wx", load, context(hdd, cdd)).values
        before = full[full.index <= cut]
        unchanged &= before.index.equals(short.index) and np.allclose(before, short, rtol=1e-9, atol=1e-9)
    assert ch.STEPS["wx"].sided == (ch.ONE_SIDED if unchanged else ch.TWO_SIDED)
    assert ch.STEPS["wx"].sided == ch.ONE_SIDED


def test_wx_in_a_chain_gives_normalised_demand_and_its_yoy() -> None:
    hdd, cdd = climate(DATES)
    load = demand(DATES, hdd, cdd)
    result = ch.apply("wx>yoy", load, context(hdd, cdd))
    model = wx.normalise(load, hdd, cdd, holiday_calendar="DE")
    expected = wx.normalised_yoy(model).dropna()
    pd.testing.assert_series_equal(result.values, expected, check_names=False, check_freq=False)
    step = ch.apply("wx", load, context(hdd, cdd))
    assert dict(step.removed)["wx"].abs().max() > 50  # the weather effect it removed
    assert step.provisional_from is None


@pytest.mark.parametrize("chain, ctx_kw, message", [
    ("wx", {"frequency": "M"}, "needs daily load"),
    ("wx", {"signal_group": None}, "only applies to power-demand indicators"),
    ("yoy>wx", {}, "needs a positive series"),
])
def test_wx_rejects_what_it_cant_model(chain: str, ctx_kw: dict, message: str) -> None:
    hdd, cdd = climate(DATES)
    base = context(hdd, cdd)
    ctx = ch.Context(**{**{f: getattr(base, f) for f in ("frequency", "unit", "indicator", "country", "calendar",
                                                          "signal_group", "holiday_calendar", "extra")}, **ctx_kw})
    with pytest.raises(ch.ChainError, match=message):
        ch.validate(ch.parse(chain), ctx)


def test_wx_explains_missing_degree_days_and_too_little_data() -> None:
    hdd, cdd = climate(DATES)
    load = demand(DATES, hdd, cdd)
    no_weather = ch.Context(frequency="D", unit="MWh", indicator="entsoe_load", signal_group="power_demand")
    with pytest.raises(ch.ChainError, match="needs the country's degree days"):
        ch.apply("wx", load, no_weather)
    short = load[:200]
    with pytest.raises(ch.ChainError, match="at least a year of load and degree days"):
        ch.apply("wx", short, context(hdd[:200], cdd[:200]))
    assert ch.weather_unavailable() is None


def test_context_carries_the_holiday_calendar() -> None:
    reg = load_registry()
    assert ch.context_for(reg, "entsoe_load", "DEU").holiday_calendar == "DE"
    assert ch.context_for(reg, "neso_load", "GBR").holiday_calendar == "GB"


# ---------------------------------------------------- backtest engine (D39)


def _rows(dates, ind: str, values, lag_days: int, vintage: str = "estimated") -> pd.DataFrame:
    available = (pd.DatetimeIndex(dates) + pd.Timedelta(days=lag_days)).tz_localize("UTC")
    return pd.DataFrame({"date": pd.DatetimeIndex(dates), "country": "DEU", "indicator": ind,
                         "value": np.asarray(values, float), "unit": "x", "frequency": "D", "source": "test",
                         "fetched_at": available, "available_at": available, "vintage_type": vintage})


def _engine_registry() -> Registry:
    real = load_registry()
    load = Indicator(id="test_load", source="test", name="load", unit="MWh", frequency="D", category="energy",
                     role="alt", higher_is_stronger=True, default_transform="yoy_28d", publication_lag_days=1,
                     refetch_window_days=30, countries=["DEU"], signal_group="power_demand")
    return Registry(indicators={"test_load": load}, countries={"DEU": real.countries["DEU"]})


def test_the_engine_feeds_wx_degree_days_as_known_and_nothing_later_leaks() -> None:
    """Through pit.compute, `wx` uses degree days from the same snapshot. Poisoning everything
    published after the test date (huge values, poisoned revisions a second later, poisoned
    future days) changes no result for that date."""
    hdd, cdd = climate(DATES)
    load = demand(DATES, hdd, cdd)
    clean = pd.concat([_rows(DATES, "test_load", load, 1), _rows(DATES, wx.HDD, hdd, 1),
                       _rows(DATES, wx.CDD, cdd, 1)], ignore_index=True)
    cut = pd.Timestamp("2023-06-15", tz="UTC")
    rng = np.random.default_rng(7)
    poisoned = clean.copy()
    future = poisoned["available_at"] > cut
    poisoned.loc[future, "value"] = rng.uniform(1e6, 1e9, int(future.sum()))
    revisions = poisoned[~future].copy()
    revisions["available_at"] = revisions["fetched_at"] = cut + pd.Timedelta(seconds=1)
    revisions["value"] = 1e9
    poisoned = pd.concat([poisoned, revisions], ignore_index=True)

    registry = _engine_registry()
    signals = [pit.ChainSignal("wx", "wx", ("test_load",)), pit.ChainSignal("wx_yoy", "wx>yoy", ("test_load",))]
    dates = [cut - pd.Timedelta(days=28), cut]
    a = pit.compute(store.Vintages(clean), registry, dates, series=signals)
    b = pit.compute(store.Vintages(poisoned), registry, dates, series=signals)
    pd.testing.assert_frame_equal(a, b)
    assert a["value"].notna().all() and (a["problem"] == "").all()
    assert not a["provisional"].any()  # one-sided: its end value is final
    # The end value equals the model run on the data as known then.
    known = DATES[DATES + pd.Timedelta(days=1) <= cut.tz_localize(None)]
    model = wx.normalise(load[known], hdd[known], cdd[known], holiday_calendar="DE")
    end = a[(a["signal"] == "wx") & (a["as_of"] == cut)].iloc[0]
    assert end["value"] == pytest.approx(model.daily["normalised"].iloc[-1])


# ------------------------------------------------------------ the report


def test_weather_command_outputs_and_report(tmp_path) -> None:
    """`python -m tracker weather`'s pieces: every daily power-demand series with data gets a model,
    a summary row and a chart page; series without data are skipped."""
    from tracker import diagnostics

    hdd, cdd = climate(DATES)
    load = demand(DATES, hdd, cdd)

    def rows(ind: str, values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"date": DATES, "country": "DEU", "indicator": ind, "value": values.to_numpy()})

    obs = pd.concat([rows("entsoe_load", load), rows(wx.HDD, hdd), rows(wx.CDD, cdd)], ignore_index=True)
    reg = load_registry()
    assert ("entsoe_load", "DEU") in diagnostics.daily_load_series(reg)
    assert all(c != "ZAF" for _, c in diagnostics.daily_load_series(reg))  # excluded from power signals (D22)
    daily, fits, summary = diagnostics.run_models(obs, reg)
    assert list(summary["indicator"]) == ["entsoe_load"] and summary.loc[0, "verdict"] == "good"
    assert summary.loc[0, "weather_in_normalised_yoy"] < 0.05
    assert {"anomaly", "normalised", "yoy_raw", "yoy_normalised"} <= set(daily.columns)
    paths = diagnostics.write_report(daily, fits, summary, reg, tmp_path)
    assert [p.name for p in paths] == ["weather_fit.md", "weather_DEU.html"]
    assert "| entsoe_load | DEU |" in paths[0].read_text(encoding="utf-8")
