"""Transform chains (tracker/signals/chain.py): labels, known answers, strings, rejections."""

from __future__ import annotations

import re
import sys
import types

import numpy as np
import pandas as pd
import pytest

from tracker.registry import load_registry
from tracker.signals import chain as ch
from tracker.signals import transforms as tf
from tracker.signals.moving_holidays import holiday_windows

RNG = np.random.default_rng(32)
CN_WINDOWS = holiday_windows("CN", ["lunar_new_year"], 2010, 2027)
MONTHLY_CTX = ch.Context(frequency="M", unit="USD", country="CHN", lunar_new_year=True)
DAILY_CTX = ch.Context(frequency="D", unit="metric tons", country="CHN", lunar_new_year=True,
                       holiday_windows=CN_WINDOWS)


def random_walk(dates: pd.DatetimeIndex, seasonal_period: float, scale: float = 1.0) -> pd.Series:
    """A positive random walk with drift plus a seasonal wave."""
    t = np.arange(len(dates))
    walk = np.cumsum(RNG.normal(0.1, 1.0, len(dates)))
    return pd.Series(scale * (200 + walk + 10 * np.sin(2 * np.pi * t / seasonal_period)), index=dates)


MONTHLY = random_walk(pd.date_range("2010-01-01", "2026-06-01", freq="MS"), 12)
DAILY = random_walk(pd.date_range("2019-01-01", "2025-06-30", freq="D"), 365.25)
MONTHLY_CUT = pd.Timestamp("2025-01-01")  # a January: the Lunar New Year combination is still open
DAILY_CUT = pd.Timestamp("2025-01-29")  # inside China's 2025 Lunar New Year window

# Chain strings for the append test: (monthly, daily). Daily band edges are shorter,
# because the daily test series covers 6.5 years.
APPEND_CASES = {
    "sma": ("sma(3m)", "sma(28d)"), "csma": ("csma(5m)", "csma(29d)"), "ema": ("ema(6m)", "ema(28d)"),
    "yoy": ("yoy", "yoy"), "stl": ("stl", "stl"), "hol": ("hol", "hol"),
    "d1": ("d1", "d1"), "dlog": ("dlog", "dlog"), "lin": ("lin", "lin"), "hp": ("hp", "hp"),
    "ham": ("ham", "ham(1y)"), "bk": ("bk", "bk(60d,1y,k=120d)"), "cf": ("cf", "cf(60d,1y)"),
    "bwz": ("bwz", "bwz(60d,1y)"), "bwc": ("bwc", "bwc(60d,1y)"),
    "level": ("level", "level"), "log": ("log", "log"), "idx": ("idx(2012-01-01)", "idx(2019-06-03)"),
    "rz": ("rz", "rz"), "pr": ("pr", "pr(2y)"), "prf": ("prf", "prf"),
}


def unchanged_after_append(chain: str, series: pd.Series, cut: pd.Timestamp, ctx: ch.Context) -> bool:
    """True if appending the data after `cut` leaves every earlier output value as it was."""
    short = ch.apply(chain, series[series.index <= cut], ctx).values
    full = ch.apply(chain, series, ctx).values
    before = full[full.index <= cut]
    return before.index.equals(short.index) and np.allclose(before, short, rtol=1e-9, atol=1e-9)


def test_every_step_has_an_append_case() -> None:
    # Weather adjustment needs daily load with degree days; its append test is in test_weather.py.
    assert set(APPEND_CASES) == set(ch.STEPS) - {"wx"}


@pytest.mark.parametrize("key", sorted(APPEND_CASES))
def test_append_test_decides_the_label(key: str) -> None:
    """One-sided means appending data never changes an earlier value (monthly and daily);
    a step that changes one in either case is two-sided."""
    monthly, daily = APPEND_CASES[key]
    unchanged = (unchanged_after_append(monthly, MONTHLY, MONTHLY_CUT, MONTHLY_CTX)
                 and unchanged_after_append(daily, DAILY, DAILY_CUT, DAILY_CTX))
    assert ch.STEPS[key].sided == (ch.ONE_SIDED if unchanged else ch.TWO_SIDED)


def test_registry_covers_the_required_transforms() -> None:
    required = {"Smoothing": {"sma", "csma", "ema"}, "Seasonality": {"yoy", "stl", "hol", "wx"},
                "Detrending": {"d1", "dlog", "lin", "hp", "ham"}, "Band-pass": {"bk", "cf", "bwz", "bwc"},
                "Scaling": {"level", "log", "idx", "rz", "pr", "prf"}}
    for group, keys in required.items():
        assert {k for k, s in ch.STEPS.items() if s.group == group} == keys
    for spec in ch.STEPS.values():
        assert spec.help and spec.name and spec.sided in (ch.ONE_SIDED, ch.TWO_SIDED)
        assert (spec.revises == "none") == (spec.sided == ch.ONE_SIDED)


# ------------------------------------------------------------ known answers

MONTHS = pd.date_range("1996-01-01", periods=360, freq="MS")  # 30 years
DAYS = pd.date_range("2024-01-01", periods=200, freq="D")


def run(chain: str, s: pd.Series, ctx: ch.Context = MONTHLY_CTX) -> ch.ChainResult:
    return ch.apply(chain, s, ctx)


def test_moving_averages_on_a_ramp() -> None:
    ramp = pd.Series(np.arange(200.0), index=DAYS)
    daily = ch.Context(frequency="D", unit="ships")
    trailing = run("sma(28d)", ramp, daily)
    assert np.allclose(trailing.values[DAYS[27]:], ramp[DAYS[27]:] - 13.5)
    centred = run("csma(29d)", ramp, daily)
    assert np.allclose(centred.values[DAYS[14]:DAYS[-15]], ramp[DAYS[14]:DAYS[-15]])
    assert centred.provisional == 14
    ema = run("ema(28d)", ramp, daily)
    assert ramp.iloc[-1] - ema.values.iloc[-1] == pytest.approx(13.5, abs=0.05)  # steady-state lag (n-1)/2
    monthly = pd.Series(np.arange(360.0), index=MONTHS)
    assert monthly.iloc[-1] - run("ema(5m)", monthly).values.iloc[-1] == pytest.approx(2.0)
    removed = dict(trailing.removed)["sma(28d)"]
    assert np.allclose(removed[DAYS[27]:], 13.5)  # the noise component: raw minus smooth


def test_yoy_step_uses_the_s17_rules() -> None:
    s = pd.Series(100 * 1.1 ** (np.arange(36) // 12), index=MONTHS[:36])
    usa = ch.Context(frequency="M", unit="USD", country="USA")
    assert np.allclose(run("yoy", s, usa).values, 10.0)
    chn = run("yoy", s, MONTHLY_CTX).values
    assert (chn.index.month != 1).all()  # January left empty for Lunar New Year countries
    temps = ch.Context(frequency="M", unit="degC")
    assert np.allclose(run("yoy", s - 150, temps).values, run("yoy(diff)", s, usa).values)


def test_stl_and_holiday_steps_return_what_they_removed() -> None:
    season = 10 * np.sin(2 * np.pi * MONTHS.month / 12)
    s = pd.Series(np.linspace(100, 300, 360) + season, index=MONTHS)
    result = run("stl", s)
    assert result.keeps_units and result.provisional == 12
    assert np.corrcoef(dict(result.removed)["stl"], season)[0, 1] > 0.99

    jan_feb = pd.Series([10.0, 20.0, 30.0] * 2, index=pd.to_datetime(
        ["2024-01-01", "2024-02-01", "2024-03-01", "2025-01-01", "2025-02-01", "2025-03-01"]))
    assert run("hol", jan_feb).values.tolist() == [15.0, 15.0, 30.0, 15.0, 15.0, 30.0]
    usa = ch.Context(frequency="M", unit="USD", country="USA")
    assert run("hol", jan_feb, usa).values.equals(jan_feb)
    assert run("hol", jan_feb.iloc[:4]).provisional == 1  # January 2025 waits for February


def test_daily_holiday_step_and_its_provisional_window() -> None:
    dates = pd.date_range("2024-11-01", "2025-01-29", freq="D")
    s = pd.Series(100.0, index=dates)
    s[s.index >= "2025-01-28"] = 40.0  # the 2025 holiday has started
    result = run("hol", s, DAILY_CTX)
    assert np.allclose(result.values, 100.0)
    assert result.provisional == 2  # 28 and 29 January: the window is still open
    assert np.allclose(dict(result.removed)["hol"]["2025-01-28":], -60.0)


def test_differences() -> None:
    ramp = pd.Series(3.0 * np.arange(24), index=MONTHS[:24])
    assert np.allclose(run("d1", ramp).values, 3.0)
    grow = pd.Series(100 * 1.01 ** np.arange(24), index=MONTHS[:24])
    assert np.allclose(run("dlog", grow).values, 100 * np.log(1.01))


@pytest.mark.filterwarnings("ignore:The design matrix is rank-deficient")  # a perfect line: collinear lags
def test_detrending_a_straight_line_leaves_nothing() -> None:
    # Daily, so a line in observations is also a line in calendar time (which `lin` uses).
    days = pd.date_range("2020-01-01", periods=1100, freq="D")
    line = pd.Series(50 + 2.0 * np.arange(1100), index=days)
    for chain in ("lin", "hp", "ham"):
        result = run(chain, line, ch.Context(frequency="D", unit="ships"))
        assert np.abs(result.values).max() < 1e-6 * line.max(), chain
        trend = dict(result.removed)[chain].reindex(result.values.index)
        assert np.allclose(trend, line[result.values.index], rtol=0, atol=1e-6 * line.max()), chain


def test_linear_detrend_recovers_a_cycle() -> None:
    wave = 5 * np.sin(2 * np.pi * np.arange(360) / 40)
    s = pd.Series(100 + 0.5 * np.arange(360) + wave, index=MONTHS)
    assert np.corrcoef(run("lin", s).values, wave)[0, 1] > 0.99


def band_parts() -> tuple[np.ndarray, np.ndarray, pd.Series]:
    t = np.arange(360)
    inside = 5 * np.sin(2 * np.pi * t / 36)  # a 3-year cycle: inside 1.5-8 years
    outside = 5 * np.sin(2 * np.pi * t / 4)  # a 4-month cycle: outside
    return inside, outside, pd.Series(100 + inside + outside, index=MONTHS)


@pytest.mark.parametrize("key", ["bk", "cf", "bwz"])
def test_band_pass_keeps_the_band_and_removes_a_sine_outside_it(key: str) -> None:
    inside, outside, s = band_parts()
    out = run(key, s).values
    middle = out.index[(out.index >= MONTHS[120]) & (out.index < MONTHS[240])]
    kept = pd.Series(inside, index=MONTHS)[middle]
    assert np.corrcoef(out[middle], kept)[0, 1] > 0.98
    assert np.std(out[middle] - kept) < 0.1 * np.std(outside)


def test_causal_butterworth_passes_the_band_and_blocks_outside_it() -> None:
    inside, outside, _ = band_parts()
    later = MONTHS[120:]
    kept = run("bwc", pd.Series(100 + inside, index=MONTHS)).values[later]
    blocked = run("bwc", pd.Series(100 + outside, index=MONTHS)).values[later]
    assert np.std(kept) > 0.6 * np.std(inside) and np.std(blocked) < 0.05 * np.std(outside)


def test_scaling_steps() -> None:
    s = pd.Series(np.arange(1.0, 61.0), index=MONTHS[:60])
    assert run("level", s).values.equals(s)
    assert np.allclose(run("log", s).values, np.log(s))
    idx = run("idx(1996-03-15)", s).values  # the first value on or after the date: April
    assert idx[MONTHS[3]] == 100.0 and idx.iloc[0] == 25.0
    assert np.allclose(run("rz", s).values, tf.robust_z(s, "M").dropna())
    assert np.allclose(run("pr(3y)", s).values, 100.0)  # a rising series is always at its top
    assert np.allclose(run("prf", s).values, s / 60 * 100)


# ----------------------------------------------------------------- strings


@pytest.mark.parametrize("text, canonical", [
    ("stl>sma(28d)>rz(3y)", "stl>sma(28d)>rz(3y)"),
    ("  hol > yoy ", "hol>yoy"),
    ("level", "level"),
    ("bk(k=2y)", "bk(auto,auto,2y)"),
    ("bk(1.50y, 8y, auto)", "bk(1.5y,8y)"),
    ("hp(129600)", "hp(129600)"),
    ("idx(2020-01-01)>log", "idx(2020-01-01)>log"),
    ("yoy(diff)>ema(6m)", "yoy(diff)>ema(6m)"),
    ("cf(2m,18m)", "cf(2m,18m)"),
    ("ham(2y,12)>pr(5y)>prf", "ham(2y,12)>pr(5y)>prf"),
    ("bwz(order=4)>bwc(1y,4y,3)", "bwz(auto,auto,4)>bwc(1y,4y,3)"),
    ("", ""),
])
def test_chains_round_trip_through_strings(text: str, canonical: str) -> None:
    chain = ch.parse(text)
    assert str(chain) == canonical
    assert ch.parse(str(chain)) == chain


def test_every_step_round_trips_with_parameters() -> None:
    samples = {"length": ch.Length(2.5, "y"), "number": 1234.5, "int": 3, "choice": "diff",
               "date": pd.Timestamp("2021-06-01")}
    for key, spec in ch.STEPS.items():
        step = ch.Step(key, tuple(samples[p.kind] for p in spec.params))
        assert ch.parse_step(str(step)) == step, key
        assert "|" not in str(step)


@pytest.mark.parametrize("text, message", [
    ("stl|sma(28d)", "don't use '|'"),
    ("smooth(28d)", "unknown step 'smooth'"),
    ("sma(28x)", "is not a length"),
    ("sma(0d)", "is not a length"),
    ("sma(28d,3)", "at most 1 parameter"),
    ("bk(q=3)", "has no parameter 'q'"),
    ("bk(low=1y,8y)", "positional parameters must come before"),
    ("hp(-5)", "must be positive"),
    ("idx(yesterday)", "must be a date"),
    ("yoy(percent)", "must be one of pct, diff"),
    ("sma(28d", "can't read step"),
])
def test_bad_strings_are_rejected_clearly(text: str, message: str) -> None:
    with pytest.raises(ch.ChainError, match=message.replace("(", r"\(").replace("|", r"\|")):
        ch.parse(text)


# -------------------------------------------------------------- rejections

REG = load_registry()


def ctx_of(indicator: str, country: str = "DEU") -> ch.Context:
    return ch.context_for(REG, indicator, country)


@pytest.mark.parametrize("chain, indicator, message", [
    ("yoy>log", "ember_demand", "natural log needs a positive series.*after `yoy`"),
    ("log", "openmeteo_temp_mean", "its unit is degC"),
    ("dlog", "openmeteo_hdd18", "can be zero or negative"),
    ("d1>idx(2020-01-01)", "portwatch_exports", "index = 100 at a date needs a positive series"),
    ("sma(28d)", "ember_demand", "a length in days .* can't be used on monthly data; use m, q, y"),
    ("stl>rz(2w)", "comtrade_exports", "a length in weeks"),
    ("wx", "portwatch_exports", "only applies to power-demand indicators"),
    ("wx", "ember_demand", "needs daily load .* is monthly"),
    ("yoy>wx", "entsoe_load", "weather adjustment needs a positive series.*after `yoy`"),
    ("hp>yoy(pct)", "ember_demand", "needs a positive series"),
    ("cf(1m,8y)", "ember_demand", "under two periods of monthly data"),
    ("bwz(8y,1.5y)", "eurostat_ip", "must be shorter than the longest"),
])
def test_incompatible_combinations_are_rejected(chain: str, indicator: str, message: str) -> None:
    problems = ch.check(ch.parse(chain), ctx_of(indicator))
    assert any(re.search(message, p) for p in problems), problems
    with pytest.raises(ch.ChainError):
        ch.validate(ch.parse(chain), ctx_of(indicator))


def test_compatible_chains_pass_the_checks() -> None:
    for chain, indicator in [("stl>sma(28d)>rz(3y)", "portwatch_exports"), ("hp>yoy", "ember_demand"),
                             ("cf(2m,18m)", "portwatch_exports"), ("yoy>rz", "openmeteo_temp_mean"),
                             ("sma(2m)>idx(2020-01-01)", "prices_country_etf")]:
        assert ch.check(ch.parse(chain), ctx_of(indicator)) == [], chain


def test_too_little_data_is_a_clear_runtime_error() -> None:
    short = pd.Series(np.arange(1.0, 25.0), index=MONTHS[:24])
    with pytest.raises(ch.ChainError, match="STL needs at least 3 years"):
        run("stl", short)
    with pytest.raises(ch.ChainError, match="one full longest cycle"):
        run("cf", short)


# ---------------------------------------------------------------- weather


def test_weather_step_calls_the_weather_model_and_says_when_it_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """`wx` runs tracker.signals.weather.adjust (S18; its own tests are in test_weather.py) and keeps
    the generic refusal for a module without `adjust`."""
    load = ctx_of("entsoe_load")
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    assert ch.weather_unavailable() is None
    weather = sys.modules[ch.WEATHER_MODULE]
    monkeypatch.setattr(weather, "adjust", lambda values, context: values * 0.5)
    result = ch.apply("wx", pd.Series(2.0, index=days), load)
    assert np.allclose(result.values, 1.0) and np.allclose(dict(result.removed)["wx"], 1.0)
    monkeypatch.setitem(sys.modules, ch.WEATHER_MODULE, types.ModuleType(ch.WEATHER_MODULE))
    assert "doesn't exist yet" in ch.weather_unavailable()
    with pytest.raises(ch.ChainError, match="doesn't exist yet"):
        ch.apply("wx", pd.Series(2.0, index=days), load)


# ------------------------------------------------------- chain-level results


def test_provisional_points_add_up_through_two_sided_steps() -> None:
    s = MONTHLY.iloc[:120]
    assert run("csma(3m)>sma(3m)", s).provisional == 1
    assert run("csma(5m)>stl", s).provisional == 2 + 12
    assert run("sma(3m)>rz", s).provisional == 0
    both = run("csma(5m)>stl>rz", s)
    assert both.two_sided and both.provisional_from == s.index[-14]


def test_real_time_only_drops_two_sided_steps_and_flags_describe_the_result() -> None:
    chain = ch.parse("stl>sma(28d)>rz(3y)")
    kept, dropped = chain.one_sided_only()
    assert str(kept) == "sma(28d)>rz(3y)" and [str(s) for s in dropped] == ["stl"]
    assert not kept.two_sided and chain.two_sided
    result = run("stl>sma(3m)", MONTHLY)
    assert result.keeps_units and [name for name, _ in result.removed] == ["stl", "sma(3m)"]
    assert not run("yoy", MONTHLY).keeps_units
    assert run("lin", MONTHLY).revises_history and not run("hp", MONTHLY).revises_history


def test_precomputed_stl_is_used_only_for_the_same_series() -> None:
    parts = tf.stl_components(MONTHLY, "M")
    fake = parts.assign(seasonal=1.0)
    used = ch.apply("stl", MONTHLY, ch.Context(frequency="M", unit="USD", extra={"stl": fake}))
    assert np.allclose(used.values, MONTHLY - 1.0)
    other = ch.apply("stl", MONTHLY * 2, ch.Context(frequency="M", unit="USD", extra={"stl": fake}))
    assert not np.allclose(other.values, MONTHLY * 2 - 1.0)  # different data: fitted afresh


# ------------------------------------------------- frequency-aware lengths (D37)


@pytest.mark.parametrize("length, frequency, expected", [
    ("28d", "M", "1m"), ("45d", "M", "1m"), ("46d", "M", "2m"), ("7d", "M", "1m"), ("2w", "M", "1m"),
    ("90d", "M", "3m"), ("1q", "M", "3m"),
    ("1m", "D", "30d"), ("3m", "D", "91d"), ("4w", "D", "28d"), ("1q", "D", "91d"),
    ("28d", "W", "4w"), ("3m", "W", "13w"), ("6m", "Q", "2q"), ("90d", "Q", "1q"),
    ("28d", "D", "28d"), ("3m", "M", "3m"), ("3y", "D", "3y"), ("1.5y", "M", "1.5y"),
])
def test_lengths_convert_to_the_nearest_whole_unit_of_the_series(length: str, frequency: str, expected: str) -> None:
    assert str(ch.convert_length(ch.Length.parse(length), frequency)) == expected


def test_adapt_rewrites_only_the_steps_it_must_and_lists_them() -> None:
    chain = ch.parse("hol>sma(28d)>cf(2m,18m)>rz(3y)")
    monthly, conversions = ch.adapt(chain, MONTHLY_CTX)
    assert str(monthly) == "hol>sma(1m)>cf(2m,18m)>rz(3y)"
    assert [(str(c.original), str(c.converted)) for c in conversions] == [("sma(28d)", "sma(1m)")]
    assert str(conversions[0]) == "`sma(28d)` → `sma(1m)` (monthly data)"
    assert ch.check(monthly, MONTHLY_CTX) == []
    daily, conversions = ch.adapt(chain, DAILY_CTX)
    assert str(daily) == "hol>sma(28d)>cf(61d,548d)>rz(3y)"
    assert [str(c.converted) for c in conversions] == ["cf(61d,548d)"]
    unchanged, none = ch.adapt(ch.parse("stl>sma>rz"), MONTHLY_CTX)  # auto lengths already fit
    assert str(unchanged) == "stl>sma>rz" and none == []


def test_apply_converts_lengths_only_when_asked() -> None:
    with pytest.raises(ch.ChainError, match="a length in days"):
        run("sma(28d)", MONTHLY)  # the engine stays strict: backtests quote chains exactly (D32)
    result = ch.apply("sma(28d)", MONTHLY, MONTHLY_CTX, convert=True)
    assert str(result.chain) == "sma(1m)" and [str(c.original) for c in result.conversions] == ["sma(28d)"]
    assert np.allclose(result.values, MONTHLY.dropna())  # a one-month average of monthly data is the data
    three = ch.apply("sma(90d)", MONTHLY, MONTHLY_CTX, convert=True)
    assert np.allclose(three.values, run("sma(3m)", MONTHLY).values)
    assert run("sma(3m)", MONTHLY).conversions == []
    daily = pd.Series(np.arange(200.0), index=DAYS)
    days = ch.apply("sma(1m)", daily, ch.Context(frequency="D", unit="ships"), convert=True)
    assert np.allclose(days.values, ch.apply("sma(30d)", daily, ch.Context(frequency="D", unit="ships")).values)


def test_yoy_and_robust_z_steps_respect_known_breaks() -> None:
    ctx = ctx_of("entsoe_load", "NLD")
    assert ctx.breaks == (pd.Timestamp("2016-01-01"), pd.Timestamp("2026-09-10"))
    assert ctx_of("entsoe_load", "DEU").breaks == ()
    dates = pd.date_range("2015-01-01", "2019-12-31", freq="D")
    s = pd.Series(np.where(dates >= "2016-01-01", 332.0, 283.0), index=dates)
    yoy = ch.apply("yoy", s, ctx).values
    assert yoy.index.min() >= pd.Timestamp("2017-01-01") and np.allclose(yoy, 0.0)
    plain = ch.apply("yoy", s, ctx_of("entsoe_load", "DEU")).values
    assert plain.max() > 15  # without the break the step reads as +17%
    z = ch.apply("rz(3y)", s + np.arange(len(s)) % 7, ctx).values
    assert z.index.min() >= pd.Timestamp("2018-01-01")
