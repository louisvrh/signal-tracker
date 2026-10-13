"""Display transform chains (PLAN D32, §5.1-5.3; SIGNALS.md §1).

A chain is an ordered list of steps, written as a short string such as
`stl>sma(28d)>rz(3y)`, that turns one raw series into what a chart shows.
Every step is a pure function (series and context in, `StepOutput` out)
registered in `STEPS` with typed parameters and a label:

* **one-sided**: uses only past data, so appending data never changes an
  earlier output value. Safe to read at any past date.
* **two-sided**: uses later data, so its last values (or, for full-sample
  fits, the whole history) revise as data arrives. The step says how many
  end points are provisional.

The labels are set by tests/test_chain.py's append test, not by argument.
Steps that match an S17 transform (YoY, moving holidays, STL, robust z) wrap
`tracker/signals/transforms.py`. The maths uses statsmodels and scipy only.

Chain strings: steps are separated by `>`; parameters go in brackets,
positionally or as `name=value` (`bk(1.5y,8y,k=3y)`); `auto` (or leaving a
parameter out) lets the step choose from the series' frequency. Lengths are a
number and a unit: d, w, m, q or y. A length can't be finer than the series'
frequency (no days on monthly data), unless the caller asks `apply` to
convert lengths: `adapt` then rewrites every length to the nearest whole
number of the series' own unit (28d -> 1m on monthly data, 1m -> 30d on daily
data; years are kept), and the result lists what changed (PLAN D37). The
dashboard converts; backtests quote chains exactly. `|` is never used,
because it separates fields in the SIGNALS.md hypothesis template.
"""

from __future__ import annotations

import importlib
import math
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.hp_filter import hpfilter

from tracker.signals import transforms as tf
from tracker.signals.moving_holidays import holiday_windows

if TYPE_CHECKING:
    from tracker.registry import Registry

ONE_SIDED, TWO_SIDED = "one-sided", "two-sided"
Sided = Literal["one-sided", "two-sided"]
Domain = Literal["positive", "signed"]
GROUPS = ("Smoothing", "Seasonality", "Detrending", "Band-pass", "Scaling")
SEPARATOR = ">"

# Units whose values can be zero or negative: no logs, no percent changes.
SIGNED_UNITS = {"degC", "degC days"}
FREQUENCY_NAMES = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly"}
FREQUENCY_UNIT = {"D": "d", "W": "w", "M": "m", "Q": "q"}
UNIT_ORDER = "dwmqy"  # finest to coarsest
UNIT_NAMES = {"d": "days", "w": "weeks", "m": "months", "q": "quarters", "y": "years"}
DAYS_PER_UNIT = {"d": 1.0, "w": 7.0, "m": 365.25 / 12, "q": 365.25 / 4, "y": 365.25}
UNITS_PER_YEAR = {"w": 365.25 / 7, "m": 12.0, "q": 4.0, "y": 1.0}
WEEKDAY_SHARE = 5 / 7

# Provisional end points of two-sided filters: the points that typically move
# by more than 20% of the filtered series' spread when more data arrives
# (measured on simulated random walks with drift, 2026-09-27; NOTES.md).
HP_PROVISIONAL = 1.2  # x lambda ** 0.25 periods
CF_PROVISIONAL = 0.75  # x the band's longest period
BUTTERWORTH_PROVISIONAL = 0.5  # x the band's longest period

WEATHER_MODULE = "tracker.signals.weather"
WEATHER_FUNCTION = "adjust"


class ChainError(ValueError):
    """A chain string, a combination of steps or a step's input is invalid; the message says why."""


class StepUnavailable(ChainError):
    """A registered step that can't run yet (weather adjustment until S18 exists)."""


# ------------------------------------------------------------------ lengths

LENGTH_RE = re.compile(r"^(\d+(?:\.\d+)?)([dwmqy])$")


@dataclass(frozen=True)
class Length:
    """A span of time: a positive number and a unit (d, w, m, q or y)."""

    value: float
    unit: str

    @classmethod
    def parse(cls, text: str) -> "Length":
        """Read `28d`, `3m`, `1.5y`, ..."""
        match = LENGTH_RE.match(str(text).strip().lower())
        if not match or float(match.group(1)) <= 0:
            raise ChainError(f"'{text}' is not a length: write a positive number and a unit "
                             "(d, w, m, q or y), e.g. 28d, 3m or 1.5y")
        return cls(float(match.group(1)), match.group(2))

    def __str__(self) -> str:
        return f"{self.value:g}{self.unit}"

    def fits(self, frequency: str) -> bool:
        """True if the unit is no finer than the series' frequency (no days on monthly data)."""
        return UNIT_ORDER.index(self.unit) >= UNIT_ORDER.index(FREQUENCY_UNIT[frequency])

    def days(self) -> float:
        """Length in calendar days."""
        return self.value * DAYS_PER_UNIT[self.unit]

    def years(self) -> float:
        """Length in years."""
        return self.days() / 365.25 if self.unit in "dw" else self.value / UNITS_PER_YEAR[self.unit]

    def periods(self, frequency: str, calendar: str = "all") -> float:
        """Length in observations of a series with this frequency (weekday series: 5 per 7 days)."""
        if frequency == "D":
            return self.days() * (WEEKDAY_SHARE if calendar == "weekdays" else 1.0)
        per_year = {"W": 365.25 / 7, "M": 12.0, "Q": 4.0}[frequency]
        return self.years() * per_year


def convert_length(length: Length, frequency: str) -> Length:
    """`length` as a whole number (at least one) of the series' own unit, rounded to the nearest.

    28d -> 1m and 2w -> 1m on monthly data, 1m -> 30d and 4w -> 28d on daily data,
    1q -> 3m on monthly data. Years, and lengths already in the series' unit, are kept.
    """
    unit = FREQUENCY_UNIT[frequency]
    if length.unit in (unit, "y"):
        return length
    count = max(1, math.floor(length.days() / DAYS_PER_UNIT[unit] + 0.5))
    return Length(float(count), unit)


# --------------------------------------------------------------- parameters


@dataclass(frozen=True)
class Param:
    """A typed step parameter. A default of None means `auto`: chosen from the series."""

    name: str
    kind: Literal["length", "number", "int", "choice", "date"]
    default: Any = None
    help: str = ""
    choices: tuple[str, ...] = ()

    def parse(self, text: str) -> Any:
        """Convert the text form (from a chain string or a form field) to a value."""
        text = str(text).strip()
        if text in ("", "auto"):
            return None
        if self.kind == "length":
            return Length.parse(text)
        if self.kind == "choice":
            if text not in self.choices:
                raise ChainError(f"{self.name} must be one of {', '.join(self.choices)} (got '{text}')")
            return text
        if self.kind == "date":
            try:
                return pd.Timestamp(text).normalize()
            except ValueError:
                raise ChainError(f"{self.name} must be a date like 2020-01-01 (got '{text}')") from None
        try:
            value = int(text) if self.kind == "int" else float(text)
        except ValueError:
            raise ChainError(f"{self.name} must be a{'n integer' if self.kind == 'int' else ' number'} "
                             f"(got '{text}')") from None
        if value <= 0:
            raise ChainError(f"{self.name} must be positive (got '{text}')")
        return value

    def format(self, value: Any) -> str:
        """The text form of a value (`auto` for None)."""
        if value is None:
            return "auto"
        if self.kind == "date":
            return pd.Timestamp(value).strftime("%Y-%m-%d")
        if self.kind == "number":
            return f"{value:.12g}"
        return str(value)


# -------------------------------------------------------------- the context


@dataclass(frozen=True, eq=False)
class Context:
    """What a step may know besides the series itself; built by the caller.

    Attributes:
        frequency: D, W, M or Q.
        unit: the raw series' unit (decides whether it can be negative).
        indicator, country: ids, for messages and the weather model.
        calendar: `all` or `weekdays` (weekday-only series).
        signal_group: the indicator's signal group (weather adjustment needs power_demand).
        lunar_new_year: the country observes Lunar New Year (monthly January-February combining).
        holiday_calendar: the country's `holidays` package code (e.g. DE), for the weather model's
            public-holiday terms.
        holiday_windows: moving-holiday windows (holiday, start, end) covering the series.
        extra: other series or tables, e.g. degree days for weather adjustment, or a
            precomputed STL (`stl`: a frame with observed and seasonal columns).
        breaks: dates of known structural breaks in the series (indicators.yaml
            `breaks`); YoY and robust z never compare across one (PLAN D41).
        domain: the sign of the step's input; set by `apply` as it walks the chain.
    """

    frequency: str
    unit: str = ""
    indicator: str = ""
    country: str = ""
    calendar: str = "all"
    signal_group: str | None = None
    lunar_new_year: bool = False
    holiday_calendar: str = ""
    holiday_windows: pd.DataFrame | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    breaks: tuple[pd.Timestamp, ...] = ()
    domain: Domain | None = None

    @property
    def input_domain(self) -> Domain:
        """`signed` if the input can be zero or negative, else `positive`."""
        if self.domain is not None:
            return self.domain
        return "signed" if self.unit in SIGNED_UNITS else "positive"


@dataclass
class StepOutput:
    """What a step returns: its values on the input's dates (NaN where undefined), the
    component it removed (seasonal, trend or noise) if any, and how many of the last
    input dates are provisional (two-sided steps)."""

    values: pd.Series
    removed: pd.Series | None = None
    provisional: int = 0


# ---------------------------------------------------------------- the steps


@dataclass(frozen=True)
class StepSpec:
    """A registered transform.

    Attributes:
        key: the name used in chain strings.
        name, group, help: for people.
        sided: one-sided or two-sided (decided by the append test).
        params: typed parameters, in positional order.
        func: `func(series, context, **params) -> StepOutput`.
        output: the sign of the output: `same` as the input, `signed` or `positive`.
        needs_positive: rejects an input that can be zero or negative.
        keeps_units: the output is in the input's units and scale (drawn over the raw series).
        revises: for two-sided steps, whether revisions sit at the `end` or spread over
            `all` of the history (full-sample fits).
        check: extra static checks `check(params, context) -> [problems]`.
        unavailable: returns why the step can't run yet, or None.
    """

    key: str
    name: str
    group: str
    sided: Sided
    params: tuple[Param, ...]
    func: Callable[..., StepOutput]
    help: str
    output: Literal["same", "signed", "positive"] = "same"
    needs_positive: bool = False
    keeps_units: bool = False
    revises: Literal["none", "end", "all"] = "none"
    check: Callable[[dict[str, Any], Context], list[str]] | None = None
    unavailable: Callable[[], str | None] | None = None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.key})"


STEPS: dict[str, StepSpec] = {}


def _register(spec: StepSpec) -> None:
    STEPS[spec.key] = spec


# Helpers shared by the step functions -----------------------------------


def _count_since(index: pd.DatetimeIndex, start: pd.Timestamp) -> int:
    return int((index >= start).sum())


def _last_days(index: pd.DatetimeIndex, days: float) -> int:
    """How many dates fall in the last `days` calendar days (inclusive of the last date)."""
    return int((index > index[-1] - pd.Timedelta(days=days)).sum()) if len(index) else 0


def _window(length: Length, ctx: Context) -> tuple[int | str, float]:
    """A rolling window (calendar days for daily series, else a count) and the values a full one holds."""
    if ctx.frequency == "D":
        days = max(1, round(length.days()))
        return f"{days}D", length.periods("D", ctx.calendar)
    n = max(1, round(length.periods(ctx.frequency)))
    return n, n


def _count(length: Length, ctx: Context) -> int:
    return max(1, round(length.periods(ctx.frequency, ctx.calendar)))


def _grid(s: pd.Series, ctx: Context) -> pd.Series:
    """The series on a regular grid (gaps interpolated), for filters that assume one."""
    return tf.regular(s, ctx.frequency, ctx.calendar)


def _back(values: np.ndarray | pd.Series, grid: pd.Series, s: pd.Series) -> pd.Series:
    """Grid results back on the input's dates."""
    return pd.Series(np.asarray(values, dtype=float), index=grid.index).reindex(s.index)


def _grid_tail(grid: pd.Series, s: pd.Series, periods: int) -> int:
    """Input dates within the last `periods` grid periods."""
    if periods <= 0 or grid.empty:
        return 0
    return _count_since(s.index, grid.index[max(len(grid) - periods, 0)])


def _auto(length: Length | None, defaults: Mapping[str, str], ctx: Context) -> Length:
    return length if length is not None else Length.parse(defaults[ctx.frequency])


def _check_lengths(params: Mapping[str, Any], ctx: Context) -> list[str]:
    problems = []
    for name, value in params.items():
        if isinstance(value, Length) and not value.fits(ctx.frequency):
            freq = FREQUENCY_NAMES[ctx.frequency]
            allowed = ", ".join(u for u in UNIT_ORDER if UNIT_ORDER.index(u) >= UNIT_ORDER.index(FREQUENCY_UNIT[ctx.frequency]))
            problems.append(f"a length in {UNIT_NAMES[value.unit]} ({name}={value}) can't be used on {freq} data; "
                            f"use {allowed}")
    return problems


# Smoothing ------------------------------------------------------------------

SMOOTH_DEFAULT = {"D": "28d", "W": "4w", "M": "3m", "Q": "4q"}
LENGTH_HELP = "window length, e.g. 28d, 3m or 1y"


def _sma(s: pd.Series, ctx: Context, n: Length | None) -> StepOutput:
    window, expected = _window(_auto(n, SMOOTH_DEFAULT, ctx), ctx)
    out = s.rolling(window, min_periods=max(1, math.ceil(tf.MIN_WINDOW_SHARE * expected))).mean()
    return StepOutput(out, removed=s - out)


def _csma(s: pd.Series, ctx: Context, n: Length | None) -> StepOutput:
    length = _auto(n, SMOOTH_DEFAULT, ctx)
    window, expected = _window(length, ctx)
    out = s.rolling(window, center=True, min_periods=max(1, math.ceil(expected / 2))).mean()
    if ctx.frequency == "D":
        provisional = _last_days(s.index, math.floor(length.days() / 2))
    else:
        provisional = int(window) // 2
    return StepOutput(out, removed=s - out, provisional=provisional)


def _ema(s: pd.Series, ctx: Context, n: Length | None) -> StepOutput:
    length = _auto(n, SMOOTH_DEFAULT, ctx)
    if ctx.frequency == "D":
        # Span in calendar days turned into a half-life, so gaps (weekends) count as time.
        alpha = 2 / (length.days() + 1)
        halflife = math.log(2) / -math.log(1 - alpha)
        out = s.ewm(halflife=pd.Timedelta(days=halflife), times=s.index).mean()
    else:
        out = s.ewm(span=_count(length, ctx)).mean()
    return StepOutput(out, removed=s - out)


_WINDOW_PARAM = Param("n", "length", None, LENGTH_HELP)
_register(StepSpec("sma", "Trailing moving average", "Smoothing", ONE_SIDED, (_WINDOW_PARAM,), _sma,
                   "Average of the window ending on each date (default 28d daily, 3m monthly).",
                   keeps_units=True))
_register(StepSpec("csma", "Centred moving average", "Smoothing", TWO_SIDED, (_WINDOW_PARAM,), _csma,
                   "Average of the window centred on each date; the last half-window uses fewer values "
                   "and revises.", keeps_units=True, revises="end"))
_register(StepSpec("ema", "Exponential moving average", "Smoothing", ONE_SIDED,
                   (Param("n", "length", None, "span, e.g. 28d or 6m"),), _ema,
                   "Exponentially weighted average with the given span (daily: in calendar days).",
                   keeps_units=True))


# Seasonality ------------------------------------------------------------------


def _yoy(s: pd.Series, ctx: Context, mode: str | None) -> StepOutput:
    method = mode or ("pct" if ctx.input_domain == "positive" else "diff")
    return StepOutput(tf.yoy(s, ctx.frequency, method=method, lunar_new_year=ctx.lunar_new_year,
                             calendar=ctx.calendar, breaks=ctx.breaks))


def _check_yoy(params: Mapping[str, Any], ctx: Context) -> list[str]:
    if params["mode"] == "pct" and ctx.input_domain == "signed":
        return ["a YoY percent change needs a positive series, but this input can be zero or negative; "
                "use yoy(diff)"]
    return []


def _precomputed_stl(s: pd.Series, ctx: Context) -> pd.DataFrame | None:
    """A precomputed STL (data/signals, from the daily run) that was fitted on exactly this series.

    `ctx.extra["stl"]` is one frame or a list of frames (columns observed and
    seasonal, indexed by date), e.g. fitted on the raw and on the
    holiday-adjusted series.
    """
    frames = ctx.extra.get("stl")
    if frames is None:
        return None
    for pre in frames if isinstance(frames, (list, tuple)) else [frames]:
        if (len(pre) == len(s) and pre.index.equals(s.index)
                and np.allclose(pre["observed"].to_numpy(), s.to_numpy(), rtol=1e-12, atol=0)):
            return pre
    return None


def _stl(s: pd.Series, ctx: Context) -> StepOutput:
    parts = _precomputed_stl(s, ctx)
    if parts is None:
        parts = tf.stl_components(s, ctx.frequency, ctx.calendar)
    periods = tf.seasonal_periods(ctx.frequency, ctx.calendar)
    if ctx.frequency == "D":
        provisional = _last_days(s.index, 365)
    else:
        provisional = min(periods[-1], len(s))
    return StepOutput(parts["observed"] - parts["seasonal"], removed=parts["seasonal"], provisional=provisional)


def _hol(s: pd.Series, ctx: Context) -> StepOutput:
    windows = ctx.holiday_windows
    if ctx.frequency == "D" and windows is not None and not windows.empty:
        out = tf.interpolate_holidays(s, windows)
        # A window is settled once the same weekday has an ordinary day after it (7 days).
        last = s.index[-1]
        open_windows = windows[(windows["end"] >= last - pd.Timedelta(days=6)) & (windows["start"] <= last)]
        provisional = _count_since(s.index, open_windows["start"].min()) if not open_windows.empty else 0
    elif ctx.frequency == "M" and ctx.lunar_new_year:
        out = tf.combine_jan_feb(s)
        provisional = 1 if s.index[-1].month == 1 else 0
    else:
        out, provisional = s, 0  # no moving holiday to adjust for
    return StepOutput(out, removed=s - out, provisional=provisional)


def weather_unavailable() -> str | None:
    """Why weather adjustment can't run yet (None once S18's model exists)."""
    try:
        module = importlib.import_module(WEATHER_MODULE)
    except ImportError:
        return ("Weather adjustment needs the weather-normalised demand model from session S18 "
                "(tracker/signals/weather.py), which isn't built yet.")
    if not callable(getattr(module, WEATHER_FUNCTION, None)):
        return f"Weather adjustment needs {WEATHER_MODULE}.{WEATHER_FUNCTION}(values, context), which doesn't exist yet."
    return None


def _weather(s: pd.Series, ctx: Context) -> StepOutput:
    reason = weather_unavailable()
    if reason:
        raise StepUnavailable(reason)
    adjusted = getattr(importlib.import_module(WEATHER_MODULE), WEATHER_FUNCTION)(s, ctx)
    adjusted = pd.Series(adjusted, dtype="float64").reindex(s.index)
    return StepOutput(adjusted, removed=s - adjusted)


def _check_weather(params: Mapping[str, Any], ctx: Context) -> list[str]:
    what = ctx.indicator or "this series"
    if ctx.signal_group != "power_demand":
        return [f"weather adjustment only applies to power-demand indicators (e.g. entsoe_load, eia_us48_load); "
                f"{what} isn't one"]
    if ctx.frequency != "D":
        return [f"weather adjustment needs daily load (the model works day by day); {what} is "
                f"{FREQUENCY_NAMES[ctx.frequency]}"]
    return []


_register(StepSpec("yoy", "Year-on-year change", "Seasonality", ONE_SIDED,
                   (Param("mode", "choice", None, "pct (percent) or diff (difference); auto picks pct for "
                          "positive series", ("pct", "diff")),), _yoy,
                   "Daily: 28-day window vs the same window 364 days earlier. Monthly: vs 12 months earlier; "
                   "January and February combined for Lunar New Year countries (SIGNALS.md §1).",
                   output="signed", check=_check_yoy))
_register(StepSpec("stl", "STL seasonal removal", "Seasonality", TWO_SIDED, (), _stl,
                   "Removes the seasonal component (STL; MSTL with weekly and annual cycles for daily data). "
                   "Needs 3 years of data; the last year revises.", keeps_units=True, revises="end"))
_register(StepSpec("hol", "Moving-holiday adjustment", "Seasonality", TWO_SIDED, (), _hol,
                   "Daily: holiday days replaced by a line between the nearest ordinary days of the same weekday. "
                   "Monthly, Lunar New Year countries: January and February both set to their average.",
                   keeps_units=True, revises="end"))
_register(StepSpec("wx", "Weather adjustment", "Seasonality", ONE_SIDED, (), _weather,
                   "Daily power demand in normal weather: actual load minus the heating and cooling degree-day "
                   "effect of a model refitted each month on the 3 years before it (SIGNALS.md §2).",
                   needs_positive=True, keeps_units=True, check=_check_weather, unavailable=weather_unavailable))


# Detrending -------------------------------------------------------------------


def _d1(s: pd.Series, ctx: Context) -> StepOutput:
    return StepOutput(s.diff())


def _dlog(s: pd.Series, ctx: Context) -> StepOutput:
    return StepOutput(100 * np.log(s.where(s > 0)).diff())


def _lin(s: pd.Series, ctx: Context) -> StepOutput:
    t = ((s.index - s.index[0]).days / 365.25).to_numpy()
    fit = OLS(s.to_numpy(), np.column_stack([np.ones_like(t), t])).fit()
    trend = pd.Series(np.asarray(fit.fittedvalues), index=s.index)
    return StepOutput(s - trend, removed=trend)


def hp_lambda(frequency: str, calendar: str = "all") -> float:
    """Ravn-Uhlig smoothing parameter: 1600 x (periods per quarter)^4 (monthly 129,600)."""
    per_year = {"D": 365.25 * (WEEKDAY_SHARE if calendar == "weekdays" else 1.0), "W": 365.25 / 7,
                "M": 12.0, "Q": 4.0}[frequency]
    return 1600 * (per_year / 4) ** 4


def _hp(s: pd.Series, ctx: Context, lamb: float | None) -> StepOutput:
    lamb = lamb or hp_lambda(ctx.frequency, ctx.calendar)
    grid = _grid(s, ctx)
    if len(grid) < 3:
        raise ChainError("the HP filter needs at least 3 values")
    cycle, trend = hpfilter(grid.to_numpy(), lamb)
    provisional = _grid_tail(grid, s, math.ceil(HP_PROVISIONAL * lamb ** 0.25))
    return StepOutput(_back(cycle, grid, s), removed=_back(trend, grid, s), provisional=provisional)


HAMILTON_LAGS = {"D": 7, "W": 4, "M": 12, "Q": 4}


def _ham(s: pd.Series, ctx: Context, h: Length | None, p: int | None) -> StepOutput:
    grid = _grid(s, ctx)
    horizon = _count(h or Length(2, "y"), ctx)
    lags = p or HAMILTON_LAGS[ctx.frequency]
    y = grid.to_numpy()
    first = horizon + lags - 1
    if len(y) < first + 2 * (lags + 1):
        raise ChainError(f"the Hamilton filter with h={h or '2y'} and p={lags} needs more data "
                         f"(at least {first + 2 * (lags + 1)} values; the series has {len(y)})")
    rows = np.arange(first, len(y))
    X = np.column_stack([np.ones(len(rows))] + [y[rows - horizon - j] for j in range(lags)])
    fit = OLS(y[rows], X).fit()
    trend = np.full(len(y), np.nan)
    trend[rows] = fit.fittedvalues
    trend_s = _back(trend, grid, s)
    return StepOutput(s - trend_s, removed=trend_s)


_register(StepSpec("d1", "First difference", "Detrending", ONE_SIDED, (), _d1,
                   "Change from the previous value.", output="signed"))
_register(StepSpec("dlog", "Log difference", "Detrending", ONE_SIDED, (), _dlog,
                   "100 x change in the natural log from the previous value (about a percent change).",
                   output="signed", needs_positive=True))
_register(StepSpec("lin", "Linear detrend", "Detrending", TWO_SIDED, (), _lin,
                   "Deviation from a straight line fitted to the whole series; the whole line moves as data "
                   "arrives.", output="signed", revises="all"))
_register(StepSpec("hp", "Hodrick-Prescott filter", "Detrending", TWO_SIDED,
                   (Param("lamb", "number", None, "smoothing parameter; auto: 1600 x (periods per quarter)^4"),),
                   _hp, "Cycle around the HP trend (the trend is the removed component).",
                   output="signed", revises="end"))
_register(StepSpec("ham", "Hamilton filter", "Detrending", TWO_SIDED,
                   (Param("h", "length", None, "horizon, default 2y"),
                    Param("p", "int", None, "number of lags; auto: 12 monthly, 4 quarterly, 7 daily")),
                   _ham, "Error of a regression of each value on the p values h earlier (Hamilton 2018). "
                   "Coefficients use the whole sample, so the history revises as data arrives.",
                   output="signed", revises="all"))


# Band-pass --------------------------------------------------------------------

_LOW = Param("low", "length", None, "shortest cycle kept, default 1.5y")
_HIGH = Param("high", "length", None, "longest cycle kept, default 8y")


def _band(ctx: Context, low: Length | None, high: Length | None) -> tuple[float, float]:
    low, high = low or Length(1.5, "y"), high or Length(8, "y")
    return low.periods(ctx.frequency, ctx.calendar), high.periods(ctx.frequency, ctx.calendar)


def _check_band(params: Mapping[str, Any], ctx: Context) -> list[str]:
    problems: list[str] = []
    low, high = params["low"] or Length(1.5, "y"), params["high"] or Length(8, "y")
    lo, hi = _band(ctx, low, high)
    if lo >= hi:
        problems.append(f"the shortest cycle ({low}) must be shorter than the longest ({high})")
    if lo < 2 - 1e-9:
        problems.append(f"the shortest cycle ({low}) is under two periods of {FREQUENCY_NAMES[ctx.frequency]} data, "
                        "the shortest cycle it can show")
    return problems


def _need_cycle(grid: pd.Series, hi: float, high: Length | None, key: str) -> None:
    if len(grid) < hi:
        raise ChainError(f"{key} needs at least one full longest cycle of data ({high or Length(8, 'y')}); "
                         f"the series covers {len(grid) / hi * (high or Length(8, 'y')).years():.1f} years")


def _bk(s: pd.Series, ctx: Context, low: Length | None, high: Length | None, k: Length | None) -> StepOutput:
    lo, hi = _band(ctx, low, high)
    grid = _grid(s, ctx)
    K = _count(k or Length(3, "y"), ctx)
    if len(grid) <= 2 * K:
        raise ChainError(f"Baxter-King with k={k or '3y'} loses {K} values at each end and needs more than "
                         f"{2 * K} values; the series has {len(grid)}")
    cycle = np.full(len(grid), np.nan)
    cycle[K:len(grid) - K] = np.asarray(bkfilter(grid.to_numpy(), lo, hi, K)).ravel()
    out = _back(cycle, grid, s)
    return StepOutput(out, removed=s - out, provisional=_grid_tail(grid, s, K))


def _cf(s: pd.Series, ctx: Context, low: Length | None, high: Length | None) -> StepOutput:
    lo, hi = _band(ctx, low, high)
    grid = _grid(s, ctx)
    _need_cycle(grid, hi, high, "Christiano-Fitzgerald")
    cycle, _ = cffilter(grid.to_numpy(), lo, hi, drift=True)
    out = _back(cycle, grid, s)
    return StepOutput(out, removed=s - out, provisional=_grid_tail(grid, s, math.ceil(CF_PROVISIONAL * hi)))


def _butter(lo: float, hi: float, order: int) -> np.ndarray:
    """Butterworth band-pass in second-order sections; a high-pass when `lo` is the 2-period limit."""
    if lo <= 2 + 1e-9:
        return signal.butter(order, 1 / hi, btype="highpass", output="sos", fs=1)
    return signal.butter(order, [1 / hi, 1 / lo], btype="bandpass", output="sos", fs=1)


def _bwz(s: pd.Series, ctx: Context, low: Length | None, high: Length | None, order: int | None) -> StepOutput:
    lo, hi = _band(ctx, low, high)
    grid = _grid(s, ctx)
    _need_cycle(grid, hi, high, "The Butterworth filter")
    x = grid.to_numpy()
    cycle = signal.sosfiltfilt(_butter(lo, hi, order or 2), x - x.mean())
    out = _back(cycle, grid, s)
    return StepOutput(out, removed=s - out,
                      provisional=_grid_tail(grid, s, math.ceil(BUTTERWORTH_PROVISIONAL * hi)))


def _bwc(s: pd.Series, ctx: Context, low: Length | None, high: Length | None, order: int | None) -> StepOutput:
    lo, hi = _band(ctx, low, high)
    grid = _grid(s, ctx)
    sos = _butter(lo, hi, order or 2)
    x = grid.to_numpy()
    cycle, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])
    out = _back(cycle, grid, s)
    return StepOutput(out, removed=s - out)


_ORDER = Param("order", "int", None, "filter order, default 2")
_register(StepSpec("bk", "Baxter-King band-pass", "Band-pass", TWO_SIDED,
                   (_LOW, _HIGH, Param("k", "length", None, "lead/lag length, default 3y")), _bk,
                   "Symmetric moving-average band-pass; the last k of data have no value until more arrives.",
                   output="signed", revises="end", check=_check_band))
_register(StepSpec("cf", "Christiano-Fitzgerald band-pass", "Band-pass", TWO_SIDED, (_LOW, _HIGH), _cf,
                   "Full-sample asymmetric band-pass (random walk with drift); the end revises most.",
                   output="signed", revises="end", check=_check_band))
_register(StepSpec("bwz", "Butterworth band-pass, zero-phase", "Band-pass", TWO_SIDED, (_LOW, _HIGH, _ORDER), _bwz,
                   "Butterworth filter run forwards and backwards (no phase shift); the end revises.",
                   output="signed", revises="end", check=_check_band))
_register(StepSpec("bwc", "Butterworth band-pass, causal", "Band-pass", ONE_SIDED, (_LOW, _HIGH, _ORDER), _bwc,
                   "Butterworth filter run forwards only: real-time, but cycles are shifted later.",
                   output="signed", check=_check_band))


# Scaling ----------------------------------------------------------------------


def _level(s: pd.Series, ctx: Context) -> StepOutput:
    return StepOutput(s)


def _log(s: pd.Series, ctx: Context) -> StepOutput:
    return StepOutput(np.log(s.where(s > 0)))


def _idx(s: pd.Series, ctx: Context, date: pd.Timestamp | None) -> StepOutput:
    pos = 0 if date is None else int(s.index.searchsorted(date))
    if pos >= len(s):  # the base date isn't in the data yet
        return StepOutput(pd.Series(np.nan, index=s.index))
    base = s.iloc[pos]
    if not base > 0:
        raise ChainError(f"idx: the value on the base date {s.index[pos]:%Y-%m-%d} is {base:g}; it must be positive")
    return StepOutput(100 * s / base)


def _min_history(years: float, ctx: Context) -> tuple[int | str, int, pd.Timedelta | None]:
    """Rolling window over `years` and the history a value needs: two thirds of the window."""
    if ctx.frequency == "D":
        return f"{round(365 * years)}D", 1, pd.Timedelta(days=round(365 * years * 2 / 3))
    per_year = {"W": 52, "M": 12, "Q": 4}[ctx.frequency]
    n = max(1, round(per_year * years))
    return n, max(1, round(n * 2 / 3)), None


def _rz(s: pd.Series, ctx: Context, window: Length | None) -> StepOutput:
    years = (window or Length(3, "y")).years()
    return StepOutput(tf.robust_z(s, ctx.frequency, window_years=years, min_history_months=round(years * 8),
                                  breaks=ctx.breaks))


def _pr(s: pd.Series, ctx: Context, window: Length | None) -> StepOutput:
    win, min_periods, history = _min_history((window or Length(5, "y")).years(), ctx)
    out = s.rolling(win, min_periods=min_periods).apply(lambda w: (w <= w[-1]).mean() * 100, raw=True)
    if history is not None:
        out = out.where(s.index >= s.index[0] + history)
    return StepOutput(out)


def _prf(s: pd.Series, ctx: Context) -> StepOutput:
    return StepOutput(s.rank(pct=True) * 100)


_register(StepSpec("level", "Level (no change)", "Scaling", ONE_SIDED, (), _level,
                   "The series as it is.", keeps_units=True))
_register(StepSpec("log", "Natural log", "Scaling", ONE_SIDED, (), _log,
                   "Natural logarithm (zero values give no value).", output="signed", needs_positive=True))
_register(StepSpec("idx", "Index = 100 at a date", "Scaling", ONE_SIDED,
                   (Param("date", "date", None, "base date (the first value on or after it); auto: the first value"),),
                   _idx, "The series divided by its value on the base date, times 100.",
                   output="positive", needs_positive=True))
_register(StepSpec("rz", "Robust z-score", "Scaling", ONE_SIDED,
                   (Param("window", "length", None, "trailing window, default 3y"),), _rz,
                   "(x - rolling median) / (1.4826 x rolling MAD) over a trailing window; needs two thirds of "
                   "the window as history (24 months for 3y), PLAN §5.3.", output="signed"))
_register(StepSpec("pr", "Percentile rank, trailing", "Scaling", ONE_SIDED,
                   (Param("window", "length", None, "trailing window, default 5y"),), _pr,
                   "Share of the trailing window's values at or below the current one (0-100); needs two "
                   "thirds of the window as history.", output="positive"))
_register(StepSpec("prf", "Percentile rank, full sample", "Scaling", TWO_SIDED, (), _prf,
                   "Rank among all values of the series (0-100); every rank moves as data arrives.",
                   output="positive", revises="all"))


# ------------------------------------------------------------- steps, chains


def step_spec(key: str) -> StepSpec:
    """The registered step called `key`, with a clear error for unknown names."""
    try:
        return STEPS[key]
    except KeyError:
        raise ChainError(f"unknown step '{key}'; known steps: {', '.join(STEPS)}") from None


@dataclass(frozen=True)
class Step:
    """One step of a chain: a registered key and one value per parameter (None = auto)."""

    key: str
    args: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        spec = step_spec(self.key)
        if len(self.args) > len(spec.params):
            raise ChainError(f"{self.key} takes at most {len(spec.params)} parameter(s), got {len(self.args)}")
        if len(self.args) < len(spec.params):
            object.__setattr__(self, "args", tuple(self.args) + tuple(p.default for p in spec.params[len(self.args):]))

    @property
    def spec(self) -> StepSpec:
        return STEPS[self.key]

    @property
    def params(self) -> dict[str, Any]:
        """Parameter values by name (None = auto)."""
        return {p.name: v for p, v in zip(self.spec.params, self.args)}

    def __str__(self) -> str:
        params = self.spec.params
        shown = list(self.args)
        while shown and shown[-1] == params[len(shown) - 1].default:
            shown.pop()
        if not shown:
            return self.key
        return f"{self.key}({','.join(p.format(v) for p, v in zip(params, shown))})"


@dataclass(frozen=True)
class Chain:
    """An ordered list of steps; `str(chain)` is its short form, `parse` reads it back."""

    steps: tuple[Step, ...] = ()

    def __str__(self) -> str:
        return SEPARATOR.join(str(s) for s in self.steps)

    @property
    def two_sided(self) -> bool:
        return any(s.spec.sided == TWO_SIDED for s in self.steps)

    def one_sided_only(self) -> tuple["Chain", list[Step]]:
        """The chain without its two-sided steps (the "real-time only" view), and the steps dropped."""
        dropped = [s for s in self.steps if s.spec.sided == TWO_SIDED]
        return Chain(tuple(s for s in self.steps if s.spec.sided == ONE_SIDED)), dropped


TOKEN_RE = re.compile(r"^([a-z][a-z0-9]*)(?:\((.*)\))?$")


def parse_step(text: str) -> Step:
    """Read one step, e.g. `sma(28d)`, `bk(1.5y,8y,k=3y)` or `stl`."""
    text = text.strip()
    match = TOKEN_RE.match(text)
    if not match:
        raise ChainError(f"can't read step '{text}': write a step name, optionally with parameters in brackets, "
                         "e.g. sma(28d)")
    key, inner = match.group(1), match.group(2)
    spec = step_spec(key)
    values: list[Any] = [p.default for p in spec.params]
    if inner is not None and inner.strip():
        names = [p.name for p in spec.params]
        keyword_seen = False
        for position, part in enumerate(inner.split(",")):
            name, sep, value = part.partition("=")
            if sep:
                name = name.strip()
                if name not in names:
                    raise ChainError(f"{key} has no parameter '{name}'; its parameters are {', '.join(names) or 'none'}")
                i, keyword_seen = names.index(name), True
            else:
                if keyword_seen:
                    raise ChainError(f"{key}: positional parameters must come before name=value ones")
                if position >= len(names):
                    raise ChainError(f"{key} takes at most {len(names)} parameter(s): {', '.join(names) or 'none'}")
                i, value = position, part
            try:
                values[i] = spec.params[i].parse(value)
            except ChainError as exc:
                raise ChainError(f"{key}: {exc}") from None
    return Step(key, tuple(values))


def parse(text: str) -> Chain:
    """Read a chain string such as `stl>sma(28d)>rz(3y)`; an empty string is the empty chain."""
    if "|" in text:
        raise ChainError("don't use '|' in a chain: it separates fields in the SIGNALS.md hypothesis template; "
                         "separate steps with '>'")
    text = text.strip()
    if not text:
        return Chain()
    return Chain(tuple(parse_step(part) for part in text.split(SEPARATOR)))


@dataclass(frozen=True)
class Conversion:
    """A step whose lengths `adapt` rewrote in the series' own unit."""

    original: Step
    converted: Step
    frequency: str

    def __str__(self) -> str:
        return f"`{self.original}` → `{self.converted}` ({FREQUENCY_NAMES[self.frequency]} data)"


def adapt(chain: Chain, ctx: Context) -> tuple[Chain, list[Conversion]]:
    """The chain with every length converted to the series' own unit (`convert_length`),
    and the steps that changed.

    Lets one chain string serve series of every frequency: the preset `sma(28d)`
    runs as `sma(1m)` on monthly data instead of being rejected (PLAN D37).
    """
    steps, conversions = [], []
    for step in chain.steps:
        args = tuple(convert_length(a, ctx.frequency) if isinstance(a, Length) else a for a in step.args)
        converted = Step(step.key, args)
        if converted != step:
            conversions.append(Conversion(step, converted, ctx.frequency))
        steps.append(converted)
    return Chain(tuple(steps)), conversions


def _next_domain(spec: StepSpec, domain: Domain) -> Domain:
    return domain if spec.output == "same" else spec.output


def check(chain: Chain, ctx: Context) -> list[str]:
    """Every reason the chain can't be applied to a series with this context (empty if fine)."""
    problems: list[str] = []
    domain = ctx.input_domain
    for n, step in enumerate(chain.steps, start=1):
        spec = step.spec
        where = f"step {n} `{step}`"
        reason = spec.unavailable() if spec.unavailable else None
        if reason:
            problems.append(f"{where}: {reason}")
        step_ctx = replace(ctx, domain=domain)
        own = _check_lengths(step.params, step_ctx)
        if not own and spec.check:
            own = spec.check(step.params, step_ctx)
        problems += [f"{where}: {p}" for p in own]
        if spec.needs_positive and domain == "signed":
            source = f"after `{chain.steps[n - 2]}`" if n > 1 else f"its unit is {ctx.unit}"
            problems.append(f"{where}: {spec.name.lower()} needs a positive series, but this input can be zero or "
                            f"negative ({source})")
        domain = _next_domain(spec, domain)
    return problems


def validate(chain: Chain, ctx: Context) -> None:
    """Raise ChainError listing every problem `check` finds."""
    problems = check(chain, ctx)
    if problems:
        raise ChainError("; ".join(problems))


@dataclass
class ChainResult:
    """A chain applied to one series.

    Attributes:
        chain: the chain applied.
        raw: the input series (missing values dropped).
        values: the transformed series (missing values dropped).
        removed: (step, component) for each step that removed something (seasonal, trend, noise).
        provisional_from: first date whose value may still revise, or None.
        revises_history: a full-sample step means every value may move a little.
        keeps_units: every step kept the raw series' units and scale.
        conversions: steps whose lengths were converted to the series' unit (`apply(convert=True)`);
            `chain` is then the converted chain.
    """

    chain: Chain
    raw: pd.Series
    values: pd.Series
    removed: list[tuple[str, pd.Series]]
    provisional_from: pd.Timestamp | None
    revises_history: bool
    keeps_units: bool
    conversions: list[Conversion] = field(default_factory=list)

    @property
    def two_sided(self) -> bool:
        return self.chain.two_sided

    @property
    def provisional(self) -> int:
        """How many of the last transformed values may still revise."""
        return 0 if self.provisional_from is None else _count_since(self.values.index, self.provisional_from)


def _extend_provisional(start: pd.Timestamp | None, index: pd.DatetimeIndex, count: int) -> pd.Timestamp | None:
    """Move the provisional start `count` dates earlier (a two-sided step looks that far ahead)."""
    if len(index) == 0 or (start is None and count <= 0):
        return start
    end = len(index) if start is None else int(index.searchsorted(start))
    position = max(end - count, 0)
    return index[position] if position < len(index) else start


def apply(chain: Chain | str, values: pd.Series, ctx: Context, convert: bool = False) -> ChainResult:
    """Apply a chain to one series (indexed by date) and return its `ChainResult`.

    With `convert`, lengths are first converted to the series' own unit
    (`adapt`: 28d -> 1m on monthly data) instead of being rejected, and the
    result lists the conversions. Raises ChainError (or StepUnavailable) with
    a clear message if the chain doesn't fit the series or a step can't run
    on it (e.g. too little data).
    """
    chain = parse(chain) if isinstance(chain, str) else chain
    conversions: list[Conversion] = []
    if convert:
        chain, conversions = adapt(chain, ctx)
    validate(chain, ctx)
    raw = tf.clean(values)
    s, domain = raw, ctx.input_domain
    removed: list[tuple[str, pd.Series]] = []
    start: pd.Timestamp | None = None
    for step in chain.steps:
        spec = step.spec
        if s.empty:
            break
        try:
            out = spec.func(s, replace(ctx, domain=domain), **step.params)
        except ChainError as exc:
            raise type(exc)(f"`{step}`: {exc}") from None
        except ValueError as exc:
            raise ChainError(f"`{step}`: {exc}") from None
        if spec.sided == TWO_SIDED:
            start = _extend_provisional(start, out.values.index, out.provisional)
        if out.removed is not None:
            removed.append((str(step), tf.clean(out.removed)))
        s, domain = tf.clean(out.values), _next_domain(spec, domain)
    return ChainResult(
        chain=chain, raw=raw, values=s, removed=removed, provisional_from=start,
        revises_history=any(st.spec.revises == "all" for st in chain.steps),
        keeps_units=all(st.spec.keeps_units for st in chain.steps),
        conversions=conversions,
    )


def series_context(
    registry: "Registry",
    indicator_id: str,
    country: str,
    values: pd.Series,
    extra: Mapping[str, Any] | None = None,
) -> Context:
    """`context_for` with the country's moving-holiday windows for the years the series covers.

    The dashboard and the daily STL precompute both build contexts this way,
    so a precomputed `hol>stl` sees exactly the input the dashboard's does.
    """
    ref = registry.countries.get(country)
    windows = None
    if ref is not None and ref.moving_holidays and len(values):
        dates = pd.DatetimeIndex(values.index)
        windows = holiday_windows(ref.holiday_calendar, ref.moving_holidays, dates.min().year, dates.max().year)
    return context_for(registry, indicator_id, country, windows, extra)


def context_for(
    registry: "Registry",
    indicator_id: str,
    country: str,
    holiday_windows: pd.DataFrame | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Context:
    """The context of one registered series (frequency, unit, calendar, Lunar New Year, known breaks, ...).

    `holiday_windows` comes from `moving_holidays.holiday_windows` for the
    country (None if it has none).
    """
    ind = registry.indicator(indicator_id)
    ref = registry.countries.get(country)
    return Context(
        frequency=ind.frequency, unit=ind.unit, indicator=ind.id, country=country, calendar=ind.calendar,
        signal_group=ind.signal_group,
        lunar_new_year=ref is not None and "lunar_new_year" in ref.moving_holidays,
        holiday_calendar=ref.holiday_calendar if ref is not None else "",
        holiday_windows=holiday_windows, extra=dict(extra or {}),
        breaks=tuple(pd.Timestamp(b.date) for b in registry.breaks_for(ind, country)),
    )
