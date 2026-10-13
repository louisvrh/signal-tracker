"""Display transforms and Plotly figures for the dashboard.

The functions here are pure (data in, data or figures out). The YoY used for
display is deliberately simple and follows PLAN §5.2 for daily data (28-day
window compared with the same window 364 days earlier, which keeps weekdays
aligned); S17's `tracker/signals/transforms.py` adds the moving-holiday
handling that signals need.
"""

from __future__ import annotations

import html
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from tracker.signals import transforms as tf
from tracker.signals.chain import Chain, ChainResult, Length

# Interval-scale or zero-heavy units: percent changes mean nothing, so YoY is a difference.
DIFFERENCE_UNITS = {"degC", "degC days"}
DAILY_WINDOW = "28D"
DAILY_MIN_POINTS = 14  # a 28-day window needs this many values (weekday series have ~20)

CATEGORY_ORDER = ["trade", "industry", "official", "energy", "weather", "market"]
CATEGORY_LABELS = {
    "trade": "Trade", "industry": "Industry", "official": "Official statistics",
    "energy": "Energy", "weather": "Weather", "market": "Markets",
}

# Official series drawn over an alternative-data panel in YoY mode (SIGNALS.md
# H1: port volumes vs official trade; H2: power vs industrial production).
# The first listed series with data for the country is used.
IP_SERIES = ["fred_indpro", "eurostat_ip", "oecd_ip", "oecd_ip_manufacturing"]
OFFICIAL_OVERLAY = {
    "portwatch_exports": ["comtrade_exports", "imf_exports", "comtrade_mirror_exports"],
    "portwatch_imports": ["comtrade_imports", "imf_imports", "comtrade_mirror_imports"],
    "ember_demand": IP_SERIES,
    "eia_us48_load": IP_SERIES,
    "eia_us48_overnight_load": IP_SERIES,
    "eia_industrial_sales": IP_SERIES,
    "entsoe_load": IP_SERIES,
    "entsoe_overnight_load": IP_SERIES,
    "neso_load": IP_SERIES,
    "neso_overnight_load": IP_SERIES,
}

# Reference palette (dataviz skill): slot 1 blue for the series, slot 2 orange
# for the official overlay; both read on light and dark backgrounds.
MAIN_COLOR = "#2a78d6"
RAW_COLOR = "rgba(42, 120, 214, 0.2)"
OFFICIAL_COLOR = "#eb6834"
HOLIDAY_FILL = "rgba(128, 128, 128, 0.18)"
BREAK_COLOR = "rgba(128, 128, 128, 0.95)"  # known structural breaks: dashed vertical lines (PLAN D41)
# Diverging pair for signed values (blue up, red down) around a neutral midpoint.
POSITIVE_COLOR, NEGATIVE_COLOR = "#2a78d6", "#e34948"
NEUTRAL_LIGHT, NEUTRAL_DARK = "#f0efec", "#383835"
BAND_FILL = "rgba(128, 128, 128, 0.25)"
BAND_MEAN_COLOR = "rgba(128, 128, 128, 0.9)"
PANEL_HEIGHT = 300  # total, so panels in a grid row line up
REMOVED_COLORS = ["rgba(128, 128, 128, 0.75)", "rgba(235, 104, 52, 0.6)", "rgba(128, 128, 128, 0.45)"]
Z_CLIP = 5.0  # charts clip robust z-scores at ±5 (SIGNALS.md §1)
BAND_STEPS = {"bk", "cf", "bwz", "bwc"}

LEVEL, YOY, CHAIN = "Level", "YoY", "Chain"
PERIODS = {"1 year": 1, "2 years": 2, "5 years": 5, "All": None}
DEFAULT_PERIOD = "5 years"
BAND_YEARS = 5
BAND_SMOOTH_DAYS = 7
REFERENCE_YEAR = 2001  # a non-leap year: seasonal charts put every year on its calendar


def uses_difference(unit: str) -> bool:
    """True if YoY for this unit is a difference rather than a percent change."""
    return unit in DIFFERENCE_UNITS


def daily_average(values: pd.Series) -> pd.Series:
    """28-day trailing average of a daily series indexed by date."""
    return values.sort_index().rolling(DAILY_WINDOW, min_periods=DAILY_MIN_POINTS).mean()


def yoy(values: pd.Series, frequency: str, unit: str, breaks: Sequence[pd.Timestamp] = ()) -> pd.Series:
    """Year-on-year change of one series indexed by date (period start).

    Daily data: the 28-day trailing average compared with the same window 364
    days earlier. Weekly: 364 days earlier. Monthly and quarterly: 12 months
    earlier. Returns percent (x100), or a plain difference for units in
    DIFFERENCE_UNITS. Missing comparison periods, percent changes from a
    zero or negative base, and comparisons across a known structural break
    (`breaks`, PLAN D41) give NaN.
    """
    s = values.sort_index().astype("float64")
    if frequency == "D":
        s = daily_average(s)
        prior_dates = s.index - pd.Timedelta(days=364)
        first, last = prior_dates - pd.Timedelta(days=27), s.index
    elif frequency == "W":
        prior_dates = s.index - pd.Timedelta(days=364)
        first, last = prior_dates, s.index + pd.Timedelta(days=6)
    else:
        prior_dates = s.index - pd.DateOffset(months=12)
        first = prior_dates
        last = s.index + pd.DateOffset(months=1 if frequency == "M" else 3) - pd.Timedelta(days=1)
    prior = pd.Series(s.reindex(prior_dates).to_numpy(), index=s.index)
    change = s - prior if uses_difference(unit) else (s / prior.where(prior > 0) - 1) * 100
    return change.where(tf.comparable(first, last, breaks))


def yoy_label(unit: str, frequency: str) -> str:
    """Axis label for a YoY panel."""
    base = f"YoY change ({unit})" if uses_difference(unit) else "YoY %"
    return f"{base}, 28-day avg" if frequency == "D" else base


def series(obs: pd.DataFrame, indicator: str, country: str) -> pd.Series:
    """One series from an observations frame (date, country, indicator, value), indexed by date."""
    rows = obs[(obs["indicator"] == indicator) & (obs["country"] == country)]
    return pd.Series(rows["value"].to_numpy(), index=pd.DatetimeIndex(rows["date"]), name=indicator).sort_index()


def display_series(
    values: pd.Series, frequency: str, unit: str, mode: str, with_raw: bool = True,
    breaks: Sequence[pd.Timestamp] = (),
) -> tuple[pd.Series, pd.Series | None, str]:
    """What a panel draws in the chosen mode: (main line, faint raw line or None, y-axis label).

    YoY mode: `yoy()` (no value across a known break). Level mode: the values, except daily
    series, whose main line is the 28-day average (with the raw daily values behind it if `with_raw`).
    """
    if mode == YOY:
        return yoy(values, frequency, unit, breaks), None, yoy_label(unit, frequency)
    if frequency == "D":
        return daily_average(values), values if with_raw else None, f"{unit}, 28-day avg"
    return values, None, unit


def every_seventh_day(values: pd.Series) -> pd.Series:
    """Every 7th point counted back from the last one: a smooth daily line
    (a 28-day average) drawn with a seventh of the points, ending on the latest day."""
    return values.iloc[::-1].iloc[::7].iloc[::-1]


def latest_yoy(obs: pd.DataFrame, frequency: str, unit: str,
               breaks: Mapping[str, Sequence[pd.Timestamp]] | None = None) -> pd.DataFrame:
    """Latest level and latest YoY per country of one indicator's observations.

    `breaks` gives each country's known structural breaks (no YoY across one).
    Returns one row per country (country, latest_date, latest_value, yoy_date,
    yoy), sorted by YoY, highest first; countries without a YoY value come last.
    """
    rows = []
    for country, part in obs.groupby("country", sort=True):
        values = pd.Series(part["value"].to_numpy(), index=pd.DatetimeIndex(part["date"])).sort_index()
        change = yoy(values, frequency, unit, (breaks or {}).get(country, ())).dropna()
        rows.append({
            "country": country,
            "latest_date": values.index.max(),
            "latest_value": values.iloc[-1],
            "yoy_date": change.index.max() if not change.empty else pd.NaT,
            "yoy": change.iloc[-1] if not change.empty else float("nan"),
        })
    columns = ["country", "latest_date", "latest_value", "yoy_date", "yoy"]
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values("yoy", ascending=False, na_position="last").reset_index(drop=True)


def seasonal_band(
    values: pd.Series, year: int, years: int = BAND_YEARS, smooth_days: int = BAND_SMOOTH_DAYS
) -> tuple[pd.DataFrame, list[int]]:
    """A daily series' `year` against the min-max range of the years before it.

    The series is smoothed with a trailing `smooth_days` average first (so
    early January uses late December). Every year is then put on the calendar
    of REFERENCE_YEAR (29 February is dropped).

    Returns:
        (band, band_years): band is indexed by reference date with columns
        min, max, mean (over the band years) and current (the chosen year);
        band_years are the years the band is built from (up to `years`, fewer
        when the series starts later).
    """
    s = values.sort_index().rolling(f"{smooth_days}D", min_periods=1).mean()
    s = s[~((s.index.month == 2) & (s.index.day == 29))]
    frame = pd.DataFrame({
        "value": s.to_numpy(),
        "year": s.index.year,
        "ref": pd.to_datetime({"year": REFERENCE_YEAR, "month": s.index.month, "day": s.index.day}),
    })
    band_years = sorted(y for y in frame["year"].unique() if year - years <= y < year)
    prior = frame[frame["year"].isin(band_years)]
    band = prior.groupby("ref")["value"].agg(["min", "max", "mean"])
    current = frame[frame["year"] == year].set_index("ref")["value"].rename("current")
    out = band.join(current, how="outer").sort_index()
    out.index.name = "ref"
    return out, [int(y) for y in band_years]


def band_position(band: pd.DataFrame) -> dict[str, object] | None:
    """Where the chosen year's latest value sits in the band (None if the year has no values).

    Returns ref (reference date), current, mean, min, max, vs_mean (percent
    difference from the band average) and position ("below range", "in range"
    or "above range").
    """
    current = band["current"].dropna()
    if current.empty:
        return None
    ref = current.index[-1]
    row = band.loc[ref]
    value, low, high, mean = row["current"], row["min"], row["max"], row["mean"]
    if pd.isna(low) or pd.isna(high):
        position = "no band"
    else:
        position = "below range" if value < low else "above range" if value > high else "in range"
    vs_mean = (value / mean - 1) * 100 if pd.notna(mean) and mean > 0 else float("nan")
    return {"ref": ref, "current": value, "mean": mean, "min": low, "max": high, "vs_mean": vs_mean,
            "position": position}


def band_figure(band: pd.DataFrame, title: str, subtitle: str, y_title: str, height: int = PANEL_HEIGHT) -> go.Figure:
    """The chosen year (blue line) over the earlier years' min-max band (grey) and their average (dashed)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=band.index, y=band["max"], mode="lines", line={"width": 0}, name="Band max",
                             hovertemplate="max %{y:,.4~g}"))
    fig.add_trace(go.Scatter(x=band.index, y=band["min"], mode="lines", line={"width": 0}, fill="tonexty",
                             fillcolor=BAND_FILL, name="Band min", hovertemplate="min %{y:,.4~g}"))
    fig.add_trace(go.Scatter(x=band.index, y=band["mean"], mode="lines", name="Average",
                             line={"color": BAND_MEAN_COLOR, "width": 1, "dash": "dash"},
                             hovertemplate="avg %{y:,.4~g}"))
    fig.add_trace(go.Scatter(x=band.index, y=band["current"], mode="lines", name="Selected year",
                             line={"color": MAIN_COLOR, "width": 2}, hovertemplate="%{y:,.4~g}"))
    fig.update_layout(
        title={"text": title, "font": {"size": 14}, "subtitle": {"text": subtitle, "font": {"size": 11}}},
        height=height, margin={"l": 10, "r": 10, "t": 58, "b": 10}, showlegend=False, hovermode="x unified",
        xaxis={"tickformat": "%b", "hoverformat": "%d %b",
               "range": [f"{REFERENCE_YEAR}-01-01", f"{REFERENCE_YEAR}-12-31"]},
        yaxis={"title": {"text": y_title, "font": {"size": 11}}},
    )
    return fig


def ranking_figure(labels: list[str], values: list[float], x_title: str) -> go.Figure:
    """Horizontal bars, one per country, blue above zero and red below; first label on top."""
    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker={"color": colors},
                           hovertemplate="%{y}: %{x:,.1f}<extra></extra>"))
    fig.update_layout(
        height=max(200, 26 * len(labels) + 60), margin={"l": 10, "r": 10, "t": 10, "b": 40},
        xaxis={"title": {"text": x_title, "font": {"size": 11}}, "zeroline": True},
        yaxis={"autorange": "reversed"}, bargap=0.25,
    )
    return fig


def yoy_heatmap(matrix: pd.DataFrame, dark: bool = False, limit: float = 30.0) -> go.Figure:
    """Rows x columns of YoY % with a diverging colour scale (red down, blue up), clipped at ±limit."""
    neutral = NEUTRAL_DARK if dark else NEUTRAL_LIGHT
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
        zmin=-limit, zmid=0, zmax=limit,
        colorscale=[[0, NEGATIVE_COLOR], [0.5, neutral], [1, POSITIVE_COLOR]],
        texttemplate="%{z:+.1f}", textfont={"size": 11, "color": "#ffffff" if dark else "#0b0b0b"},
        hovertemplate="%{y} · %{x}: %{z:+.1f}%<extra></extra>", xgap=2, ygap=2,
        colorbar={"title": {"text": "YoY %"}, "thickness": 12},
    ))
    fig.update_layout(height=max(250, 24 * len(matrix) + 80), margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis={"side": "top"}, yaxis={"autorange": "reversed"})
    return fig


def score_heatmap(matrix: pd.DataFrame, hover: pd.DataFrame, dark: bool = False, limit: float = 3.0) -> go.Figure:
    """Countries x score columns (robust z units) with a diverging scale (red weaker, blue stronger),
    clipped at ±limit; `hover` holds each cell's extra hover text (coverage and components)."""
    neutral = NEUTRAL_DARK if dark else NEUTRAL_LIGHT
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(dtype=float), x=list(matrix.columns), y=list(matrix.index),
        zmin=-limit, zmid=0, zmax=limit, customdata=hover.to_numpy(),
        colorscale=[[0, NEGATIVE_COLOR], [0.5, neutral], [1, POSITIVE_COLOR]],
        texttemplate="%{z:+.2f}", textfont={"size": 11, "color": "#ffffff" if dark else "#0b0b0b"},
        hovertemplate="<b>%{y}</b> · %{x}: %{z:+.2f}<br>%{customdata}<extra></extra>", xgap=2, ygap=2,
        colorbar={"title": {"text": "z"}, "thickness": 12},
    ))
    fig.update_layout(height=max(250, 26 * len(matrix) + 80), margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      xaxis={"side": "top"}, yaxis={"autorange": "reversed"})
    return fig


def period_windows(windows: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Holiday windows widened to whole periods for monthly/quarterly charts.

    Monthly values are plotted at the month start, so a window becomes the
    whole month (or months) it falls in: start of its first month to the start
    of the month after its last day. Daily and weekly windows cover their days
    (end + 1 day, so a one-day window has width).
    """
    if windows.empty:
        return windows
    out = windows.copy()
    if frequency in ("M", "Q"):
        freq = "M" if frequency == "M" else "Q"
        out["start"] = out["start"].dt.to_period(freq).dt.start_time
        out["end"] = (out["end"].dt.to_period(freq) + 1).dt.start_time
    else:
        out["end"] = out["end"] + pd.Timedelta(days=1)
    return out


def add_break_markers(
    fig: go.Figure,
    breaks: Sequence[tuple[pd.Timestamp, str]] | None,
    x_min: pd.Timestamp | None,
    x_max: pd.Timestamp | None,
    row: int | None = None,
) -> None:
    """A dashed vertical line at each known structural break between `x_min` and `x_max`,
    spanning the data drawn so far, with the break's date and reason on hover (PLAN D41).

    Drawn as a line trace rather than a layout shape, because shapes have no hover text.
    With `row` (subplots), only that row's traces set the height and the line goes there.
    """
    if not breaks or x_min is None or x_max is None:
        return
    axes = {None, "y"} if row in (None, 1) else {f"y{row}"}
    ys = [np.asarray(tr.y, dtype=float) for tr in fig.data if tr.y is not None and len(tr.y) and tr.yaxis in axes]
    ys = [y[np.isfinite(y)] for y in ys]
    ys = [y for y in ys if y.size]
    if not ys:
        return
    low, high = min(y.min() for y in ys), max(y.max() for y in ys)
    for date, reason in breaks:
        date = pd.Timestamp(date)
        if not x_min <= date <= x_max:
            continue
        trace = go.Scatter(x=[date, date], y=[low, high], mode="lines", name="Known break", showlegend=False,
                           line={"color": BREAK_COLOR, "width": 1.5, "dash": "dash"},
                           hovertemplate=f"Break {date:%Y-%m-%d}: {html.escape(reason)}<extra></extra>")
        if row is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=1)


def _holiday_shapes(windows: pd.DataFrame, x_min: pd.Timestamp, x_max: pd.Timestamp) -> list[dict]:
    shown = windows[(windows["end"] >= x_min) & (windows["start"] <= x_max)]
    return [
        {"type": "rect", "xref": "x", "yref": "paper", "x0": w.start, "x1": w.end, "y0": 0, "y1": 1,
         "fillcolor": HOLIDAY_FILL, "line": {"width": 0}, "layer": "below"}
        for w in shown.itertuples(index=False)
    ]


def panel_figure(
    main: pd.Series,
    title: str,
    y_title: str,
    subtitle: str = "",
    raw: pd.Series | None = None,
    official: pd.Series | None = None,
    official_name: str = "",
    windows: pd.DataFrame | None = None,
    zero_line: bool = False,
    height: int = PANEL_HEIGHT,
    breaks: Sequence[tuple[pd.Timestamp, str]] | None = None,
) -> go.Figure:
    """A small-multiple line chart for one series.

    Args:
        main: the series to draw (already in the chosen mode).
        title: panel title.
        y_title: y-axis label.
        subtitle: smaller line under the title (e.g. id and latest date).
        raw: optional raw daily values drawn faintly behind `main`.
        official: optional official series on the same axis (same units as `main`).
        official_name: legend name for `official`.
        windows: holiday windows (start, end, holiday) to shade.
        zero_line: draw a line at 0 (YoY panels).
        height: total figure height in pixels (the plot area shrinks to fit a legend).
        breaks: known structural breaks (date, reason), drawn as dashed vertical lines.
    """
    fig = go.Figure()
    if raw is not None and not raw.empty:
        fig.add_trace(go.Scatter(x=raw.index, y=raw.to_numpy(), name="Daily", mode="lines",
                                 line={"color": RAW_COLOR, "width": 0.8}, hovertemplate="%{y:,.4~g}"))
    fig.add_trace(go.Scatter(x=main.index, y=main.to_numpy(), name="28-day avg" if raw is not None else title,
                             mode="lines", line={"color": MAIN_COLOR, "width": 2}, hovertemplate="%{y:,.4~g}"))
    if official is not None and not official.empty:
        fig.add_trace(go.Scatter(x=official.index, y=official.to_numpy(), name=official_name, mode="lines+markers",
                                 line={"color": OFFICIAL_COLOR, "width": 2, "dash": "dot"}, marker={"size": 4},
                                 hovertemplate="%{y:,.4~g}"))
    shapes = []
    if windows is not None and not windows.empty and not main.empty:
        shapes = _holiday_shapes(windows, main.index.min(), main.index.max())
    if not main.empty:
        add_break_markers(fig, breaks, main.index.min(), main.index.max())
    if zero_line:
        shapes.append({"type": "line", "xref": "paper", "yref": "y", "x0": 0, "x1": 1, "y0": 0, "y1": 0,
                       "line": {"color": "rgba(128, 128, 128, 0.6)", "width": 1}})
    # Daily values and their average are one series (named in the page caption);
    # a legend is needed only when an official series is drawn too.
    legend = official is not None and not official.empty
    fig.update_layout(
        title={"text": title, "font": {"size": 14}, "subtitle": {"text": subtitle, "font": {"size": 11}}},
        height=height,
        margin={"l": 10, "r": 10, "t": 58 if subtitle else 40, "b": 55 if legend else 10},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "x": 0, "font": {"size": 11}},
        hovermode="x unified",
        shapes=shapes,
        yaxis={"title": {"text": y_title, "font": {"size": 11}}},
    )
    return fig


# ------------------------------------------------------------------ workbench


def chain_axis_label(chain: Chain, unit: str) -> str:
    """Y-axis label for a chain's output, from the steps that change the scale."""
    label = unit
    for step in chain.steps:
        key = step.key
        if key == "yoy":
            mode = step.params["mode"] or ("diff" if uses_difference(unit) or label != unit else "pct")
            label = "YoY %" if mode == "pct" else f"YoY change ({label})"
        elif key == "d1":
            label = f"change ({label})"
        elif key == "dlog":
            label = "log change x100"
        elif key in {"lin", "hp", "ham"} | BAND_STEPS:
            label = f"cycle ({label})"
        elif key == "log":
            label = f"log {label}"
        elif key == "idx":
            label = "index = 100"
        elif key == "rz":
            label = "robust z"
        elif key in ("pr", "prf"):
            label = "percentile rank"
    return label


def _line(x: pd.Index, y: np.ndarray, name: str, color: str, width: float, dash: str | None = None) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines", line={"color": color, "width": width, "dash": dash},
                      hovertemplate="%{y:,.4~g}")


def chain_figure(
    result: ChainResult,
    title: str,
    y_title: str,
    subtitle: str = "",
    windows: pd.DataFrame | None = None,
    start: pd.Timestamp | None = None,
    with_raw: bool = True,
    height: int = PANEL_HEIGHT,
    breaks: Sequence[tuple[pd.Timestamp, str]] | None = None,
) -> go.Figure:
    """A chain's output over the raw series, with its provisional end dashed.

    The raw series is drawn faintly behind the result when the chain keeps
    its units (and `with_raw`). Values of two-sided steps that may still
    revise are dashed. What each step removed (seasonal, trend, noise) is a
    faint line in a strip below. Robust z-scores are clipped at ±5. Only
    dates from `start` on are drawn. Known structural breaks are dashed vertical lines.
    """
    def clip(s: pd.Series) -> pd.Series:
        return s[s.index >= start] if start is not None else s

    values = clip(result.values)
    if result.chain.steps and result.chain.steps[-1].key == "rz":
        values = values.clip(-Z_CLIP, Z_CLIP)
    # A step that removed nothing here (e.g. `hol` without a holiday in range) gets no line.
    removed = [(name, clip(part)) for name, part in result.removed if part.abs().max() > 0]
    rows = 2 if removed else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.75, 0.25] if rows == 2 else [1.0])
    if with_raw and result.keeps_units and result.chain.steps:
        raw = clip(result.raw)
        fig.add_trace(_line(raw.index, raw.to_numpy(), "Raw", RAW_COLOR, 0.8), row=1, col=1)
    settled = values if result.provisional_from is None else values[values.index < result.provisional_from]
    fig.add_trace(_line(settled.index, settled.to_numpy(), str(result.chain) or "Raw", MAIN_COLOR, 2), row=1, col=1)
    if result.provisional_from is not None:
        first = settled.index[-1] if len(settled) else result.provisional_from
        tail = values[values.index >= first]
        fig.add_trace(_line(tail.index, tail.to_numpy(), "Provisional", MAIN_COLOR, 2, "dash"), row=1, col=1)
    if not values.empty:
        add_break_markers(fig, breaks, values.index.min(), values.index.max(), row=1)
    for n, (name, part) in enumerate(removed):
        fig.add_trace(_line(part.index, part.to_numpy(), f"Removed by {name}", REMOVED_COLORS[n % 3], 1),
                      row=2, col=1)
    shapes = []
    if windows is not None and not windows.empty and not values.empty:
        shapes = _holiday_shapes(windows, values.index.min(), values.index.max())
    if not result.keeps_units:
        shapes.append({"type": "line", "xref": "paper", "yref": "y", "x0": 0, "x1": 1, "y0": 0, "y1": 0,
                       "line": {"color": "rgba(128, 128, 128, 0.6)", "width": 1}})
    fig.update_layout(
        title={"text": title, "font": {"size": 14}, "subtitle": {"text": subtitle, "font": {"size": 11}}},
        height=height, margin={"l": 10, "r": 10, "t": 58 if subtitle else 40, "b": 10},
        showlegend=False, hovermode="x unified", shapes=shapes,
    )
    fig.update_yaxes(title={"text": y_title, "font": {"size": 11}}, row=1, col=1)
    if removed:
        fig.update_yaxes(title={"text": "removed", "font": {"size": 10}}, row=2, col=1)
    return fig


def periods_per_year(frequency: str, calendar: str = "all") -> float:
    """Observations per year of a regular series with this frequency."""
    if frequency == "D":
        return 261.0 if calendar == "weekdays" else 365.25
    return {"W": 365.25 / 7, "M": 12.0, "Q": 4.0}[frequency]


def periodogram(values: pd.Series, frequency: str, calendar: str = "all") -> pd.DataFrame:
    """Periodogram of a series (on a regular grid, linear trend removed, Hann window).

    Returns the cycle length in years (`period_years`, longest first) and
    each frequency's share of the total variance (`share`), so spectra of
    series in different units can share one axis. Empty for under 8 values.
    """
    grid = tf.regular(values, frequency, calendar)
    if len(grid) < 8:
        return pd.DataFrame({"period_years": [], "share": []})
    freqs, power = signal.periodogram(grid.to_numpy(), window="hann", detrend="linear", scaling="spectrum")
    keep = freqs > 0
    freqs, power = freqs[keep], power[keep]
    total = power.sum()
    share = power / total if total > 0 else power
    return pd.DataFrame({"period_years": 1 / (freqs * periods_per_year(frequency, calendar)), "share": share})


def band_years(chain: Chain) -> tuple[float, float] | None:
    """(shortest, longest) cycle in years of the chain's first band-pass step, if it has one."""
    for step in chain.steps:
        if step.key in BAND_STEPS:
            low = step.params["low"] or Length(1.5, "y")
            high = step.params["high"] or Length(8, "y")
            return low.years(), high.years()
    return None


def spectrum_figure(
    raw: pd.DataFrame,
    transformed: pd.DataFrame | None,
    title: str,
    subtitle: str = "",
    band: tuple[float, float] | None = None,
    height: int = PANEL_HEIGHT,
) -> go.Figure:
    """Periodograms (see `periodogram`) of the raw series (faint) and the chain's output (solid)
    against cycle length in years, both axes logarithmic; a band-pass's band is shaded."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw["period_years"], y=raw["share"], name="Raw", mode="lines",
                             line={"color": RAW_COLOR, "width": 1},
                             hovertemplate="%{x:.3~g} years: %{y:.2~%}<extra>raw</extra>"))
    if transformed is not None:
        fig.add_trace(go.Scatter(x=transformed["period_years"], y=transformed["share"], name="Transformed",
                                 mode="lines", line={"color": MAIN_COLOR, "width": 1.5},
                                 hovertemplate="%{x:.3~g} years: %{y:.2~%}<extra>transformed</extra>"))
    shapes = []
    if band is not None:
        shapes.append({"type": "rect", "xref": "x", "yref": "paper", "x0": float(np.log10(band[0])),
                       "x1": float(np.log10(band[1])), "y0": 0, "y1": 1, "fillcolor": HOLIDAY_FILL,
                       "line": {"width": 0}, "layer": "below"})
    fig.update_layout(
        title={"text": title, "font": {"size": 14}, "subtitle": {"text": subtitle, "font": {"size": 11}}},
        height=height, margin={"l": 10, "r": 10, "t": 58 if subtitle else 40, "b": 10}, showlegend=False,
        shapes=shapes,
        xaxis={"type": "log", "autorange": "reversed",
               "title": {"text": "cycle length (years)", "font": {"size": 11}}},
        yaxis={"type": "log", "exponentformat": "power", "title": {"text": "share of variance", "font": {"size": 11}}},
    )
    return fig
