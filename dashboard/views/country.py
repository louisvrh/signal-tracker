"""Country page: every stored indicator of one country as small multiples.

Level, YoY or Chain view. In YoY view the matching official series is drawn
over the alternative-data panels (charts.OFFICIAL_OVERLAY). In Chain view
every panel shows the transform chain built in the sidebar
(dashboard/workbench.py). Moving holidays are shaded. Open a country directly
with `?country=DEU`.

Replay (PLAN D43): with the switch on, the page shows the data exactly as it
was known at the end of the chosen day (`store.latest(as_of=...)`), and chains
keep only their one-sided steps. The Level and YoY views use only past data
(trailing averages, a year earlier), so they need no change.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, data, widgets, workbench
from tracker.registry import Indicator
from tracker.signals.moving_holidays import LABELS as HOLIDAY_LABELS

GRID_COLUMNS = 3

reg = data.registry()
countries = reg.watchlist()

st.title("Country")
c1, c2, c3 = st.columns([2, 1, 1])
country = widgets.url_selectbox(c1, "Country", countries, key="country", default=countries[0],
                                format_func=lambda c: f"{reg.countries[c].name} ({c})")
mode, years = widgets.mode_and_period(c2, c3)

obs = data.observations(countries=[country])
ref = reg.countries[country]
if obs.empty:
    st.info(f"No data stored for {ref.name} yet.")
    st.stop()

replay = widgets.replay_date(obs["date"].min().date(), pd.Timestamp.now(tz="UTC").date())
wb = None
if mode == charts.CHAIN:
    wb = workbench.sidebar(signal=f"[indicator] {country}", realtime_forced="replay is on" if replay else None)
if replay is not None:
    st.warning(f"**Replay: {ref.name} as known at the end of {replay.day} {replay:%B %Y} (UTC).** "
               "Values published later and later revisions are hidden; chains use one-sided steps only. Most history "
               "was downloaded in September 2026, so before then \"known\" rests on estimated publication dates "
               "(period end + the usual lag), and revisions made before our first download can't be replayed "
               "(PLAN D43).",
               icon=":material/history:")
    obs = data.observations_as_of([country], replay)
    if obs.empty:
        st.info(f"Nothing about {ref.name} had been published by {replay.day} {replay:%B %Y}.")
        st.stop()

end = pd.Timestamp(replay) if replay is not None else obs["date"].max()
start = widgets.period_start(obs["date"].min(), end, years)
windows = data.holiday_windows(country, start.year, end.year)
present = set(obs["indicator"].unique())

notes = []
if wb is not None:
    notes.append(workbench.caption(wb))
elif mode == charts.LEVEL:
    notes.append("Daily series: faint line = daily values, solid line = 28-day average.")
else:
    notes.append("Daily series: 28-day average vs the same 28 days a year (364 days) earlier.")
if ref.moving_holidays:
    names = ", ".join(HOLIDAY_LABELS[h] for h in ref.moving_holidays)
    notes.append(f"Shaded: {names} (whole months on monthly charts).")
if mode == charts.YOY:
    notes.append("Dotted orange line: the official series for the same concept (trade or industrial production), "
                 "YoY %.")
if any(start <= date <= end for i in present for date in data.break_dates(i, country)):
    notes.append("Dashed vertical line: a known structural break (a reporting change); YoY and robust z-scores "
                 "don't compare across it. Hover it for the reason.")
if ref.exclude_from_signals:
    notes.append(f"{ref.name}'s {' and '.join(g.replace('_', ' ') for g in ref.exclude_from_signals)} data is shown "
                 "here but excluded from signals (PLAN D22).")
if notes:
    st.caption(" ".join(notes))


def official_overlay(indicator: Indicator) -> tuple[pd.Series | None, str]:
    """YoY of the first official counterpart with data for this country."""
    for official_id in charts.OFFICIAL_OVERLAY.get(indicator.id, []):
        if official_id in present:
            off = reg.indicator(official_id)
            values = charts.series(obs, official_id, country)
            return charts.yoy(values, off.frequency, off.unit, data.break_dates(official_id, country)), off.name
    return None, ""


def clip(s: pd.Series | None) -> pd.Series | None:
    return None if s is None else s[s.index >= start]


problems: dict[str, str] = {}
converted: dict[str, list] = {}


def panel(indicator: Indicator):
    """The chart for one indicator in the chosen mode."""
    values = charts.series(obs, indicator.id, country)
    subtitle = f"{indicator.id} · {indicator.unit} · latest {values.index.max():%Y-%m-%d}"
    if replay is not None:
        subtitle += f" · as known {replay:%Y-%m-%d}"
    if wb is not None:
        fig, problem, conversions = workbench.panel(wb, values, indicator.id, country, indicator.name, subtitle,
                                                    start, end, windows, replay_day=replay)
        if problem:
            problems[indicator.name] = problem
        if conversions:
            converted[indicator.name] = conversions
        return fig
    main, raw, y_title = charts.display_series(values, indicator.frequency, indicator.unit, mode,
                                               breaks=data.break_dates(indicator.id, country))
    official, official_name = official_overlay(indicator) if mode == charts.YOY else (None, "")
    fig = charts.panel_figure(
        clip(main), indicator.name, y_title, subtitle=subtitle, raw=clip(raw), official=clip(official),
        official_name=official_name, windows=charts.period_windows(windows, indicator.frequency),
        zero_line=mode == charts.YOY, breaks=data.breaks(indicator.id, country),
    )
    fig.update_xaxes(range=[start, end])
    return fig


for category in charts.CATEGORY_ORDER:
    shown = [i for i in reg.indicators.values() if i.category == category and i.id in present]
    if not shown:
        continue
    st.subheader(charts.CATEGORY_LABELS[category])
    for row in range(0, len(shown), GRID_COLUMNS):
        for col, indicator in zip(st.columns(GRID_COLUMNS), shown[row:row + GRID_COLUMNS]):
            with col:
                st.plotly_chart(panel(indicator), key=f"{country}-{indicator.id}", config={"displayModeBar": False})

workbench.problems_note(problems)
workbench.conversions_note(converted)
missing = [i.id for i in reg.indicators.values() if country in reg.countries_for(i) and i.id not in present]
if missing:
    what = f"nothing published by {replay:%Y-%m-%d}" if replay is not None else "no data stored"
    st.caption(f"Registered for {country} but {what}: {', '.join(missing)}.")
