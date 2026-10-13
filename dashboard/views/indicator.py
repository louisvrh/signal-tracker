"""Indicator page: one indicator across countries, with a ranking of the latest YoY values.

Level, YoY or Chain view for the small multiples; the Chain view draws the
transform chain built in the sidebar (dashboard/workbench.py). Open an
indicator directly with `?indicator=portwatch_exports`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, data, widgets, workbench

DEFAULT_INDICATOR = "portwatch_exports"
GRID_COLUMNS = 4
SMALL_PANEL_HEIGHT = 260

reg = data.registry()
st.title("Indicator")

available = data.indicators_with_data()
if not available:
    st.info("No data stored yet.")
    st.stop()

registry_order = list(reg.indicators)
options = sorted(available, key=lambda i: (charts.CATEGORY_ORDER.index(reg.indicator(i).category),
                                           registry_order.index(i)))


def indicator_label(indicator_id: str) -> str:
    ind = reg.indicator(indicator_id)
    return f"{charts.CATEGORY_LABELS[ind.category]} · {ind.name} ({indicator_id})"


c1, c2, c3 = st.columns([2, 1, 1])
indicator_id = widgets.url_selectbox(c1, "Indicator", options, key="indicator", default=DEFAULT_INDICATOR,
                                     format_func=indicator_label)
mode, years = widgets.mode_and_period(c2, c3)
wb = workbench.sidebar(signal=f"{indicator_id} [country]") if mode == charts.CHAIN else None
ind = reg.indicator(indicator_id)
obs = data.observations(indicators=[indicator_id])

about = [ind.description or ind.name, f"Unit: {ind.unit}.", f"Frequency: {ind.frequency}."]
if ind.signal_group:
    excluded = [c for c in reg.countries_for(ind) if reg.is_excluded(ind, c)]
    if excluded:
        about.append(f"Shown but excluded from signals: {', '.join(excluded)} (PLAN D22).")
if ind.breaks:
    about.append("Known structural breaks are dashed vertical lines where they fall in the period shown (hover "
                 "for the reason); YoY and robust z-scores don't compare across them.")
st.caption(" ".join(about))

# ---------------------------------------------------------------- ranking
yoy_title = charts.yoy_label(ind.unit, ind.frequency)
st.subheader("Latest YoY")
table = data.latest_yoy(indicator_id)
table = table.assign(name=table["country"].map(data.place_name))
ranked = table.dropna(subset=["yoy"])
table["rank"] = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
table["note"] = [
    "excluded from signals" if c in reg.countries and reg.is_excluded(ind, c) else "" for c in table["country"]
]
if ind.frequency == "D":
    st.caption("Latest 28-day average against the same 28 days a year (364 days) earlier. Dates differ where "
               "sources lag; check the \"as of\" column before comparing.")
else:
    st.caption("Latest period against the same period a year earlier. Countries publish with different lags; check "
               "the \"as of\" column before comparing.")

left, right = st.columns([1, 1])
if ranked.empty:
    left.info("Not enough history for a YoY value yet.")
else:
    left.plotly_chart(
        charts.ranking_figure([f"{n} ({c})" for n, c in zip(ranked["name"], ranked["country"])],
                              ranked["yoy"].tolist(), yoy_title),
        key=f"rank-{indicator_id}", config={"displayModeBar": False},
    )
value_format = "%+.2f" if charts.uses_difference(ind.unit) else "%+.1f%%"
right.dataframe(
    table,
    hide_index=True,
    height=min(35 * (len(table) + 1) + 3, 900),
    column_order=["rank", "name", "country", "yoy", "yoy_date", "latest_value", "latest_date", "note"],
    column_config={
        "rank": st.column_config.NumberColumn("#", format="%d"),
        "name": "Name", "country": "Code",
        "yoy": st.column_config.NumberColumn(yoy_title, format=value_format),
        "yoy_date": st.column_config.DateColumn("As of"),
        "latest_value": st.column_config.NumberColumn(f"Latest ({ind.unit})", format="%.4g"),
        "latest_date": st.column_config.DateColumn("Latest date"),
        "note": "Note",
    },
)

# ---------------------------------------------------------------- small multiples
st.subheader("By country")
if wb is not None:
    st.caption(workbench.caption(wb) + (" Daily series are drawn without the raw values." if ind.frequency == "D"
                                        else ""))
elif ind.frequency == "D":
    st.caption("Daily series are drawn as 28-day averages, one point a week. Moving holidays are shaded.")
end = obs["date"].max()
start = widgets.period_start(obs["date"].min(), end, years)
place_order = reg.watchlist() + list(reg.chokepoints) + ["WLD"]
present = set(obs["country"].unique())
codes = [c for c in place_order if c in present] + sorted(present - set(place_order))
problems: dict[str, str] = {}
converted: dict[str, list] = {}
for row in range(0, len(codes), GRID_COLUMNS):
    for col, code in zip(st.columns(GRID_COLUMNS), codes[row:row + GRID_COLUMNS]):
        values = charts.series(obs, indicator_id, code)
        windows = data.holiday_windows(code, start.year, end.year) if code in reg.countries else None
        if wb is not None:
            fig, problem, conversions = workbench.panel(
                wb, values, indicator_id, code, data.place_name(code), f"{code} · latest {values.index.max():%Y-%m-%d}",
                start, end, windows, with_raw=ind.frequency != "D", height=SMALL_PANEL_HEIGHT)
            if problem:
                problems[data.place_name(code)] = problem
            if conversions:
                converted[data.place_name(code)] = conversions
            with col:
                st.plotly_chart(fig, key=f"{indicator_id}-{code}", config={"displayModeBar": False})
            continue
        main, _, y_title = charts.display_series(values, ind.frequency, ind.unit, mode, with_raw=False,
                                                 breaks=data.break_dates(indicator_id, code))
        if ind.frequency == "D":
            main = charts.every_seventh_day(main.dropna())  # 25 panels: keep the page light
        fig = charts.panel_figure(
            main[main.index >= start], data.place_name(code), y_title,
            subtitle=f"{code} · latest {values.index.max():%Y-%m-%d}",
            windows=charts.period_windows(windows, ind.frequency) if windows is not None else None,
            zero_line=mode == charts.YOY, height=SMALL_PANEL_HEIGHT, breaks=data.breaks(indicator_id, code),
        )
        fig.update_xaxes(range=[start, end])
        with col:
            st.plotly_chart(fig, key=f"{indicator_id}-{code}", config={"displayModeBar": False})
workbench.problems_note(problems)
workbench.conversions_note(converted)
