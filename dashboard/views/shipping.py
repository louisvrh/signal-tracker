"""Shipping & Chokepoints page: chokepoint transits against their 5-year range, and port activity by country."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, data, widgets

TRANSIT_IDS = ["portwatch_transits", "portwatch_tanker_transits", "portwatch_transit_volume"]
PORT_IDS = ["portwatch_portcalls", "portwatch_imports", "portwatch_exports"]
PORT_LABELS = {"portwatch_portcalls": "Port calls", "portwatch_imports": "Imports", "portwatch_exports": "Exports"}
GRID_COLUMNS = 3
PORT_PANEL_HEIGHT = 240

reg = data.registry()
st.title("Shipping & Chokepoints")

transits = data.observations(indicators=TRANSIT_IDS)
ports = data.observations(indicators=PORT_IDS)
if transits.empty and ports.empty:
    st.info("No PortWatch data stored yet.")
    st.stop()

dates = pd.concat([transits["date"], ports["date"]])
all_years = sorted(dates.dt.year.unique())
year_options = [int(y) for y in reversed(all_years[1:])] or [int(all_years[0])]
c1, c2 = st.columns([2, 1])
measure = c1.selectbox("Chokepoint measure", TRANSIT_IDS, format_func=lambda i: reg.indicator(i).name)
year = c2.selectbox("Year", year_options, help="Drawn against the min-max range of the 5 years before it.")


def band_caption(band_years: list[int]) -> str:
    span = f"{band_years[0]}–{band_years[-1]}" if band_years else "no earlier years"
    return (f"Blue: {year}, {charts.BAND_SMOOTH_DAYS}-day average. Grey band: lowest to highest of {span} on the same "
            f"calendar day; dashed line: their average.")


# ------------------------------------------------------------- chokepoints
st.subheader("Chokepoint transits")
chk = transits[transits["indicator"] == measure]
unit = reg.indicator(measure).unit
if chk.empty:
    st.info("No chokepoint data stored for this measure.")
else:
    stored = set(chk["country"].unique())
    codes = [c for c in reg.chokepoints if c in stored]
    bands, summary = {}, []
    for code in codes:
        band, band_years = charts.seasonal_band(charts.series(chk, measure, code), year)
        bands[code] = band
        pos = charts.band_position(band)
        if pos is not None:
            summary.append({"chokepoint": data.place_name(code), **pos})
    st.caption(band_caption(band_years))
    if summary:
        table = pd.DataFrame(summary)
        table["date"] = [pd.Timestamp(year=year, month=r.month, day=r.day) for r in table["ref"]]
        st.dataframe(
            table,
            hide_index=True,
            height=35 * (len(table) + 1) + 3,  # every row, no inner scrolling
            column_order=["chokepoint", "date", "current", "mean", "vs_mean", "min", "max", "position"],
            column_config={
                "chokepoint": "Chokepoint",
                "date": st.column_config.DateColumn("Latest day"),
                "current": st.column_config.NumberColumn(f"{year} ({unit}, 7-day avg)", format="%.4g"),
                "mean": st.column_config.NumberColumn("5-year average", format="%.4g"),
                "vs_mean": st.column_config.NumberColumn("vs average", format="%+.1f%%"),
                "min": st.column_config.NumberColumn("5-year min", format="%.4g"),
                "max": st.column_config.NumberColumn("5-year max", format="%.4g"),
                "position": "Position",
            },
        )
    for row in range(0, len(codes), GRID_COLUMNS):
        for col, code in zip(st.columns(GRID_COLUMNS), codes[row:row + GRID_COLUMNS]):
            fig = charts.band_figure(bands[code], data.place_name(code), f"{code} · {year}", unit)
            with col:
                st.plotly_chart(fig, key=f"band-{measure}-{code}-{year}", config={"displayModeBar": False})

# ------------------------------------------------------------- ports
st.subheader("Port activity by country")
if ports.empty:
    st.info("No port data stored yet.")
    st.stop()

yoy_columns = {}
as_of = []
for indicator_id in PORT_IDS:
    latest = data.latest_yoy(indicator_id).set_index("country")
    yoy_columns[PORT_LABELS[indicator_id]] = latest["yoy"]
    as_of.append(latest["yoy_date"])
matrix = pd.DataFrame(yoy_columns)
matrix = matrix.sort_values(PORT_LABELS[PORT_IDS[0]], ascending=False, na_position="last")
matrix.index = [f"{data.place_name(c)} ({c})" for c in matrix.index]
dates_as_of = pd.concat(as_of).dropna()

left, right = st.columns([1, 1])
with left:
    st.caption(f"Latest 28-day total against the same 28 days a year (364 days) earlier, %, as of "
               f"{dates_as_of.min():%d %b %Y}" + ("" if dates_as_of.min() == dates_as_of.max() else
                                                  f" to {dates_as_of.max():%d %b %Y}") + ". Colours stop at ±30%.")
    dark = getattr(st.context.theme, "type", None) == "dark"
    st.plotly_chart(charts.yoy_heatmap(matrix, dark=dark), key="port-heatmap", config={"displayModeBar": False})
with right:
    stored = set(ports["country"].unique())
    port_countries = [c for c in reg.watchlist() if c in stored]
    country = widgets.url_selectbox(st, "Country", port_countries, key="country", default=port_countries[0],
                                    format_func=lambda c: f"{reg.countries[c].name} ({c})")
    st.caption(band_caption(charts.seasonal_band(charts.series(ports, PORT_IDS[0], country), year)[1]))
    for indicator_id in PORT_IDS:
        values = charts.series(ports, indicator_id, country)
        if values.empty:
            continue
        band, _ = charts.seasonal_band(values, year)
        fig = charts.band_figure(band, PORT_LABELS[indicator_id], f"{country} · {year}",
                                 reg.indicator(indicator_id).unit, height=PORT_PANEL_HEIGHT)
        st.plotly_chart(fig, key=f"port-{indicator_id}-{country}-{year}", config={"displayModeBar": False})
