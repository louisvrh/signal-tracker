"""Overview page (S20, the landing page): composite scores, this week's movers and active divergence flags.

Everything comes from the latest `python -m tracker signals` run
(data/signals/signals.parquet and flags.parquet, PLAN D46), through the same
summary functions the Telegram digest uses (tracker/daily_signals.py), so the
numbers here are the signals output. Each flag links to its Country page.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard import charts, data
from tracker import daily_signals as ds

COUNTRY_PAGE = "views/country.py"
CATEGORY_COLUMNS = {"trade": "Trade", "energy": "Energy", "activity": "Score", "activity_z": "Score z"}
DIVERGENCE_NAMES = {"d_mkt": "Market divergence", "d_off": "Official divergence"}

reg = data.registry()

# Old links opened a country with /?country=DEU when Country was the landing page: keep them working.
if st.session_state.get("first_run") and st.query_params.get("country") in reg.countries:
    st.switch_page(COUNTRY_PAGE, query_params={"country": st.query_params["country"]})

st.title("Overview")
signals = data.signals()
flags = data.signal_flags()
if signals.empty:
    st.info("No signals yet. They are computed by `python -m tracker signals`, a step of the daily run.")
    st.stop()

week = pd.Timestamp(signals["date"].max())
computed = pd.Timestamp(signals["computed_at"].max())
scores = ds.latest_scores(signals)
active = ds.active_flags(flags, signals)
alerts = ds.anomalies(signals)
st.caption(f"Week ending {week:%a %d %b %Y}, computed {computed:%d %b %H:%M} UTC by `python -m tracker signals`. "
           "Scores average the robust z-scores of the YoY changes in port volumes (trade) and weather-normalised "
           "power demand (energy); negative means weaker than usual for that country (PLAN §5.5, D46).")

with st.container(horizontal=True):
    st.metric("Countries scored", f"{scores['activity'].notna().sum()} / {len(reg.watchlist())}", border=True)
    st.metric("Market flags", int((active["signal"] == "d_mkt").sum()), border=True,
              help="D_mkt below -1.5 for 3 weeks: activity weak, the country ETF hasn't reacted.")
    st.metric("Official flags", int((active["signal"] == "d_off").sum()), border=True,
              help="D_off below -1.5 for 3 weeks: activity weak, the latest official print hasn't shown it.")
    st.metric("Anomalies", len(alerts), border=True, help="Components with |robust z| above 2 this week.")


def hover(country: str, column: str) -> str:
    """Hover text of one heatmap cell: coverage and the components behind it."""
    row = scores.loc[country]
    if column == "trade":
        text = f"{row['n_trade']} trade component(s)"
    elif column == "energy":
        text = f"{row['n_energy']} energy component(s)"
    else:
        text = f"coverage {row['coverage']} component(s)"
    comps = row["components"] if isinstance(row["components"], dict) else {}
    want = {"trade": "trade", "energy": "energy"}.get(column)
    lines = [f"{k}: {v['z']:+.2f}" + (" (half weight, holiday)" if v["w"] < 1 else "") + f", to {v['date']}"
             for k, v in comps.items() if want is None or v["cat"] == want]
    return "<br>".join([text, *lines])


left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Scores by country")
    matrix = scores[list(CATEGORY_COLUMNS)].rename(columns=CATEGORY_COLUMNS)
    hover_text = pd.DataFrame({CATEGORY_COLUMNS[c]: [hover(k, c) for k in scores.index] for c in CATEGORY_COLUMNS},
                              index=scores.index)
    labels = [f"{reg.countries[c].name} ({c})" for c in scores.index]
    matrix.index = hover_text.index = labels
    dark = getattr(st.context.theme, "type", None) == "dark"
    st.plotly_chart(charts.score_heatmap(matrix, hover_text, dark=dark), key="score-heatmap",
                    config={"displayModeBar": False})
    st.caption("Robust z units, colours stop at ±3. Sorted weakest first. Score needs at least 2 components; "
               "Score z is the score against its own last 3 years, which the divergences use. Hover a cell for "
               "its components.")

with right:
    st.subheader("Movers this week")
    moved = ds.movers(signals)
    if moved.empty:
        st.caption("No score a week earlier to compare with.")
    else:
        moved["name"] = [f"{reg.countries[c].name} ({c})" for c in moved["country"]]
        st.dataframe(moved, hide_index=True, column_order=["name", "now", "week_before", "change"],
                     column_config={"name": "Country",
                                    "now": st.column_config.NumberColumn("Score", format="%+.2f"),
                                    "week_before": st.column_config.NumberColumn("Week before", format="%+.2f"),
                                    "change": st.column_config.NumberColumn("Change", format="%+.2f")})

    st.subheader("Active divergence flags")
    if active.empty:
        st.caption("No divergence flag is on.")
    for r in active.itertuples():
        info = json.loads(r.components) if isinstance(r.components, str) else {}
        with st.container(border=True):
            st.page_link(COUNTRY_PAGE, label=f"{reg.countries[r.country].name} ({r.country}): "
                                             f"{DIVERGENCE_NAMES[r.signal].lower()} {r.value:+.2f}",
                         icon=":material/flag:", query_params={"country": r.country})
            other = (f"{info.get('etf', 'ETF')} vs URTH, z {info.get('market_z', float('nan')):+.2f}"
                     if r.signal == "d_mkt" else
                     f"{info.get('official', 'official')}, z {info.get('official_z', float('nan')):+.2f}")
            st.caption(f"On since {pd.Timestamp(r.since):%d %b %Y} ({r.weeks} week{'s' if r.weeks != 1 else ''}). "
                       f"Activity z {info.get('activity_z', float('nan')):+.2f}; {other}.")

    if not alerts.empty:
        with st.expander(f"Anomalies: {len(alerts)} component(s) with |z| > {ds.ANOMALY_Z:g}"):
            alerts["name"] = [f"{reg.countries[c].name} ({c})" for c in alerts["country"]]
            st.dataframe(alerts, hide_index=True, column_order=["name", "indicator", "z"],
                         column_config={"name": "Country", "indicator": "Indicator",
                                        "z": st.column_config.NumberColumn("Robust z", format="%+.2f")})
